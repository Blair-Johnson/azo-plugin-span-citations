"""Exact, immutable file-span citations for Agent Zoo (stdlib-only)."""
from __future__ import annotations

import codecs
from collections import deque
from concurrent.futures import Future, TimeoutError
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import uuid

from agent_utils import Feature, Tool
from agent_utils.components import (
    BufferManagerProvider, HarnessEventHandler, InterruptCheck, ToolDispatchStart,
)
from agent_utils.session_cwd import get_session_cwd
from agent_utils.tools.buffers import CloseBuffer

PREFIX = "span_"
WAIT_SECONDS = 0.5
CHUNK_BYTES = 1024 * 1024
MAX_ANCHOR_BYTES = 64 * 1024
MAX_SPAN_BYTES = 16 * 1024 * 1024
MAX_PENDING = 4


class ClipError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _extremes(data, anchor, base, floor, positions, first):
    """Keep the first/last two occurrences, including overlapping matches."""
    start = max(0, floor - base)
    if first:
        while len(positions) < 2:
            index = data.find(anchor, start)
            if index < 0:
                break
            positions.append(base + index)
            start = index + 1
    else:
        end = len(data)
        found = []
        for _ in range(2):
            index = data.rfind(anchor, start, end)
            if index < 0:
                break
            found.append(base + index)
            end = index + len(anchor) - 1
        positions.extend(reversed(found))


def _scan_snapshot(source, snapshot, prefix, suffix):
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    starts, ends = [], deque(maxlen=2)
    tail, total = b"", 0
    overlap = max(len(prefix), len(suffix)) - 1
    floors = [0, 0]
    while chunk := source.read(CHUNK_BYTES):
        digest.update(chunk)
        decoder.decode(chunk)
        snapshot.write(chunk)
        data, base = tail + chunk, total - len(tail)
        for i, (anchor, positions) in enumerate(((prefix, starts), (suffix, ends))):
            _extremes(data, anchor, base, floors[i], positions, i == 0)
            floors[i] = max(0, base + len(data) - len(anchor) + 1)
        total += len(chunk)
        tail = data[-overlap:] if overlap else b""
    decoder.decode(b"", final=True)
    minimum_gap = max(0, len(prefix) - len(suffix))
    if not starts or not ends or ends[-1] < starts[0] + minimum_gap:
        raise ClipError("no_match", "No span matches both anchors in order.")
    if ((len(starts) > 1 and ends[-1] >= starts[1] + minimum_gap)
            or (len(ends) > 1 and ends[-2] >= starts[0] + minimum_gap)):
        raise ClipError("ambiguous", "More than one span matches; use more specific anchors.")
    return starts[0], ends[-1] + len(suffix), total, digest.hexdigest()


def extract_citation(path, prefix, suffix, cwd):
    """Read one byte snapshot; never reread the live file to extract its quote."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(cwd) / resolved
    resolved = resolved.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(resolved, flags), "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ClipError("invalid_file", "Only regular UTF-8 text files are supported.")
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            start, end, size, digest = _scan_snapshot(
                source, snapshot, prefix.encode("utf-8"), suffix.encode("utf-8"))
            if (_signature(before) != _signature(os.fstat(source.fileno()))
                    or _signature(before) != _signature(resolved.stat())
                    or size != before.st_size):
                raise ClipError("file_changed", "File changed while reading; retry against a stable file.")
            if end - start > MAX_SPAN_BYTES:
                raise ClipError("span_too_large", f"Span exceeds {MAX_SPAN_BYTES} bytes; narrow the anchors.")
            snapshot.seek(0)
            remaining, char_start, line_start = start, 0, 1
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            while remaining:
                chunk = snapshot.read(min(CHUNK_BYTES, remaining))
                char_start += len(decoder.decode(chunk))
                line_start += chunk.count(b"\n")
                remaining -= len(chunk)
            char_start += len(decoder.decode(b"", final=True))
            text = snapshot.read(end - start).decode("utf-8")
    return {
        "schema": "azo.span-citation", "version": 1,
        "file": {"path": str(resolved), "encoding": "utf-8", "size_bytes": size,
                 "hash": {"algorithm": "sha256", "hex": digest}},
        "match": {"mode": "exact", "prefix": prefix, "suffix": suffix,
                  "anchors_included": True},
        "span": {"text": text, "byte_start": start, "byte_end": end,
                 "char_start": char_start, "char_end": char_start + len(text),
                 "line_start": line_start,
                 "line_end": line_start + text[:-1].count("\n")},
    }


def _worker(path, prefix, suffix, cwd):
    try:
        return {"status": "complete", "citation_json": encode(extract_citation(path, prefix, suffix, cwd))}
    except ClipError as exc:
        code = exc.code
        message = str(exc)
    except UnicodeError:
        code, message = "invalid_utf8", "File and anchors must be valid UTF-8 text."
    except OSError as exc:
        code, message = "io_error", str(exc)
    except Exception as exc:
        code, message = "internal_error", f"{type(exc).__name__}: {exc}"
    return {"status": "error", "code": code, "message": message}


def citation_provider(state, buffer_id):
    return getattr(state, "span_citation_records", {}).get(buffer_id)


def bind_namespace(state):
    state.buffer_manager.register_special_buffer_namespace(PREFIX, citation_provider, replace=True)


class ClipService:
    """Runtime-only futures. Workers never access mutable harness state."""
    def __init__(self):
        self.futures = {}
        self.enqueued = set()

    def start(self, job_id, path, prefix, suffix, cwd):
        future = Future()
        self.futures[job_id] = future

        def run():
            future.set_result(_worker(path, prefix, suffix, cwd))

        try:
            threading.Thread(target=run, name=f"clip-{job_id}", daemon=True).start()
        except Exception:
            self.futures.pop(job_id, None)
            raise
        return future

    def finish(self, state, job_id, result):
        state.span_citation_pending.pop(job_id, None)
        self.futures.pop(job_id, None)
        if result["status"] == "error":
            return {"job_id": job_id, **result}
        buffer_id = PREFIX + uuid.uuid4().hex
        state.span_citation_records[buffer_id] = result["citation_json"]
        return {"status": "complete", "job_id": job_id, "buffer_id": buffer_id,
                "readonly": True, "message": "Use view(buffer=...) to read; close_buffer(buffer=...) to release."}


class SpanSetup:
    reads = {"buffer_manager"}
    writes = {"span_citation_records", "span_citation_pending"}
    init = {"span_citation_records": dict, "span_citation_pending": dict, "span_citation_notices": dict}

    def __call__(self, state):
        bind_namespace(state)
        return state


class SpanStartup(HarnessEventHandler):
    event_kinds = {"runtime.session_ready"}
    optional_reads = {"buffer_manager"}
    init = SpanSetup.init

    def handle_harness_event(self, event, state):
        if getattr(state, "buffer_manager", None) is not None:
            bind_namespace(state)
        return None

class ClipSpan(Tool):
    reads = {"buffer_manager", "span_citation_records", "span_citation_pending"}
    writes = {"span_citation_records", "span_citation_pending"}
    optional_reads = {"session_cwd", "terminal_cwd"}
    group = "Span citations"

    def __init__(self, service):
        super().__init__()
        self.service = service

    @staticmethod
    def fn(path: str, prefix: str, suffix: str) -> str:
        """Clip an exact, uniquely anchored span from a UTF-8 file into immutable JSON.

        Includes both anchors, preserving all source bytes and whitespace. Anchors
        may overlap. Ambiguous matches fail; no normalization or fuzzy matching.
        The JSON records absolute path, whole-file SHA-256, text, and zero-based
        half-open byte/character offsets. Maximum span is 16 MiB, anchors 64 KiB.
        Waits up to 0.5 seconds, then returns a pending job ID if unfinished;
        completion follows as an interrupt. Read the resulting span_* ID with view/grep;
        release it with close_buffer(buffer=...). Copies are NOT citation IDs.

        Args:
            path: Disk file path, relative to session cwd or absolute; not a buffer ID.
            prefix: Nonempty exact string at the beginning of the desired span.
            suffix: Nonempty exact string at the end of the desired span.
        """

    def execute(self, state, path, prefix, suffix):
        deadline = time.monotonic() + WAIT_SECONDS
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ClipError("invalid_path", "path must be a nonempty disk file path.")
        for name, value in (("prefix", prefix), ("suffix", suffix)):
            if not isinstance(value, str) or not value:
                raise ClipError("invalid_anchor", f"{name} must be a nonempty string.")
            if len(value.encode("utf-8")) > MAX_ANCHOR_BYTES:
                raise ClipError("invalid_anchor", f"{name} exceeds {MAX_ANCHOR_BYTES} UTF-8 bytes.")
        if len(state.span_citation_pending) >= MAX_PENDING:
            raise ClipError("busy", "Four clip jobs are outstanding; wait for their completion interrupts.")
        bind_namespace(state)
        job_id = "clip_" + uuid.uuid4().hex
        future = self.service.start(job_id, path, prefix, suffix, get_session_cwd(state))
        state.span_citation_pending[job_id] = {"path": path}
        try:
            result = future.result(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError:
            return {"status": "pending", "job_id": job_id,
                    "message": "Clipping continues in the background; completion will arrive as an interrupt."}
        return self.service.finish(state, job_id, result)

    def transform(self, state):
        for tc, kwargs in self.pending(state):
            try:
                result = self.execute(state, **kwargs)
            except (ClipError, UnicodeError, TypeError) as exc:
                result = {"status": "error", "code": getattr(exc, "code", "invalid_params"), "message": str(exc)}
            except Exception as exc:
                result = {"status": "error", "code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
            tc.result = encode(result)
            tc.error = result["status"] == "error"
            self._notify(state, tc)
        return state


class SpanCompletionCheck(InterruptCheck):
    check_name = "span_citation_completion"
    cooldown = 0
    optional_reads = {"buffer_manager", "span_citation_pending", "span_citation_records", "entries"}
    writes = {"span_citation_pending", "span_citation_records", "span_citation_notices"}
    init = {"span_citation_notices": dict}

    def __init__(self, service):
        self.service = service

    def __call__(self, state):
        if getattr(state, "buffer_manager", None) is None:
            return state
        bind_namespace(state)
        for job_id in list(getattr(state, "span_citation_pending", {})):
            future = self.service.futures.get(job_id)
            if future is not None and not future.done():
                continue
            result = future.result() if future is not None else {
                "status": "error", "code": "interrupted",
                "message": "Clip job did not survive backend reload; call clip_span again."}
            reply = self.service.finish(state, job_id, result)
            state.span_citation_notices[job_id] = (
                f"[Span clipping completed | {job_id}]\n" + encode(reply))
        for job_id, notice in list(state.span_citation_notices.items()):
            marker = f"[Span clipping completed | {job_id}]"
            delivered = any(
                getattr(entry, "system_generated", False)
                and any(isinstance(message.get("content"), str)
                        and message["content"].startswith(marker)
                        for message in entry.messages)
                for entry in getattr(state, "entries", []))
            if delivered:
                del state.span_citation_notices[job_id]
                self.service.enqueued.discard(job_id)
            elif job_id not in self.service.enqueued:
                self._fire(state, notice)
                self.service.enqueued.add(job_id)
        return state


class CloseCitation(Tool):
    """Layer only citation IDs ahead of the ordinary close_buffer handler."""
    name = "close_buffer"
    schema = CloseBuffer.schema
    description = CloseBuffer.description
    reads = {"span_citation_records"}
    writes = {"span_citation_records"}

    def transform(self, state):
        for tc, kwargs in self.pending(state):
            buffer_id = kwargs.get("buffer", "")
            if not isinstance(buffer_id, str) or not buffer_id.strip().startswith(PREFIX):
                continue
            buffer_id = buffer_id.strip()
            if buffer_id not in state.span_citation_records:
                tc.result, tc.error = f"ERROR: No citation buffer '{buffer_id}'.", True
            else:
                del state.span_citation_records[buffer_id]
                tc.result = f"Closed citation buffer {buffer_id}."
            self._notify(state, tc)
        return state


def register_features(builder, *, session, config):
    service = ClipService()
    builder.add(Feature(
        name="span_citations",
        components=[SpanSetup(), SpanStartup(), ClipSpan(service), SpanCompletionCheck(service), CloseCitation()],
        order=[HarnessEventHandler, BufferManagerProvider, SpanSetup, ToolDispatchStart, CloseCitation, CloseBuffer],
    ))
