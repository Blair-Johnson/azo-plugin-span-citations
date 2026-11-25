"""Exact, immutable file-span citations for Agent Zoo (stdlib-only, strict Sans-I/O)."""
from __future__ import annotations

import codecs
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any
import uuid

from agent_utils.core.events import (
    Event,
    InputReceived,
    InterruptRequested,
    IoCompleted,
    IoFailed,
    IoInterrupted,
    RunToolCall,
    SessionRestored,
    TimerFired,
    ToolCallFinished,
    TurnSettled,
)
from agent_utils.core.io import IoRequest, OpId, StartTimer
from agent_utils.core.pipeline import EmitFn, Feature
from agent_utils.core.state import DurableSpec, State, ToolCall
from agent_utils.core.tool import Tool
from agent_utils.features.buffers import BufferManagerProvider, CloseBuffer
from agent_utils.features.core import IoFence, ToolRunner
from agent_utils.features.interrupts import InterruptCheck
from agent_zoo.durability import ThreadPoolDriver, session_cwd
from agent_zoo.events import SessionReady

PREFIX = "span_"
DURABLE_KEY = "span_citations"
RUNTIME_KEY = "span_citations"
WAIT_SECONDS = 0.5
CHUNK_BYTES = 1024 * 1024
MAX_ANCHOR_BYTES = 64 * 1024
MAX_SPAN_BYTES = 16 * 1024 * 1024
MAX_PENDING = 4


class ClipError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# Typed IO request, durable state, and ephemeral runtime state
# ---------------------------------------------------------------------------


@dataclass
class ClipSpanRequest(IoRequest):
    """Read and anchor-match an exact UTF-8 file span on a driver worker thread."""

    job_id: str
    path: str
    prefix: str
    suffix: str
    cwd: str = ""


@dataclass
class SpanCitationState:
    """Durable per-session span citation records, pending jobs, and notices."""

    records: dict[str, str] = field(default_factory=dict)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    notices: dict[str, str] = field(default_factory=dict)


def _fork_span_citation_state(value: Any) -> SpanCitationState:
    if isinstance(value, SpanCitationState):
        return SpanCitationState(records=dict(value.records))
    if isinstance(value, dict):
        return SpanCitationState(records=dict(value.get("records") or {}))
    return SpanCitationState()


SPAN_CITATIONS_DURABLE_SPEC = DurableSpec(
    name=DURABLE_KEY,
    cls=SpanCitationState,
    version=1,
    on_fork=_fork_span_citation_state,
)


@dataclass
class SpanCitationRuntime:
    """Strictly ephemeral per-session tracking for active clip ops and wait timers."""

    active_ops: dict[OpId, str] = field(default_factory=dict)
    job_ops: dict[str, OpId] = field(default_factory=dict)
    inline_waits: dict[OpId, tuple[str, OpId, ToolCall]] = field(default_factory=dict)
    inline_clips: dict[OpId, tuple[OpId, ToolCall]] = field(default_factory=dict)
    enqueued_notices: set[str] = field(default_factory=set)


def span_state(state: State) -> SpanCitationState:
    """Return the durable ``SpanCitationState`` registered under ``span_citations``."""
    if hasattr(state, "register_durable_spec") and DURABLE_KEY not in getattr(state, "_durable_specs", {}):
        state.register_durable_spec(SPAN_CITATIONS_DURABLE_SPEC)
    return state.durable(DURABLE_KEY, SpanCitationState)


def span_runtime(state: State) -> SpanCitationRuntime:
    """Return the ephemeral per-session ``SpanCitationRuntime`` namespace."""
    return state.runtime(RUNTIME_KEY, SpanCitationRuntime)


# ---------------------------------------------------------------------------
# Disk extraction helpers (executed ONLY inside SpanClipDriver worker threads)
# ---------------------------------------------------------------------------


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _extremes(data: bytes, anchor: bytes, base: int, floor: int, positions: Any, first: bool) -> None:
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


def _scan_snapshot(source: Any, snapshot: Any, prefix: bytes, suffix: bytes) -> tuple[int, int, int, str]:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    starts: list[int] = []
    ends: deque[int] = deque(maxlen=2)
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


def extract_citation(path: str, prefix: str, suffix: str, cwd: str) -> dict[str, Any]:
    """Read one byte snapshot; never reread the live file to extract its quote."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path(cwd) / resolved) if cwd else resolved
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


def _worker(path: str, prefix: str, suffix: str, cwd: str) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Runtime IO driver
# ---------------------------------------------------------------------------


class SpanClipDriver(ThreadPoolDriver):
    """Bounded thread-pool driver executing ``ClipSpanRequest`` disk reads."""

    durability = "ephemeral"
    request_types = (ClipSpanRequest,)
    max_workers = MAX_PENDING
    worker_prefix = "span-clip"

    def __init__(self, *, max_workers: int = MAX_PENDING) -> None:
        super().__init__(max_workers=max_workers)

    def execute(self, req: IoRequest) -> dict[str, Any]:
        if not isinstance(req, ClipSpanRequest):
            raise TypeError(f"SpanClipDriver expected ClipSpanRequest, got {type(req).__name__}")
        return _worker(req.path, req.prefix, req.suffix, req.cwd)


# ---------------------------------------------------------------------------
# Pure Sans-I/O state helpers & special-buffer provider
# ---------------------------------------------------------------------------


def citation_provider(state: State, buffer_id: str) -> str | None:
    return span_state(state).records.get(buffer_id)


def bind_namespace(state: State) -> None:
    state.buffer_manager.register_special_buffer_namespace(PREFIX, citation_provider, replace=True)


def _pending_reply(job_id: str) -> dict[str, Any]:
    return {
        "status": "pending",
        "job_id": job_id,
        "message": "Clipping continues in the background; completion will arrive as an interrupt.",
    }


def _interrupted_result() -> dict[str, Any]:
    return {
        "status": "error",
        "code": "interrupted",
        "message": "Clip job did not survive backend reload; call clip_span again.",
    }


def _finish_job(state: State, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
    durable = span_state(state)
    durable.pending.pop(job_id, None)
    if result.get("status") == "error":
        return {"job_id": job_id, **result}
    buffer_id = PREFIX + uuid.uuid4().hex
    durable.records[buffer_id] = result["citation_json"]
    return {
        "status": "complete",
        "job_id": job_id,
        "buffer_id": buffer_id,
        "readonly": True,
        "message": "Use view(buffer=...) to read; close_buffer(buffer=...) to release.",
    }


def _format_notice(job_id: str, reply: dict[str, Any]) -> str:
    return f"[Span clipping completed | {job_id}]\n" + encode(reply)


def _normalize_io_result(event: IoCompleted | IoFailed) -> dict[str, Any]:
    if isinstance(event, IoCompleted) and isinstance(event.value, dict):
        if event.value.get("status") == "error":
            return event.value
        if event.value.get("status") == "complete" and isinstance(event.value.get("citation_json"), str):
            return event.value
    if isinstance(event, IoFailed):
        err = event.error
        code = getattr(err, "code", "internal_error")
        message = str(err) if not isinstance(err, BaseException) else f"{type(err).__name__}: {err}"
        return {"status": "error", "code": code, "message": message}
    return {"status": "error", "code": "internal_error", "message": "Invalid clip driver result."}


def _is_notice_delivered(state: State, job_id: str) -> bool:
    marker = f"[Span clipping completed | {job_id}]"
    return any(
        getattr(entry, "system_generated", False)
        and any(
            isinstance(message.get("content"), str) and message["content"].startswith(marker)
            for message in entry.messages
        )
        for entry in getattr(state, "entries", [])
    )


def _reap_orphaned_pending(state: State) -> None:
    durable = span_state(state)
    rt = span_runtime(state)
    for job_id in list(durable.pending):
        if job_id in rt.job_ops:
            continue
        reply = _finish_job(state, job_id, _interrupted_result())
        durable.notices[job_id] = _format_notice(job_id, reply)


def _queue_interrupt_message(state: State, message: str, *, check_name: str = "span_citation_completion") -> None:
    if not hasattr(state, "pending_interrupts") or state.pending_interrupts is None:
        state.pending_interrupts = []
    if not hasattr(state, "_interrupt_state") or state._interrupt_state is None:
        state._interrupt_state = {}
    state.pending_interrupts.append(message)
    state._interrupt_state.setdefault(check_name, {})["last_fired"] = state.step


def _flush_notices(state: State, *, fire_fn: Any = None) -> None:
    durable = span_state(state)
    rt = span_runtime(state)
    for job_id, notice in list(durable.notices.items()):
        if _is_notice_delivered(state, job_id):
            del durable.notices[job_id]
            rt.enqueued_notices.discard(job_id)
        elif job_id not in rt.enqueued_notices:
            if fire_fn is not None:
                fire_fn(state, notice)
            else:
                _queue_interrupt_message(state, notice)
            rt.enqueued_notices.add(job_id)


# ---------------------------------------------------------------------------
# Pure Sans-I/O pipeline components
# ---------------------------------------------------------------------------


class SpanSetup:
    handles = (InputReceived, RunToolCall, TurnSettled)
    reads = {"buffer_manager"}
    writes: set[str] = set()
    init: dict[str, Any] = {}

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        span_state(state)
        if getattr(state, "buffer_manager", None) is not None:
            bind_namespace(state)


class SpanStartup:
    handles = (SessionReady, SessionRestored)
    optional_reads = {"buffer_manager", "entries", "pending_interrupts"}
    writes: set[str] = set()
    init: dict[str, Any] = {}

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        span_state(state)
        if getattr(state, "buffer_manager", None) is not None:
            bind_namespace(state)
        if isinstance(event, SessionRestored):
            _reap_orphaned_pending(state)
            _flush_notices(state)


class ClipSpan(Tool):
    handles = (RunToolCall, IoCompleted, IoFailed, TimerFired, IoInterrupted, InterruptRequested)
    reads = {"buffer_manager"}
    writes: set[str] = set()
    optional_reads = {"cwd", "session_cwd", "terminal_cwd"}
    group = "Span citations"
    io_tag = "span_citations"

    def call(self, state: State, path: str, prefix: str, suffix: str) -> str:
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
        raise NotImplementedError("ClipSpan executes asynchronously via handle()")

    @staticmethod
    def _validate(state: State, path: Any, prefix: Any, suffix: Any) -> tuple[str, str, str]:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ClipError("invalid_path", "path must be a nonempty disk file path.")
        for name, value in (("prefix", prefix), ("suffix", suffix)):
            if not isinstance(value, str) or not value:
                raise ClipError("invalid_anchor", f"{name} must be a nonempty string.")
            if len(value.encode("utf-8")) > MAX_ANCHOR_BYTES:
                raise ClipError("invalid_anchor", f"{name} exceeds {MAX_ANCHOR_BYTES} UTF-8 bytes.")
        if len(span_state(state).pending) >= MAX_PENDING:
            raise ClipError("busy", "Four clip jobs are outstanding; wait for their completion interrupts.")
        return path, prefix, suffix

    @staticmethod
    def _complete_tc(tc: ToolCall, reply: dict[str, Any], ts: float, emit: EmitFn) -> None:
        tc.waiting_op = None
        tc._owner = None
        tc.result = encode(reply)
        tc.error = reply.get("status") == "error"
        tc.finished_at = ts
        emit(ToolCallFinished(tc))

    def on_deadline(self, tc: ToolCall, state: State, emit: EmitFn) -> bool:
        rt = span_runtime(state)
        clip_op = tc.waiting_op
        if clip_op is not None and clip_op in rt.inline_clips:
            timer_op, _ = rt.inline_clips.pop(clip_op)
            entry = rt.inline_waits.pop(timer_op, None)
            state.io.cancel(timer_op)
            job_id = entry[0] if entry is not None else rt.active_ops.get(clip_op, "")
            self._complete_tc(tc, _pending_reply(job_id), 0.0, emit)
            return True
        return super().on_deadline(tc, state, emit)

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        if isinstance(event, RunToolCall):
            self._handle_run_tool_call(event, state, emit)
        elif isinstance(event, (IoCompleted, IoFailed)):
            self._handle_io_done(event, state, emit)
        elif isinstance(event, TimerFired):
            self._handle_timer_fired(event, state, emit)
        elif isinstance(event, IoInterrupted):
            self._handle_io_interrupted(event, state)
        elif isinstance(event, InterruptRequested):
            self._handle_interrupt_requested(state)

    def _handle_run_tool_call(self, event: RunToolCall, state: State, emit: EmitFn) -> None:
        tc = event.tool_call
        if tc.name != self.name or tc.result is not None:
            return
        event.stop()
        if tc.started_at is None:
            tc.started_at = event.ts
        if not self.available(state):
            tc.result = f"ERROR: {self.name} is not currently available."
            tc.error = True
            tc.finished_at = event.ts
            tc._owner = None
            emit(ToolCallFinished(tc))
            return
        try:
            path, prefix, suffix = self._validate(state, **tc.parsed_args)
        except (ClipError, UnicodeError, TypeError) as exc:
            reply = {"status": "error", "code": getattr(exc, "code", "invalid_params"), "message": str(exc)}
            self._complete_tc(tc, reply, event.ts, emit)
            return
        except Exception as exc:
            reply = {"status": "error", "code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
            self._complete_tc(tc, reply, event.ts, emit)
            return

        bind_namespace(state)
        durable = span_state(state)
        rt = span_runtime(state)
        job_id = "clip_" + uuid.uuid4().hex
        durable.pending[job_id] = {"path": path}
        clip_op = state.io.submit(
            ClipSpanRequest(
                job_id=job_id,
                path=path,
                prefix=prefix,
                suffix=suffix,
                cwd=session_cwd(state),
            ),
            tag="span_citations",
            meta={"job_id": job_id},
        )
        rt.active_ops[clip_op] = job_id
        rt.job_ops[job_id] = clip_op

        if WAIT_SECONDS <= 0:
            self._complete_tc(tc, _pending_reply(job_id), event.ts, emit)
            return

        timer_op = state.io.submit(
            StartTimer(delay_s=float(WAIT_SECONDS)),
            tag="span_citations_wait",
            meta={"job_id": job_id, "clip_op": clip_op},
        )
        rt.inline_waits[timer_op] = (job_id, clip_op, tc)
        rt.inline_clips[clip_op] = (timer_op, tc)
        tc._owner = self
        tc.waiting_op = clip_op

    def _handle_io_done(self, event: IoCompleted | IoFailed, state: State, emit: EmitFn) -> None:
        rt = span_runtime(state)
        inline = rt.inline_clips.pop(event.op_id, None)
        if inline is None:
            return
        timer_op, tc = inline
        rt.inline_waits.pop(timer_op, None)
        state.io.cancel(timer_op)
        job_id = rt.active_ops.pop(event.op_id, "")
        if job_id:
            rt.job_ops.pop(job_id, None)
        event.stop()
        if getattr(state, "buffer_manager", None) is not None:
            bind_namespace(state)
        reply = _finish_job(state, job_id, _normalize_io_result(event))
        self._complete_tc(tc, reply, event.ts, emit)

    def _handle_timer_fired(self, event: TimerFired, state: State, emit: EmitFn) -> None:
        rt = span_runtime(state)
        wait_entry = rt.inline_waits.pop(event.op_id, None)
        if wait_entry is None:
            return
        job_id, clip_op, tc = wait_entry
        rt.inline_clips.pop(clip_op, None)
        event.stop()
        # Do NOT cancel clip_op: clipping continues in the background and
        # SpanCompletionCheck handles its later IoCompleted/IoFailed.
        self._complete_tc(tc, _pending_reply(job_id), event.ts, emit)

    def _handle_io_interrupted(self, event: IoInterrupted, state: State) -> None:
        rt = span_runtime(state)
        inline = rt.inline_clips.pop(event.op_id, None)
        if inline is None:
            return
        timer_op, tc = inline
        rt.inline_waits.pop(timer_op, None)
        state.io.cancel(timer_op)
        job_id = rt.active_ops.pop(event.op_id, "")
        if job_id:
            rt.job_ops.pop(job_id, None)
            span_state(state).pending.pop(job_id, None)
        tc.waiting_op = None
        tc._owner = None

    def _handle_interrupt_requested(self, state: State) -> None:
        rt = span_runtime(state)
        durable = span_state(state)
        for clip_op, (timer_op, _tc) in list(rt.inline_clips.items()):
            rt.inline_clips.pop(clip_op, None)
            rt.inline_waits.pop(timer_op, None)
            state.io.cancel(timer_op)
            state.io.cancel(clip_op)
            job_id = rt.active_ops.pop(clip_op, "")
            if job_id:
                rt.job_ops.pop(job_id, None)
                durable.pending.pop(job_id, None)


class SpanCompletionCheck(InterruptCheck):
    check_name = "span_citation_completion"
    cooldown = 0
    handles = (IoCompleted, IoFailed, TurnSettled)
    optional_reads = {"buffer_manager", "entries"}
    writes = {"pending_interrupts", "_interrupt_state"}
    init = {"pending_interrupts": list, "_interrupt_state": dict}

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        if isinstance(event, (IoCompleted, IoFailed)):
            rt = span_runtime(state)
            job_id = rt.active_ops.pop(event.op_id, None)
            if job_id is None:
                return
            rt.job_ops.pop(job_id, None)
            if getattr(state, "buffer_manager", None) is not None:
                bind_namespace(state)
            reply = _finish_job(state, job_id, _normalize_io_result(event))
            notice = _format_notice(job_id, reply)
            durable = span_state(state)
            durable.notices[job_id] = notice
            if job_id not in rt.enqueued_notices:
                self._fire(state, notice, emit)
                rt.enqueued_notices.add(job_id)
            return

        if not isinstance(event, TurnSettled):
            return
        if getattr(state, "buffer_manager", None) is None:
            return
        bind_namespace(state)
        _reap_orphaned_pending(state)
        _flush_notices(state, fire_fn=lambda s, m: self._fire(s, m, emit))


class CloseCitation:
    """Intercept close_buffer calls for span_* citation IDs ahead of CloseBuffer."""

    handles = (RunToolCall,)
    name = "close_buffer"
    reads: set[str] = set()
    writes: set[str] = set()

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        if not isinstance(event, RunToolCall):
            return
        tc = event.tool_call
        if tc.name != self.name:
            return
        buffer_id = tc.parsed_args.get("buffer", "")
        if not isinstance(buffer_id, str) or not buffer_id.strip().startswith(PREFIX):
            return
        event.stop()
        if tc.started_at is None:
            tc.started_at = event.ts
        buffer_id = buffer_id.strip()
        records = span_state(state).records
        if buffer_id not in records:
            tc.result = f"ERROR: No citation buffer '{buffer_id}'."
            tc.error = True
        else:
            del records[buffer_id]
            tc.result = f"Closed citation buffer {buffer_id}."
            tc.error = False
        tc.finished_at = event.ts
        tc._owner = None
        emit(ToolCallFinished(tc))


def register_features(builder: Any, *, config: Any) -> None:
    del config
    builder.add(Feature(
        name="span_citations",
        components=[SpanSetup(), SpanStartup(), ClipSpan(), SpanCompletionCheck(), CloseCitation()],
        order=[IoFence, BufferManagerProvider, {SpanSetup, SpanStartup}, ToolRunner, CloseCitation, CloseBuffer],
        drivers=[SpanClipDriver],
        durable_state=SPAN_CITATIONS_DURABLE_SPEC,
    ))


# ---------------------------------------------------------------------------
# Deterministic Simulation Testing (DST) Hooks
# ---------------------------------------------------------------------------

_SIM_LARGE_TEXT = "\n".join(
    f"line {i:04d}: sample oversized preview content for view tool fuzzing"
    for i in range(1, 1100)
) + "\n"

_SIM_SPAN_FILES: dict[str, bytes] = {
    "/workspace/app.py": b"def greet():\n    return 'hello'\n",
    "/workspace/notes.txt": b"line 1\nline 2\n",
    "/workspace/large.txt": _SIM_LARGE_TEXT.encode("utf-8"),
}


def _sim_extract_citation(req: ClipSpanRequest) -> dict[str, Any]:
    """Pure in-memory deterministic span extraction mirroring ``SpanClipDriver``."""
    import io
    import posixpath

    raw_path = str(req.path or "")
    if raw_path.startswith("/"):
        resolved = posixpath.normpath(raw_path)
    else:
        resolved = posixpath.normpath(posixpath.join(req.cwd or "/workspace", raw_path))
    data = _SIM_SPAN_FILES.get(resolved)
    if data is None:
        return {"status": "error", "code": "io_error", "message": f"No such file: {resolved}"}

    try:
        source = io.BytesIO(data)
        snapshot = io.BytesIO()
        start, end, size, digest = _scan_snapshot(
            source,
            snapshot,
            req.prefix.encode("utf-8"),
            req.suffix.encode("utf-8"),
        )
        if end - start > MAX_SPAN_BYTES:
            raise ClipError("span_too_large", f"Span exceeds {MAX_SPAN_BYTES} bytes; narrow the anchors.")
        head = data[:start].decode("utf-8")
        text = data[start:end].decode("utf-8")
        char_start = len(head)
        line_start = 1 + head.count("\n")
        citation = {
            "schema": "azo.span-citation",
            "version": 1,
            "file": {
                "path": resolved,
                "encoding": "utf-8",
                "size_bytes": size,
                "hash": {"algorithm": "sha256", "hex": digest},
            },
            "match": {
                "mode": "exact",
                "prefix": req.prefix,
                "suffix": req.suffix,
                "anchors_included": True,
            },
            "span": {
                "text": text,
                "byte_start": start,
                "byte_end": end,
                "char_start": char_start,
                "char_end": char_start + len(text),
                "line_start": line_start,
                "line_end": line_start + text[:-1].count("\n"),
            },
        }
        return {"status": "complete", "citation_json": encode(citation)}
    except ClipError as exc:
        return {"status": "error", "code": exc.code, "message": str(exc)}
    except UnicodeError:
        return {"status": "error", "code": "invalid_utf8", "message": "File and anchors must be valid UTF-8 text."}


def register_sim_handlers(sim: Any, rng: Any) -> None:
    """Register deterministic in-memory simulation handlers for ``ClipSpanRequest``."""
    from agent_utils.runtime.simulation import SimOutcome

    def _handle_clip_span(_op_id: OpId, req: ClipSpanRequest, _sim: Any) -> SimOutcome:
        # Exercise both fast inline completion (< WAIT_SECONDS) and background
        # pending completion (> WAIT_SECONDS -> TimerFired + SpanCompletionCheck).
        if rng.random() < 0.65:
            delay = rng.uniform(0.02, 0.25)
        else:
            delay = rng.uniform(WAIT_SECONDS + 0.05, WAIT_SECONDS + 0.35)

        roll = rng.random()
        if roll < 0.05:
            return SimOutcome(error=OSError("simulated clip read failure"), delay=delay)
        if roll < 0.10:
            return SimOutcome(
                value={
                    "status": "error",
                    "code": "file_changed",
                    "message": "File changed while reading; retry against a stable file.",
                },
                delay=delay,
            )
        return SimOutcome(value=_sim_extract_citation(req), delay=delay)

    import sys as _sys

    sim.register_handler(ClipSpanRequest, _handle_clip_span)
    live_cls = getattr(_sys.modules.get(__name__), "ClipSpanRequest", None)
    if isinstance(live_cls, type) and live_cls is not ClipSpanRequest:
        sim.register_handler(live_cls, _handle_clip_span)


def fuzz_tool_args(tool_name: str, rng: Any, state: State) -> dict[str, Any] | None:
    """Generate deterministic valid and edge-case arguments for ``clip_span``."""
    del state
    if tool_name != "clip_span":
        return None
    roll = rng.random()
    if roll < 0.80:
        path, prefix, suffix = rng.choice(
            [
                ("/workspace/app.py", "def greet():", "return 'hello'"),
                ("app.py", "def ", "'hello'"),
                ("/workspace/notes.txt", "line 1", "line 2"),
                ("/workspace/large.txt", "line 0001:", "line 0003:"),
                ("/workspace/large.txt", "line 0010:", "line 0012:"),
            ]
        )
        return {"path": path, "prefix": prefix, "suffix": suffix}
    if roll < 0.90:
        path, prefix, suffix = rng.choice(
            [
                ("/workspace/notes.txt", "line", "line"),
                ("/workspace/app.py", "missing_prefix", "missing_suffix"),
            ]
        )
        return {"path": path, "prefix": prefix, "suffix": suffix}
    return {"path": "/workspace/missing.txt", "prefix": "start", "suffix": "end"}


def check_invariants(session: Any) -> None:
    """Verify durable and ephemeral invariants for span citation state."""
    state = session.state
    durable = state.features.get(DURABLE_KEY)
    is_durable = isinstance(durable, SpanCitationState) or type(durable).__name__ == "SpanCitationState"
    if is_durable:
        assert isinstance(durable.records, dict)
        assert isinstance(durable.pending, dict)
        assert isinstance(durable.notices, dict)
        assert len(durable.pending) <= MAX_PENDING, (
            f"pending clip jobs exceeded MAX_PENDING ({len(durable.pending)} > {MAX_PENDING})"
        )
        for bid, raw in durable.records.items():
            assert isinstance(bid, str) and bid.startswith(PREFIX), (
                f"invalid citation buffer id: {bid!r}"
            )
            assert isinstance(raw, str) and raw
            parsed = json.loads(raw)
            assert parsed.get("schema") == "azo.span-citation" and parsed.get("version") == 1
        for jid, notice in durable.notices.items():
            assert isinstance(jid, str) and jid
            assert isinstance(notice, str) and notice.startswith(f"[Span clipping completed | {jid}]")
            assert jid not in durable.pending, (
                f"job {jid!r} present in both durable.pending and durable.notices"
            )

    rt = state.runtime(RUNTIME_KEY) if callable(getattr(state, "runtime", None)) else None
    if isinstance(rt, SpanCitationRuntime) or type(rt).__name__ == "SpanCitationRuntime":
        assert len(rt.active_ops) == len(rt.job_ops), (
            f"active_ops/job_ops mismatch: {len(rt.active_ops)} != {len(rt.job_ops)}"
        )
        for op_id, jid in rt.active_ops.items():
            assert rt.job_ops.get(jid) == op_id, (
                f"job_ops bijection broken for {jid!r}: {rt.job_ops.get(jid)!r} != {op_id!r}"
            )
            if is_durable:
                assert jid in durable.pending, f"active job {jid!r} missing from durable.pending"
        for clip_op, (timer_op, _tc) in rt.inline_clips.items():
            assert clip_op in rt.active_ops, f"inline_clips op {clip_op!r} missing from active_ops"
            assert timer_op in rt.inline_waits, f"inline_clips timer {timer_op!r} missing from inline_waits"
            assert rt.inline_waits[timer_op][1] == clip_op
        for timer_op, (_jid, clip_op, _tc) in rt.inline_waits.items():
            assert clip_op in rt.inline_clips, f"inline_waits clip {clip_op!r} missing from inline_clips"
            assert rt.inline_clips[clip_op][0] == timer_op
