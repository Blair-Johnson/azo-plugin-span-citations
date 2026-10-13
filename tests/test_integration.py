"""Integration against installed Agent Zoo APIs; no model, backend, or shell needed."""
from __future__ import annotations

from concurrent.futures import Future
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

import pytest

from agent_utils.components import BufferManagerProvider, ToolDispatchEnd, ToolDispatchStart
from agent_utils.session import Session, flatten
from agent_utils.tool import Tool
from agent_utils.tools.buffers import (
    AppendLines, BufferInfo, Checkpoint, CloseBuffer, CopyBuffer,
    Grep, InsertLines, PrependLines, RawDiff, RestoreCheckpoint,
    Save, StrReplace, View,
)
from agent_utils.types import Entry, ToolCall

import span_citations as spans


ROOT = Path(__file__).resolve().parents[1]


def make_session():
    """Initialize real component defaults without running an LLM turn."""
    service = spans.ClipService()
    session = Session()
    session.pipeline = [
        BufferManagerProvider(), spans.SpanSetup(), spans.ClipSpan(service),
        spans.SpanCompletionCheck(service), spans.CloseCitation(), CloseBuffer(),
    ]
    session.ensure_initialized()
    spans.SpanSetup()(session.state)
    return session, service


def call(state, tool, **kwargs):
    """Exercise Tool.pending/result/error/_notify, not private _exec methods."""
    tools = tool if isinstance(tool, (list, tuple)) else [tool]
    tc = ToolCall(id=f"call-{state.step}", name=tools[0].name,
                  raw_args=json.dumps(kwargs), parsed_args=kwargs)
    state.pending_tool_calls = [tc]
    state.entries.append(Entry(
        messages=[{"role": "assistant", "content": ""}],
        index=len(state.entries), step=state.step, tokens=0, tool_calls=[tc],
    ))
    state.step += 1
    for component in tools:
        component(state)
    assert tc.result is not None, f"Unhandled tool call: {tc.name}"
    return tc


def close(state, **kwargs):
    return call(state, [spans.CloseCitation(), CloseBuffer()], **kwargs)


def inline_worker(monkeypatch, service):
    """Already-completed Future makes inline behavior independent of scheduling."""
    def start(job_id, path, prefix, suffix, cwd):
        future = Future()
        service.futures[job_id] = future
        future.set_result(spans._worker(path, prefix, suffix, cwd))
        return future
    monkeypatch.setattr(service, "start", start)


@pytest.fixture
def runtime(tmp_path):
    session, service = make_session()
    session.state.session_cwd = str(tmp_path)
    yield session, service
    session.state.buffer_manager.cleanup()


@pytest.fixture
def citation(runtime, tmp_path, monkeypatch):
    session, service = runtime
    inline_worker(monkeypatch, service)
    source = tmp_path / "source.txt"
    source.write_bytes("outside é\nBEGIN α\r\n  exact\tEND\nafter".encode("utf-8"))
    tc = call(session.state, spans.ClipSpan(service),
              path="source.txt", prefix="BEGIN", suffix="END")
    assert not tc.error, tc.result
    reply = json.loads(tc.result)
    assert reply["status"] == "complete"
    assert reply["readonly"] is True
    bid = reply["buffer_id"]
    assert bid.startswith("span_")
    return session, service, bid, source


def test_real_read_tools_and_copy_by_value(citation):
    session, service, bid, source = citation
    state = session.state
    bm = state.buffer_manager
    original = state.span_citation_records[bid]
    record = json.loads(original)
    assert record["span"]["text"] == "BEGIN α\r\n  exact\tEND"
    assert record["file"]["path"] == str(source.resolve())
    assert bm.resolve_for_read(state, bid).readonly is True
    assert bid not in bm.buffers()
    for tool, args, expected in [
        (View(), {"buffer": bid}, "azo.span-citation"),
        (Grep(), {"buffer": bid, "pattern": "exact"}, "exact"),
        (BufferInfo(), {"buffers": [bid]}, "[readonly]"),
    ]:
        tc = call(state, tool, **args)
        assert not tc.error, tc.result
        assert expected in tc.result
    target, _ = bm.reserve_buffer()
    tc = call(state, CopyBuffer(), source=bid, target=target)
    assert not tc.error, tc.result
    assert target.startswith("buf_") and target not in state.span_citation_records
    assert bm.require_writable(target).text == original
    assert not call(state, AppendLines(), buffer=target, text="copy-only edit").error
    assert state.span_citation_records[bid] == original
    # Reads use retained bytes, even after the live file has vanished.
    source.unlink()
    assert bm.resolve_for_read(state, bid).text == original
    assert not call(state, View(), buffer=bid).error
    assert not close(state, buffer=bid).error
    assert "copy-only edit" in bm.require_writable(target).text
    assert not close(state, buffer=target, discard=True).error
    spans.SpanCompletionCheck(service)(state)
    assert state.pending_interrupts == []


@pytest.mark.parametrize("tool,kwargs", [
    (StrReplace, {"old": "exact", "new": "forged"}),
    (StrReplace, {"start_exclusive": "BEGIN", "end_exclusive": "END", "replacement": "forged"}),
    (InsertLines, {"anchor": '"schema"', "before_anchor": True, "text": "forged"}),
    (AppendLines, {"text": "forged"}),
    (PrependLines, {"text": "forged"}),
    (Save, {}),
    (Save, {"force": True}),
    (Checkpoint, {"label": "forged"}),
    (RestoreCheckpoint, {"label": "forged"}),
    (RawDiff, {}),
])
def test_readonly_rejects_edit_save_and_checkpoint_tools(citation, tmp_path, tool, kwargs):
    session, _, bid, source = citation
    state = session.state
    original = state.span_citation_records[bid]
    original_disk = source.read_bytes()
    args = {"buffer": bid, **kwargs}
    output = tmp_path / "must-not-write.json"
    if tool is Save:
        args["path"] = str(output)
    tc = call(state, tool(), **args)
    assert tc.error, tc.result
    assert "readonly" in tc.result.lower(), tc.result
    assert state.span_citation_records[bid] == original
    assert state.buffer_manager.resolve_for_read(state, bid).text == original
    assert source.read_bytes() == original_disk
    assert not output.exists()
    assert state.buffer_manager.buffers() == []


@pytest.mark.parametrize("overwrite", [False, True])
def test_copy_cannot_overwrite_citation(citation, overwrite):
    session, _, bid, _ = citation
    state = session.state
    original = state.span_citation_records[bid]
    source, _ = state.buffer_manager.create_scratch_buffer("forged")
    tc = call(state, CopyBuffer(), source=source, target=bid, overwrite=overwrite)
    assert tc.error and "readonly" in tc.result.lower(), tc.result
    assert state.span_citation_records[bid] == original


def test_close_deletes_immediately_and_notifies_once(citation):
    session, _, bid, _ = citation
    state = session.state
    notifications = []
    state.on_tool_call = notifications.append
    tc = close(state, buffer=f"  {bid}  ")
    assert not tc.error, tc.result
    assert notifications == [tc]
    assert bid not in state.span_citation_records
    assert call(state, View(), buffer=bid).error
    assert call(state, Grep(), buffer=bid, pattern="exact").error
    assert call(state, BufferInfo(), buffers=[bid]).error
    target, _ = state.buffer_manager.reserve_buffer()
    assert call(state, CopyBuffer(), source=bid, target=target).error
    assert state.buffer_manager.is_reserved(target)
    tc = close(state, buffer=bid, discard=True)
    assert tc.error and "No citation buffer" in tc.result


@pytest.mark.parametrize("kind", ["clean", "dirty", "scratch", "reserved", "omitted", "unknown"])
def test_plain_close_semantics_are_unchanged(runtime, tmp_path, kind):
    session, _ = runtime
    state = session.state
    bm = state.buffer_manager
    if kind == "unknown":
        bid = "buf_not_real"
    elif kind == "reserved":
        bid, _ = bm.reserve_buffer()
    elif kind == "scratch":
        bid, _ = bm.create_scratch_buffer("")
    else:
        path = tmp_path / "ordinary.txt"
        path.write_text("clean disk content", encoding="utf-8")
        bid, _, _ = bm.open_buffer(str(path))
    if kind == "dirty":
        assert not call(state, AppendLines(), buffer=bid, text="unsaved").error
    args = {} if kind == "omitted" else {"buffer": bid}
    # The citation layer must leave a non-citation call completely unhandled.
    tc = ToolCall(id="probe", name="close_buffer", raw_args="{}", parsed_args=args)
    state.pending_tool_calls = [tc]
    spans.CloseCitation().transform(state)
    assert tc.result is None and not tc.error
    result = close(state, **args)
    assert result.error == (kind in {"dirty", "scratch", "unknown"}), result.result
    if kind in {"dirty", "scratch"}:
        assert bm.get(bid) is not None
        assert not close(state, buffer=bid, discard=True).error
    assert bm.get(bid) is None


def restore_session(path):
    session, service = make_session()
    session.load(str(path), verify_blobs=True)
    session.ensure_initialized()
    # Restored data has no persisted provider callbacks. Normal setup rebinds it.
    spans.SpanSetup()(session.state)
    return session, service


def test_json_session_roundtrip_preserves_readonly_and_close_deletion(citation, tmp_path):
    session, _, bid, source = citation
    original = session.state.span_citation_records[bid]
    path = tmp_path / "session.json"
    session.save(str(path))
    document = json.loads(path.read_text())
    assert document["attrs"]["span_citation_records"][bid] == original
    source.unlink()
    restored, service = restore_session(path)
    try:
        state = restored.state
        assert state.span_citation_records == {bid: original}
        assert state.buffer_manager.resolve_for_read(state, bid).readonly
        assert not call(state, View(), buffer=bid).error
        tc = call(state, AppendLines(), buffer=bid, text="forged")
        assert tc.error and "readonly" in tc.result.lower()
        assert service.futures == {}
        assert not close(state, buffer=bid).error
        restored.save(str(path))
        again, _ = restore_session(path)
        try:
            assert again.state.span_citation_records == {}
            assert call(again.state, View(), buffer=bid).error
        finally:
            again.state.buffer_manager.cleanup()
    finally:
        restored.state.buffer_manager.cleanup()


@pytest.mark.parametrize("success", [True, False])
def test_inline_completion_has_no_interrupt(runtime, tmp_path, monkeypatch, success):
    session, service = runtime
    state = session.state
    inline_worker(monkeypatch, service)
    source = tmp_path / "inline.txt"
    source.write_text("BEGIN content END", encoding="utf-8")
    tc = call(state, spans.ClipSpan(service), path=str(source),
              prefix="BEGIN" if success else "missing", suffix="END")
    reply = json.loads(tc.result)
    assert reply["status"] == ("complete" if success else "error")
    assert tc.error is (not success)
    if not success:
        assert reply["code"] == "no_match"
    for _ in range(3):
        spans.SpanCompletionCheck(service)(state)
    assert state.pending_interrupts == []
    assert state.span_citation_pending == {} and service.futures == {}
    assert len(state.span_citation_records) == int(success)


@pytest.mark.parametrize("success", [True, False])
def test_deferred_worker_delivers_exactly_one_completion(runtime, tmp_path, monkeypatch, success):
    session, service = runtime
    state = session.state
    entered, release = threading.Event(), threading.Event()
    worker = spans._worker
    def gated_worker(*args):
        entered.set()
        if not release.wait(10):
            return {"status": "error", "code": "test_timeout", "message": "gate not released"}
        return worker(*args)
    monkeypatch.setattr(spans, "_worker", gated_worker)
    monkeypatch.setattr(spans, "WAIT_SECONDS", 0)
    source = tmp_path / "deferred.txt"
    source.write_text("BEGIN content END", encoding="utf-8")
    future = None
    try:
        tc = call(state, spans.ClipSpan(service), path=str(source),
                  prefix="BEGIN" if success else "missing", suffix="END")
        assert not tc.error
        pending = json.loads(tc.result)
        assert pending["status"] == "pending"
        job = pending["job_id"]
        future = service.futures[job]
        assert entered.wait(5)
        assert not future.done()
        assert state.span_citation_pending == {job: {"path": str(source)}}
        check = spans.SpanCompletionCheck(service)
        check(state)
        assert state.pending_interrupts == [] and state.span_citation_records == {}
        release.set()
        result = future.result(timeout=5)
        assert result["status"] == ("complete" if success else "error")
        # Worker completion alone must never mutate harness state.
        assert job in state.span_citation_pending
        assert state.span_citation_records == {} and state.pending_interrupts == []
        check(state)
        assert len(state.pending_interrupts) == 1
        heading, body = state.pending_interrupts[0].split("\n", 1)
        assert heading == f"[Span clipping completed | {job}]"
        reply = json.loads(body)
        assert reply["job_id"] == job
        assert reply["status"] == ("complete" if success else "error")
        if success:
            assert reply["readonly"] is True
            assert not call(state, View(), buffer=reply["buffer_id"]).error
        else:
            assert reply["code"] == "no_match" and "buffer_id" not in reply
        for _ in range(3):
            check(state)
        assert len(state.pending_interrupts) == 1
        assert state.span_citation_pending == {} and service.futures == {}
        assert len(state.span_citation_records) == int(success)
    finally:
        release.set()
        if future is not None:
            future.result(timeout=5)


def test_pending_jobs_reload_as_interrupted_once(runtime, tmp_path):
    session, _ = runtime
    session.state.span_citation_pending = {
        "clip_lost_1": {"path": "first.txt"},
        "clip_lost_2": {"path": "second.txt"},
    }
    path = tmp_path / "pending.json"
    session.save(str(path))
    restored, fresh_service = restore_session(path)
    try:
        state = restored.state
        assert len(state.span_citation_pending) == 2
        assert fresh_service.futures == {}
        check = spans.SpanCompletionCheck(fresh_service)
        check(state)
        replies = [json.loads(message.split("\n", 1)[1]) for message in state.pending_interrupts]
        assert {reply["job_id"] for reply in replies} == {"clip_lost_1", "clip_lost_2"}
        assert len(replies) == 2
        assert all(reply["status"] == "error" and reply["code"] == "interrupted" for reply in replies)
        assert state.span_citation_pending == {} and state.span_citation_records == {}
        check(state)
        assert len(state.pending_interrupts) == 2
    finally:
        restored.state.buffer_manager.cleanup()


def schema_source():
    path = Path(os.environ.get("AZO_SCHEMA_RETURN_SOURCE", str(
        ROOT.parent / "azo-plugin-schema-return" / "src" / "schema_return.py")))
    if not path.is_file():
        pytest.skip("schema-return checkout unavailable; set AZO_SCHEMA_RETURN_SOURCE")
    return path


@pytest.fixture
def schema_plugin(monkeypatch):
    name = "span_test_schema_return"
    spec = importlib.util.spec_from_file_location(name, schema_source())
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_schema_return_reads_and_copies_but_cannot_adopt_or_mutate(citation, schema_plugin):
    session, _, bid, _ = citation
    state = session.state
    plugin = schema_plugin
    original = state.span_citation_records[bid]
    read = plugin.buffer_operation(state, "buffers.read", {"buffer_id": bid})
    assert read["text"] == original
    assert read["readonly"] is True and read["allocated"] is True
    assert read["disk_bound"] is False
    for op, params in [
        ("buffers.adopt", {}),
        ("buffers.replace", {"text": "forged", "if_match": read["etag"]}),
        ("buffers.close", {"discard": True, "if_match": read["etag"]}),
    ]:
        with pytest.raises(plugin.RpcError):
            plugin.buffer_operation(state, op, {"buffer_id": bid, **params})
        assert state.span_citation_records[bid] == original
    assert bid not in plugin.get_data(state)["owned"]
    copy = plugin.buffer_operation(state, "buffers.copy", {"source_id": bid})
    assert copy["buffer_id"].startswith("buf_") and not copy["readonly"]
    plugin.buffer_operation(state, "buffers.replace", {
        "buffer_id": copy["buffer_id"], "text": "editable copy", "if_match": copy["etag"],
    })
    assert state.span_citation_records[bid] == original
    assert not close(state, buffer=bid).error
    with pytest.raises(plugin.RpcError):
        plugin.buffer_operation(state, "buffers.read", {"buffer_id": bid})
    assert plugin.buffer_operation(state, "buffers.read", {
        "buffer_id": copy["buffer_id"],
    })["text"] == "editable copy"


@pytest.mark.parametrize("builder_name", [
    "build_default_pipeline", "build_default_rlm_pipeline", "build_readonly_rlm_pipeline",
])
@pytest.mark.parametrize("with_schema_return", [False, True])
def test_real_pipeline_builds_and_close_order(tmp_path, monkeypatch, builder_name, with_schema_return):
    from agent_zoo import pipelines, plugins
    from agent_zoo.modes import default_modes, rlm_modes

    common = tmp_path / "plugins" / "common"
    common.mkdir(parents=True)
    (common / "span_citations.py").symlink_to(ROOT / "src" / "span_citations.py")
    if with_schema_return:
        (common / "schema_return.py").symlink_to(schema_source())
    monkeypatch.setattr(plugins, "_BASE_DIR", common.parent)
    # Do not load unrelated locally installed entry-point plugins into this test.
    monkeypatch.setattr(plugins, "_iter_entry_points", lambda group: [])
    session = Session()
    modes = default_modes() if builder_name == "build_default_pipeline" else rlm_modes()
    session.pipeline = getattr(pipelines, builder_name)(
        session, config={}, mode_defs=modes, initial_mode="execution",
    )
    session.ensure_initialized()
    try:
        components = flatten(session.pipeline)
        by_name = lambda name: [c for c in components if type(c).__name__ == name]
        assert len(by_name("SpanSetup")) == len(by_name("SpanCompletionCheck")) == 1
        assert len(by_name("ClipSpan")) == len(by_name("CloseCitation")) == 1
        setup, clip, close_citation = (by_name(name)[0] for name in ("SpanSetup", "ClipSpan", "CloseCitation"))
        host_close = next(c for c in components if type(c) is CloseBuffer)
        start = next(c for c in components if isinstance(c, ToolDispatchStart))
        end = next(c for c in components if isinstance(c, ToolDispatchEnd))
        provider = next(c for c in components if isinstance(c, BufferManagerProvider))
        index = components.index
        assert index(provider) < index(setup) < index(start)
        assert index(start) < index(close_citation) < index(host_close) < index(end)
        assert index(start) < index(clip) < index(end)
        tools = [c for c in components if isinstance(c, Tool)]
        names = {c.name for c in tools}
        assert {"clip_span", "view", "grep", "buffer_info", "copy_buffer", "close_buffer"} <= names
        if builder_name == "build_readonly_rlm_pipeline":
            assert "save" not in names and "write_file" not in names
        if with_schema_return:
            assert "submit_result" in names
        setup(session.state)
        state = session.state
        state.span_citation_records["span_order_test"] = '{"immutable":true}'
        handlers = [c for c in tools if c.name == "close_buffer"]
        tc = call(state, handlers, buffer="span_order_test")
        assert not tc.error and "span_order_test" not in state.span_citation_records
        bid, _ = state.buffer_manager.create_scratch_buffer("")
        tc = call(state, handlers, buffer=bid, discard=True)
        assert not tc.error and state.buffer_manager.get(bid) is None
    finally:
        session.state.buffer_manager.cleanup()
