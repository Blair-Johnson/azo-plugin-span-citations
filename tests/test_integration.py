"""Integration against installed Agent Zoo APIs; no model, backend, or shell needed."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from agent_utils.core.events import (
    InputReceived,
    IoCompleted,
    RunToolCall,
    SessionRestored,
    ToolCallFinished,
    TurnSettled,
)
from agent_utils.core.pipeline import Pipeline, PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import Entry, ToolCall
from agent_utils.core.tool import Tool
from agent_utils.features.buffers import (
    AppendLines, BufferInfo, BufferManagerProvider, Checkpoint, CloseBuffer, CopyBuffer,
    Grep, InsertLines, PrependLines, RawDiff, RestoreCheckpoint,
    Save, StrReplace, View,
)
from agent_utils.features.core import IoFence, ToolRunner, UnknownToolFallback
from agent_utils.runtime.drivers.timer import TimerDriver
from agent_utils.runtime.loop import Runtime
from agent_utils.storage import (
    apply_fork_policies,
    load_session_json,
    restore_session_document,
    save_session_json,
)
from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io, audit_module_sans_io

import span_citations as spans


ROOT = Path(__file__).resolve().parents[1]


def _noop_emit(_ev):
    return None


def make_session():
    """Initialize real component defaults + Runtime without running an LLM turn."""
    pipeline = Pipeline(
        [
            IoFence(),
            BufferManagerProvider(),
            spans.SpanSetup(),
            spans.SpanStartup(),
            spans.ClipSpan(),
            spans.SpanCompletionCheck(),
            spans.CloseCitation(),
            CloseBuffer(),
        ],
        durable_specs={spans.DURABLE_KEY: spans.SPAN_CITATIONS_DURABLE_SPEC},
    )
    session = Session(pipeline=pipeline)
    session.ensure_initialized()
    session.step(InputReceived(message={}))
    driver = spans.SpanClipDriver()
    runtime = Runtime(session, drivers=[TimerDriver(), driver])
    return session, runtime, driver


def drain_runtime(runtime: Runtime, *, wait_for_io: bool = True, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if runtime.step_once(timeout=0.0):
            continue
        if not wait_for_io or not runtime.session.state.io.in_flight:
            if not runtime.step_once(timeout=0.0):
                break
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for runtime IO to drain")
        runtime.step_once(timeout=min(0.05, remaining))


def call(state, tool, **kwargs):
    """Exercise synchronous Tool.handle(RunToolCall) through event propagation."""
    tools = tool if isinstance(tool, (list, tuple)) else [tool]
    tc = ToolCall(id=f"call-{state.step}", name=tools[0].name,
                  raw_args=json.dumps(kwargs), parsed_args=kwargs)
    state.pending_tool_calls = [tc]
    state.entries.append(Entry(
        messages=[{"role": "assistant", "content": ""}],
        index=len(state.entries), step=state.step, tokens=0, tool_calls=[tc],
    ))
    state.step += 1
    emitted = []
    event = RunToolCall(tc)
    for component in tools:
        if event.stopped:
            break
        component.handle(event, state, emitted.append)
    state._last_emitted = emitted
    assert tc.result is not None, f"Unhandled tool call: {tc.name}"
    return tc


def call_clip(runtime: Runtime, *, timeout: float = 5.0, **kwargs) -> ToolCall:
    """Execute clip_span asynchronously through Runtime + SpanClipDriver + TimerDriver."""
    state = runtime.session.state
    tc = ToolCall(id=f"call-{state.step}", name="clip_span",
                  raw_args=json.dumps(kwargs), parsed_args=kwargs)
    state.pending_tool_calls = [tc]
    state.entries.append(Entry(
        messages=[{"role": "assistant", "content": ""}],
        index=len(state.entries), step=state.step, tokens=0, tool_calls=[tc],
    ))
    state.step += 1
    runtime.post(RunToolCall(tc))
    deadline = time.monotonic() + timeout
    while tc.result is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for clip_span ToolCall result")
        runtime.step_once(timeout=min(0.05, remaining))
    return tc


def close(state, **kwargs):
    return call(state, [spans.CloseCitation(), CloseBuffer()], **kwargs)


@pytest.fixture
def runtime_env(tmp_path):
    session, runtime, driver = make_session()
    session.state.cwd = str(tmp_path)
    with runtime:
        yield session, runtime, driver


@pytest.fixture
def citation(runtime_env, tmp_path):
    session, runtime, driver = runtime_env
    source = tmp_path / "source.txt"
    source.write_bytes("outside é\nBEGIN α\r\n  exact\tEND\nafter".encode("utf-8"))
    tc = call_clip(runtime, path="source.txt", prefix="BEGIN", suffix="END")
    assert not tc.error, tc.result
    reply = json.loads(tc.result)
    assert reply["status"] == "complete"
    assert reply["readonly"] is True
    bid = reply["buffer_id"]
    assert bid.startswith("span_")
    return session, runtime, driver, bid, source


def test_real_read_tools_and_copy_by_value(citation):
    session, runtime, _driver, bid, source = citation
    state = session.state
    bm = state.buffer_manager
    records = spans.span_state(state).records
    original = records[bid]
    record = json.loads(original)
    assert record["span"]["text"] == "BEGIN α\r\n  exact\tEND"
    assert record["file"]["path"] == str(source.resolve())
    assert bm.resolve_for_read(state, bid).readonly is True
    assert bid not in bm.buffers()
    for tool, args, expected in [
        (View(), {"buffer": bid}, "azo.span-citation"),
        (Grep(), {"buffer": bid, "pattern": "exact"}, "exact"),
        (BufferInfo(), {"buffers": [bid]}, "[readonly"),
    ]:
        tc = call(state, tool, **args)
        assert not tc.error, tc.result
        assert expected in tc.result
    target, _ = bm.reserve_buffer()
    tc = call(state, CopyBuffer(), source=bid, target=target)
    assert not tc.error, tc.result
    assert target.startswith("buf_") and target not in records
    assert bm.require_writable(target).text == original
    assert not call(state, AppendLines(), buffer=target, text="copy-only edit").error
    assert records[bid] == original
    # Reads use retained bytes, even after the live file has vanished.
    source.unlink()
    assert bm.resolve_for_read(state, bid).text == original
    assert not call(state, View(), buffer=bid).error
    assert not close(state, buffer=bid).error
    assert "copy-only edit" in bm.require_writable(target).text
    assert not close(state, buffer=target, discard=True).error
    session.step(TurnSettled())
    assert getattr(state, "pending_interrupts", []) == []


@pytest.mark.parametrize("tool,kwargs", [
    (StrReplace, {"old": "exact", "new": "forged"}),
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
    session, _runtime, _driver, bid, source = citation
    state = session.state
    records = spans.span_state(state).records
    original = records[bid]
    original_disk = source.read_bytes()
    args = {"buffer": bid, **kwargs}
    output = tmp_path / "must-not-write.json"
    if tool is Save:
        args["path"] = str(output)
    tc = call(state, tool(), **args)
    assert tc.error, tc.result
    assert "readonly" in tc.result.lower(), tc.result
    assert records[bid] == original
    assert state.buffer_manager.resolve_for_read(state, bid).text == original
    assert source.read_bytes() == original_disk
    assert not output.exists()
    assert state.buffer_manager.buffers() == []


@pytest.mark.parametrize("overwrite", [False, True])
def test_copy_cannot_overwrite_citation(citation, overwrite):
    session, _runtime, _driver, bid, _ = citation
    state = session.state
    records = spans.span_state(state).records
    original = records[bid]
    source, _ = state.buffer_manager.create_scratch_buffer("forged")
    tc = call(state, CopyBuffer(), source=source, target=bid, overwrite=overwrite)
    assert tc.error and "readonly" in tc.result.lower(), tc.result
    assert records[bid] == original


def test_close_deletes_immediately_and_notifies_once(citation):
    session, _runtime, _driver, bid, _ = citation
    state = session.state
    records = spans.span_state(state).records
    tc = close(state, buffer=f"  {bid}  ")
    assert not tc.error, tc.result
    finished = [ev for ev in state._last_emitted if isinstance(ev, ToolCallFinished)]
    assert len(finished) == 1 and finished[0].tool_call is tc
    assert bid not in records
    assert call(state, View(), buffer=bid).error
    assert call(state, Grep(), buffer=bid, pattern="exact").error
    assert call(state, BufferInfo(), buffers=[bid]).error
    target, _ = state.buffer_manager.reserve_buffer()
    assert call(state, CopyBuffer(), source=bid, target=target).error
    assert state.buffer_manager.is_reserved(target)
    tc = close(state, buffer=bid, discard=True)
    assert tc.error and "No citation buffer" in tc.result


@pytest.mark.parametrize("kind", ["clean", "dirty", "scratch", "reserved", "omitted", "unknown"])
def test_plain_close_semantics_are_unchanged(runtime_env, tmp_path, kind):
    session, _runtime, _driver = runtime_env
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
        bid, _, _ = bm.open_from_text(str(path), path.read_text(encoding="utf-8"))
    if kind == "dirty":
        assert not call(state, AppendLines(), buffer=bid, text="unsaved").error
    args = {} if kind == "omitted" else {"buffer": bid}
    # The citation layer must leave a non-citation call completely unhandled.
    tc = ToolCall(id="probe", name="close_buffer", raw_args="{}", parsed_args=args)
    state.pending_tool_calls = [tc]
    probe_event = RunToolCall(tc)
    spans.CloseCitation().handle(probe_event, state, _noop_emit)
    assert tc.result is None and not tc.error and not probe_event.stopped
    result = close(state, **args)
    assert result.error == (kind in {"dirty", "scratch", "unknown"}), result.result
    if kind in {"dirty", "scratch"}:
        assert bm.get(bid) is not None
        assert not close(state, buffer=bid, discard=True).error
    assert bm.get(bid) is None


def restore_session(path, *, emit_restored: bool = False):
    session, runtime, driver = make_session()
    doc = load_session_json(path, verify_blobs=True)
    restore_session_document(doc, session.pipeline, state=session.state)
    session.ensure_initialized()
    if emit_restored:
        session.step(SessionRestored(reason="reload"))
    else:
        session.step(InputReceived(message={}))
    return session, runtime, driver


def test_json_session_roundtrip_preserves_readonly_and_close_deletion(citation, tmp_path):
    session, _runtime, _driver, bid, source = citation
    original = spans.span_state(session.state).records[bid]
    path = tmp_path / "session.json"
    save_session_json(session.state, session.pipeline, path)
    document = json.loads(path.read_text())
    assert document["features"]["span_citations"]["__durable_v__"] == 1
    assert document["features"]["span_citations"]["__data__"]["records"][bid] == original
    assert "span_citation_records" not in document["attrs"]
    source.unlink()
    restored, _rt, _drv = restore_session(path, emit_restored=True)
    state = restored.state
    assert spans.span_state(state).records == {bid: original}
    assert state.buffer_manager.resolve_for_read(state, bid).readonly
    assert not call(state, View(), buffer=bid).error
    tc = call(state, AppendLines(), buffer=bid, text="forged")
    assert tc.error and "readonly" in tc.result.lower()
    assert spans.span_runtime(state).active_ops == {}
    assert not close(state, buffer=bid).error
    save_session_json(restored.state, restored.pipeline, path)
    again, _rt2, _drv2 = restore_session(path, emit_restored=True)
    assert spans.span_state(again.state).records == {}
    assert call(again.state, View(), buffer=bid).error


def test_on_fork_copies_records_and_clears_pending_and_notices(runtime_env):
    session, _runtime, _driver = runtime_env
    durable = spans.span_state(session.state)
    durable.records["span_keep"] = '{"ok": true}'
    durable.pending["clip_inflight"] = {"path": "a.txt"}
    durable.notices["clip_done"] = "[Span clipping completed | clip_done]\n{}"

    apply_fork_policies(session.state, session.pipeline)
    forked = spans.span_state(session.state)
    assert forked.records == {"span_keep": '{"ok": true}'}
    assert forked.records is not durable.records
    assert forked.pending == {}
    assert forked.notices == {}


@pytest.mark.parametrize("success", [True, False])
def test_inline_completion_has_no_interrupt(runtime_env, tmp_path, success):
    session, runtime, _driver = runtime_env
    state = session.state
    source = tmp_path / "inline.txt"
    source.write_text("BEGIN content END", encoding="utf-8")
    tc = call_clip(runtime, path=str(source),
                   prefix="BEGIN" if success else "missing", suffix="END")
    reply = json.loads(tc.result)
    assert reply["status"] == ("complete" if success else "error")
    assert tc.error is (not success)
    if not success:
        assert reply["code"] == "no_match"
    for _ in range(3):
        session.step(TurnSettled())
    assert getattr(state, "pending_interrupts", []) == []
    assert spans.span_state(state).pending == {}
    assert spans.span_runtime(state).active_ops == {}
    assert len(spans.span_state(state).records) == int(success)


@pytest.mark.parametrize("success", [True, False])
def test_deferred_worker_delivers_exactly_one_completion(runtime_env, tmp_path, monkeypatch, success):
    session, runtime, _driver = runtime_env
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
    try:
        tc = call_clip(runtime, path=str(source),
                       prefix="BEGIN" if success else "missing", suffix="END")
        assert not tc.error
        pending = json.loads(tc.result)
        assert pending["status"] == "pending"
        job = pending["job_id"]
        assert entered.wait(5)
        assert spans.span_state(state).pending == {job: {"path": str(source)}}
        assert job in spans.span_runtime(state).job_ops
        session.step(TurnSettled())
        assert getattr(state, "pending_interrupts", []) == []
        assert spans.span_state(state).records == {}
        release.set()
        drain_runtime(runtime, wait_for_io=True, timeout=5.0)
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
            session.step(TurnSettled())
        assert len(state.pending_interrupts) == 1
        assert spans.span_state(state).pending == {}
        assert spans.span_runtime(state).active_ops == {}
        assert len(spans.span_state(state).records) == int(success)
    finally:
        release.set()


def test_pending_jobs_reload_as_interrupted_once_on_session_restored(runtime_env, tmp_path):
    session, _runtime, _driver = runtime_env
    spans.span_state(session.state).pending = {
        "clip_lost_1": {"path": "first.txt"},
        "clip_lost_2": {"path": "second.txt"},
    }
    path = tmp_path / "pending.json"
    save_session_json(session.state, session.pipeline, path)
    restored, _rt, _drv = restore_session(path, emit_restored=True)
    state = restored.state
    replies = [json.loads(message.split("\n", 1)[1]) for message in state.pending_interrupts]
    assert {reply["job_id"] for reply in replies} == {"clip_lost_1", "clip_lost_2"}
    assert len(replies) == 2
    assert all(reply["status"] == "error" and reply["code"] == "interrupted" for reply in replies)
    assert spans.span_state(state).pending == {} and spans.span_state(state).records == {}
    restored.step(TurnSettled())
    assert len(state.pending_interrupts) == 2


def test_span_clip_driver_cancel_quiesce_and_close(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    worker = spans._worker

    def gated_worker(*args):
        entered.set()
        release.wait(5)
        return worker(*args)

    monkeypatch.setattr(spans, "_worker", gated_worker)
    source = tmp_path / "sample.txt"
    source.write_text("BEGIN hello END", encoding="utf-8")
    emitted = []
    driver = spans.SpanClipDriver(max_workers=2)
    assert driver.durability == "ephemeral"
    assert driver.request_types == (spans.ClipSpanRequest,)
    driver.start(emitted.append)
    try:
        req = spans.ClipSpanRequest(job_id="clip_cancel", path=str(source), prefix="BEGIN", suffix="END", cwd=str(tmp_path))
        driver.submit("g1:op_1", req)
        assert entered.wait(2)
        driver.cancel("g1:op_1")
        release.set()
        time.sleep(0.05)
        assert emitted == []
        assert driver.quiesce(policy="terminate") == []
    finally:
        release.set()
        driver.close()


def test_module_and_pipeline_sans_io_clean():
    assert audit_module_sans_io(ROOT / "src" / "span_citations.py") == []
    session, _runtime, _driver = make_session()
    assert_pipeline_sans_io(
        session.pipeline,
        session=session,
        events=[
            SessionRestored(reason="startup"),
            TurnSettled(),
        ],
    )


@pytest.mark.parametrize("builder_name", [
    "build_default_pipeline", "build_default_rlm_pipeline", "build_readonly_rlm_pipeline",
])
def test_real_pipeline_builds_and_close_order(tmp_path, monkeypatch, builder_name):
    from agent_zoo import pipelines
    from agent_zoo.features.modes import default_modes, rlm_modes
    from agent_zoo.plugins import loader as plugins_loader

    common = tmp_path / "plugins" / "common"
    common.mkdir(parents=True)
    (common / "span_citations.py").symlink_to(ROOT / "src" / "span_citations.py")
    monkeypatch.setattr(plugins_loader, "_BASE_DIR", common.parent)
    # Do not load unrelated locally installed entry-point plugins into this test.
    monkeypatch.setattr(plugins_loader, "_iter_entry_points", lambda group: [])
    session = Session()
    modes = default_modes() if builder_name == "build_default_pipeline" else rlm_modes()
    session.pipeline = getattr(pipelines, builder_name)(
        session, config={}, mode_defs=modes, initial_mode="execution",
        include_llm_drivers=False,
    )
    session.ensure_initialized()
    assert_pipeline_sans_io(session.pipeline)
    components = list(session.pipeline.components)
    by_name = lambda name: [c for c in components if type(c).__name__ == name]
    assert len(by_name("SpanSetup")) == len(by_name("SpanCompletionCheck")) == 1
    assert len(by_name("ClipSpan")) == len(by_name("CloseCitation")) == 1
    setup, clip, close_citation = (by_name(name)[0] for name in ("SpanSetup", "ClipSpan", "CloseCitation"))
    host_close = next(c for c in components if type(c) is CloseBuffer)
    runner = next(c for c in components if isinstance(c, ToolRunner))
    fallback = next(c for c in components if isinstance(c, UnknownToolFallback))
    provider = next(c for c in components if isinstance(c, BufferManagerProvider))
    index = components.index
    assert index(provider) < index(setup) < index(runner)
    assert index(runner) < index(close_citation) < index(host_close) < index(fallback)
    assert index(runner) < index(clip) < index(fallback)
    tools = [c for c in components if isinstance(c, Tool)]
    names = {c.name for c in tools}
    assert {"clip_span", "view", "grep", "close_buffer"} <= names
    assert sum(1 for c in tools if c.name == "close_buffer") == 1
    if builder_name == "build_readonly_rlm_pipeline":
        assert "save" not in names and "write_file" not in names
    setup.handle(InputReceived(message={}), session.state, _noop_emit)
    state = session.state
    records = spans.span_state(state).records
    records["span_order_test"] = '{"immutable":true}'
    handlers = [c for c in components if getattr(c, "name", None) == "close_buffer"]
    tc = call(state, handlers, buffer="span_order_test")
    assert not tc.error and "span_order_test" not in records
    bid, _ = state.buffer_manager.create_scratch_buffer("")
    tc = call(state, handlers, buffer=bid, discard=True)
    assert not tc.error and state.buffer_manager.get(bid) is None
