from agent_utils.core.events import IoCompleted, SessionRestored, TurnSettled
from agent_utils.core.pipeline import Pipeline
from agent_utils.core.state import Entry
from agent_utils.features.interrupts import InterruptDelivery
from agent_utils.storage import capture_session_document, restore_session_document
from agent_zoo.events import SessionReady

import span_citations as spans
from test_integration import make_session


def _noop_emit(_ev):
    return None


def completed(session):
    state = session.state
    durable = spans.span_state(state)
    rt = spans.span_runtime(state)
    durable.pending["clip_test"] = {"path": "/example"}
    op_id = state.io.submit(
        spans.ClipSpanRequest(job_id="clip_test", path="/example", prefix="P", suffix="S"),
        tag="span_citations",
    )
    rt.active_ops[op_id] = "clip_test"
    rt.job_ops["clip_test"] = op_id
    state.io.take_outbox()
    spans.SpanCompletionCheck().handle(
        IoCompleted(op_id=op_id, value={"status": "complete", "citation_json": '{"snapshot": true}'}),
        state,
        _noop_emit,
    )
    return next(iter(durable.records))


def restored(session, *, emit_restored: bool = False):
    snapshot = capture_session_document(session.state, session.pipeline)
    new, runtime, driver = make_session()
    restore_session_document(snapshot, new.pipeline, state=new.state)
    new.ensure_initialized()
    if emit_restored:
        spans.SpanStartup().handle(SessionRestored(reason="reload"), new.state, _noop_emit)
    else:
        spans.SpanStartup().handle(SessionReady(), new.state, _noop_emit)
    return new, runtime, driver


def test_buffered_notice_recovers_after_reload():
    session, _runtime, _driver = make_session()
    session.state.entries.append(Entry(messages=[{"role": "user", "content": "working"}], index=0, step=0))
    bid = completed(session)
    delivery = InterruptDelivery()
    delivery.handle(TurnSettled(), session.state, _noop_emit)
    assert len(session.state.pending_interrupts) == 1
    assert len(session.state.entries) == 1
    reloaded, _rt, _drv = restored(session)
    state = reloaded.state
    assert state.pending_interrupts == []
    assert state.buffer_manager.resolve_for_read(state, bid).text == '{"snapshot": true}'
    check = spans.SpanCompletionCheck()
    check.handle(TurnSettled(), state, _noop_emit)
    check.handle(TurnSettled(), state, _noop_emit)
    assert len(state.pending_interrupts) == 1
    state.entries.append(Entry(messages=[{"role": "assistant", "content": "ready"}], index=len(state.entries), step=state.step))
    InterruptDelivery().handle(TurnSettled(), state, _noop_emit)
    check.handle(TurnSettled(), state, _noop_emit)
    assert spans.span_state(state).notices == {}
    assert state.pending_interrupts == []
    assert "clip_test" in state.entries[-1].messages[0]["content"]


def test_delivered_notice_does_not_repeat_after_reload():
    session, _runtime, _driver = make_session()
    completed(session)
    InterruptDelivery().handle(TurnSettled(), session.state, _noop_emit)
    assert spans.span_state(session.state).notices  # save before the next cleanup check
    reloaded, _rt, _drv = restored(session, emit_restored=True)
    spans.SpanCompletionCheck().handle(TurnSettled(), reloaded.state, _noop_emit)
    assert reloaded.state.pending_interrupts == []
    assert spans.span_state(reloaded.state).notices == {}


def test_checks_only_queue_then_pipeline_delivery():
    session, _runtime, _driver = make_session()
    delivery = InterruptDelivery()
    session.pipeline = Pipeline(
        [*session.pipeline.components, delivery],
        durable_specs=session.pipeline.durable_specs,
    )
    session.state.done = True
    completed(session)
    before_step = session.state.step
    assert len(session.state.pending_interrupts) == 1
    assert session.state.done is True
    assert not session.state.entries
    assert session.state.step == before_step
    session.step(TurnSettled())
    assert session.state.done is False
    assert len(session.state.entries) == 1
    session.step(TurnSettled())
    assert len(session.state.entries) == 1


def test_malformed_io_completed_result_returns_error_not_keyerror():
    session, _runtime, _driver = make_session()
    state = session.state
    durable = spans.span_state(state)
    rt = spans.span_runtime(state)
    durable.pending["clip_bad"] = {"path": "/example"}
    op_id = state.io.submit(
        spans.ClipSpanRequest(job_id="clip_bad", path="/example", prefix="P", suffix="S"),
        tag="span_citations",
    )
    rt.active_ops[op_id] = "clip_bad"
    rt.job_ops["clip_bad"] = op_id
    state.io.take_outbox()
    spans.SpanCompletionCheck().handle(
        IoCompleted(op_id=op_id, value={"ok": True}),
        state,
        _noop_emit,
    )
    assert "clip_bad" not in durable.pending
    assert "clip_bad" in durable.notices
    assert '"code": "internal_error"' in durable.notices["clip_bad"]

