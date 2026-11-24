from concurrent.futures import Future

from agent_utils.core.events import TurnSettled
from agent_utils.core.pipeline import Pipeline
from agent_utils.core.state import Entry
from agent_utils.features.interrupts import InterruptDelivery
from agent_utils.storage import capture_session_document, restore_session_document
from agent_zoo.events import SessionReady

import span_citations as spans
from test_integration import make_session


def _noop_emit(_ev):
    return None


def completed(session, service):
    state = session.state
    future = Future()
    future.set_result({"status": "complete", "citation_json": '{"snapshot": true}'})
    service.futures["clip_test"] = future
    state.span_citation_pending["clip_test"] = {"path": "/example"}
    spans.SpanCompletionCheck(service).handle(TurnSettled(), state, _noop_emit)
    return next(iter(state.span_citation_records))


def restored(session):
    snapshot = capture_session_document(session.state, session.pipeline)
    new, service = make_session()
    restore_session_document(snapshot, new.pipeline, state=new.state)
    new.ensure_initialized()
    spans.SpanStartup().handle(SessionReady(), new.state, _noop_emit)
    return new, service


def test_buffered_notice_recovers_after_reload():
    session, service = make_session()
    session.state.entries.append(Entry(messages=[{"role": "user", "content": "working"}], index=0, step=0))
    bid = completed(session, service)
    delivery = InterruptDelivery()
    delivery.handle(TurnSettled(), session.state, _noop_emit)
    assert len(session.state.pending_interrupts) == 1
    assert len(session.state.entries) == 1
    reloaded, new_service = restored(session)
    state = reloaded.state
    assert state.pending_interrupts == []
    assert state.buffer_manager.resolve_for_read(state, bid).text == '{"snapshot": true}'
    check = spans.SpanCompletionCheck(new_service)
    check.handle(TurnSettled(), state, _noop_emit)
    check.handle(TurnSettled(), state, _noop_emit)
    assert len(state.pending_interrupts) == 1
    state.entries.append(Entry(messages=[{"role": "assistant", "content": "ready"}], index=len(state.entries), step=state.step))
    InterruptDelivery().handle(TurnSettled(), state, _noop_emit)
    check.handle(TurnSettled(), state, _noop_emit)
    assert state.span_citation_notices == {}
    assert state.pending_interrupts == []
    assert "clip_test" in state.entries[-1].messages[0]["content"]


def test_delivered_notice_does_not_repeat_after_reload():
    session, service = make_session()
    completed(session, service)
    InterruptDelivery().handle(TurnSettled(), session.state, _noop_emit)
    assert session.state.span_citation_notices  # save before the next cleanup check
    reloaded, new_service = restored(session)
    spans.SpanCompletionCheck(new_service).handle(TurnSettled(), reloaded.state, _noop_emit)
    assert reloaded.state.pending_interrupts == []
    assert reloaded.state.span_citation_notices == {}


def test_checks_only_queue_then_pipeline_delivery():
    session, service = make_session()
    delivery = InterruptDelivery()
    session.pipeline = Pipeline([*session.pipeline.components, delivery])
    session.state.done = True
    completed(session, service)
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
