from concurrent.futures import Future
from types import SimpleNamespace

from agent_utils.components import InterruptDelivery
from agent_utils.serialization import capture_session_document
from agent_utils.types import Entry

import span_citations as spans
from test_integration import make_session


def completed(session, service):
    state = session.state
    future = Future()
    future.set_result({"status": "complete", "citation_json": '{"snapshot": true}'})
    service.futures["clip_test"] = future
    state.span_citation_pending["clip_test"] = {"path": "/example"}
    spans.SpanCompletionCheck(service)(state)
    return next(iter(state.span_citation_records))


def restored(session):
    snapshot = capture_session_document(session.state, session.pipeline)
    new, service = make_session()
    new.load_document(snapshot)
    new.ensure_initialized()
    spans.SpanStartup().handle_harness_event(SimpleNamespace(kind="runtime.session_ready"), new.state)
    return new, service


def test_buffered_notice_recovers_after_reload():
    session, service = make_session()
    session.state.entries.append(Entry(messages=[{"role": "user", "content": "working"}], index=0, step=0))
    bid = completed(session, service)
    delivery = InterruptDelivery()
    delivery(session.state)
    assert session.state.pending_interrupts == []
    assert len(delivery._buffered) == 1
    reloaded, new_service = restored(session)
    state = reloaded.state
    assert state.buffer_manager.resolve_for_read(state, bid).text == '{"snapshot": true}'
    check = spans.SpanCompletionCheck(new_service)
    check(state)
    check(state)
    assert len(state.pending_interrupts) == 1
    state.entries.append(Entry(messages=[{"role": "assistant", "content": "ready"}], index=len(state.entries), step=state.step))
    InterruptDelivery()(state)
    check(state)
    assert state.span_citation_notices == {}
    assert state.pending_interrupts == []
    assert "clip_test" in state.entries[-1].messages[0]["content"]


def test_delivered_notice_does_not_repeat_after_reload():
    session, service = make_session()
    completed(session, service)
    InterruptDelivery()(session.state)
    assert session.state.span_citation_notices  # save before the next cleanup check
    reloaded, new_service = restored(session)
    spans.SpanCompletionCheck(new_service)(reloaded.state)
    assert reloaded.state.pending_interrupts == []
    assert reloaded.state.span_citation_notices == {}


def test_checks_only_queue_then_normal_poll_delivers():
    session, service = make_session()
    session.pipeline.append(InterruptDelivery())
    session.state.done = True
    completed(session, service)
    before_step = session.state.step
    session.poll_interrupts(deliver=False)
    assert session.state.done is True
    assert not session.state.entries
    assert session.state.step == before_step
    assert session.poll_interrupts() is True
    assert session.state.done is False
    session.poll_interrupts()
    assert len(session.state.entries) == 1
