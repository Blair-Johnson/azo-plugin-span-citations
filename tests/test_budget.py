from concurrent.futures import Future
import json
import time

import pytest

import span_citations as spans
from test_integration import call, make_session


def test_actual_half_second_budget_and_no_cancellation(monkeypatch):
    session, service = make_session()
    future = Future()

    def stalled(job_id, *args):
        service.futures[job_id] = future
        return future

    monkeypatch.setattr(service, "start", stalled)
    started = time.monotonic()
    tc = call(session.state, spans.ClipSpan(service), path="example.txt", prefix="P", suffix="S")
    elapsed = time.monotonic() - started
    assert 0.45 <= elapsed < 1.5  # scheduler tolerance, not a hard real-time assertion
    assert json.loads(tc.result)["status"] == "pending"
    assert not future.cancelled() and not future.done()
    future.set_result({"status": "error", "code": "no_match", "message": "not found"})
    spans.SpanCompletionCheck(service)(session.state)
    assert len(session.state.pending_interrupts) == 1


@pytest.mark.parametrize("args", [
    {"path": "", "prefix": "P", "suffix": "S"},
    {"path": "file", "prefix": "", "suffix": "S"},
    {"path": "file", "prefix": "P", "suffix": ""},
    {"path": "file", "prefix": "a" * (spans.MAX_ANCHOR_BYTES + 1), "suffix": "S"},
    {"path": "file", "prefix": "\ud800", "suffix": "S"},
])
def test_invalid_input_creates_no_job(args):
    session, service = make_session()
    tc = call(session.state, spans.ClipSpan(service), **args)
    assert tc.error and json.loads(tc.result)["status"] == "error"
    assert not session.state.span_citation_records
    assert not session.state.span_citation_pending
    assert not service.futures


def test_outstanding_job_limit():
    session, service = make_session()
    session.state.span_citation_pending = {str(i): {} for i in range(spans.MAX_PENDING)}
    tc = call(session.state, spans.ClipSpan(service), path="file", prefix="P", suffix="S")
    assert tc.error and json.loads(tc.result)["code"] == "busy"
    assert not service.futures
