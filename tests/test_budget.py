import json
import time

import pytest

from agent_utils.core.events import IoCompleted
from agent_utils.runtime.driver import Driver
from agent_utils.runtime.drivers.timer import TimerDriver
from agent_utils.runtime.loop import Runtime

import span_citations as spans
from test_integration import call, call_clip, drain_runtime, make_session


class _StalledClipDriver(Driver):
    durability = "ephemeral"
    request_types = (spans.ClipSpanRequest,)

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[tuple[str, spans.ClipSpanRequest]] = []
        self.cancelled: list[str] = []

    def submit(self, op_id, req) -> None:
        self.submitted.append((op_id, req))

    def cancel(self, op_id) -> None:
        self.cancelled.append(op_id)

    def complete(self, op_id: str, value: dict) -> None:
        self._safe_emit(IoCompleted(op_id=op_id, value=value))


def test_actual_half_second_budget_and_no_cancellation():
    session, _runtime, _driver = make_session()
    stalled = _StalledClipDriver()
    runtime = Runtime(session, drivers=[TimerDriver(), stalled])
    with runtime:
        started = time.monotonic()
        tc = call_clip(runtime, path="example.txt", prefix="P", suffix="S")
        elapsed = time.monotonic() - started
        assert 0.45 <= elapsed < 1.5  # scheduler tolerance, not a hard real-time assertion
        assert json.loads(tc.result)["status"] == "pending"
        assert len(stalled.submitted) == 1
        assert stalled.cancelled == []
        op_id, _req = stalled.submitted[0]
        stalled.complete(op_id, {"status": "error", "code": "no_match", "message": "not found"})
        drain_runtime(runtime, wait_for_io=True)
        assert len(session.state.pending_interrupts) == 1


@pytest.mark.parametrize("args", [
    {"path": "", "prefix": "P", "suffix": "S"},
    {"path": "file", "prefix": "", "suffix": "S"},
    {"path": "file", "prefix": "P", "suffix": ""},
    {"path": "file", "prefix": "a" * (spans.MAX_ANCHOR_BYTES + 1), "suffix": "S"},
    {"path": "file", "prefix": "\ud800", "suffix": "S"},
])
def test_invalid_input_creates_no_job(args):
    session, _runtime, _driver = make_session()
    tc = call(session.state, spans.ClipSpan(), **args)
    assert tc.error and json.loads(tc.result)["status"] == "error"
    assert not spans.span_state(session.state).records
    assert not spans.span_state(session.state).pending
    assert not session.state.io.in_flight


def test_outstanding_job_limit():
    session, _runtime, _driver = make_session()
    spans.span_state(session.state).pending = {str(i): {} for i in range(spans.MAX_PENDING)}
    tc = call(session.state, spans.ClipSpan(), path="file", prefix="P", suffix="S")
    assert tc.error and json.loads(tc.result)["code"] == "busy"
    assert not session.state.io.in_flight
