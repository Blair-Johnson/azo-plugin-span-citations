import json

from agent_utils.tools.buffers import StrReplace

import span_citations as spans
from test_integration import call, inline_worker, make_session


def test_unsaved_edits_are_not_cited(tmp_path, monkeypatch):
    session, service = make_session()
    inline_worker(monkeypatch, service)
    state = session.state
    source = tmp_path / "source.txt"
    source.write_text("BEGIN original END")
    buffer_id, _, _ = state.buffer_manager.open_buffer(str(source))
    assert not call(state, StrReplace(), buffer=buffer_id, old="original", new="forged").error
    tc = call(state, spans.ClipSpan(service), path=str(source), prefix="BEGIN", suffix="END")
    assert not tc.error
    bid = json.loads(tc.result)["buffer_id"]
    record = json.loads(state.buffer_manager.resolve_for_read(state, bid).text)
    assert record["span"]["text"] == "BEGIN original END"
    assert state.buffer_manager.get(buffer_id).text == "BEGIN forged END"
    state.session_cwd = str(tmp_path)
    tc = call(state, spans.ClipSpan(service), path=buffer_id, prefix="BEGIN", suffix="END")
    assert tc.error and json.loads(tc.result)["code"] == "io_error"
    state.buffer_manager.close(buffer_id, discard=True)
