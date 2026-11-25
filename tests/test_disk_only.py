import json

from agent_utils.features.buffers import StrReplace

from test_integration import call, call_clip, make_session


def test_unsaved_edits_are_not_cited(tmp_path):
    session, runtime, _driver = make_session()
    state = session.state
    source = tmp_path / "source.txt"
    source.write_text("BEGIN original END")
    with runtime:
        buffer_id, _, _ = state.buffer_manager.open_from_text(str(source), source.read_text())
        assert not call(state, StrReplace(), buffer=buffer_id, old="original", new="forged").error
        tc = call_clip(runtime, path=str(source), prefix="BEGIN", suffix="END")
        assert not tc.error
        bid = json.loads(tc.result)["buffer_id"]
        record = json.loads(state.buffer_manager.resolve_for_read(state, bid).text)
        assert record["span"]["text"] == "BEGIN original END"
        assert state.buffer_manager.get(buffer_id).text == "BEGIN forged END"
        state.cwd = str(tmp_path)
        tc = call_clip(runtime, path=buffer_id, prefix="BEGIN", suffix="END")
        assert tc.error and json.loads(tc.result)["code"] == "io_error"
        state.buffer_manager.close(buffer_id, discard=True)
