import hashlib
import io
import itertools

import pytest

import span_citations as spans


def scan(text, prefix, suffix, chunk_size, monkeypatch):
    monkeypatch.setattr(spans, "CHUNK_BYTES", chunk_size)
    return spans._scan_snapshot(io.BytesIO(text.encode()), io.BytesIO(), prefix.encode(), suffix.encode())


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 1024])
@pytest.mark.parametrize("text,prefix,suffix,start,end", [
    ("before START\r\nquote END after", "START", "END", 7, 23),
    ("éλ abcde!", "abc", "cde", 5, 10),
    ("xx abc yy", "abc", "abc", 3, 6),
    ("S irrelevant P hit S P", "P", "S", 13, 20),
    ("abcdef", "abcde", "def", 0, 6),
])
def test_exact_chunks(text, prefix, suffix, start, end, chunk_size, monkeypatch):
    result = scan(text, prefix, suffix, chunk_size, monkeypatch)
    assert result == (start, end, len(text.encode()), hashlib.sha256(text.encode()).hexdigest())


@pytest.mark.parametrize("text,prefix,suffix,code", [
    ("P S P S", "P", "S", "ambiguous"),
    ("P P S", "P", "S", "ambiguous"),
    ("P S S", "P", "S", "ambiguous"),
    ("aaaa", "aa", "aa", "ambiguous"),
    ("S P", "P", "S", "no_match"),
    ("nothing", "P", "S", "no_match"),
    ("abcde", "abcde", "abc", "no_match"),
])
def test_rejected(text, prefix, suffix, code, monkeypatch):
    with pytest.raises(spans.ClipError) as caught:
        scan(text, prefix, suffix, 2, monkeypatch)
    assert caught.value.code == code


def test_exhaustive_against_brute_force(monkeypatch):
    anchors = ["a", "b", "aa", "ab", "ba", "bb", "aaa"]
    for length in range(7):
        for chars in itertools.product("ab", repeat=length):
            text = "".join(chars)
            for prefix, suffix in itertools.product(anchors, repeat=2):
                pairs = [(i, j + len(suffix)) for i in range(len(text)) for j in range(i, len(text))
                         if text.startswith(prefix, i) and text.startswith(suffix, j)
                         and j + len(suffix) >= i + len(prefix)]
                try:
                    result = scan(text, prefix, suffix, 2, monkeypatch)
                except spans.ClipError as exc:
                    assert exc.code == ("no_match" if not pairs else "ambiguous")
                    assert len(pairs) != 1
                else:
                    assert pairs == [result[:2]], (text, prefix, suffix, pairs, result)


def test_provenance_and_offsets(tmp_path):
    path = tmp_path / "quote.txt"
    raw = "é\r\n  START λ\r\nEND\ntrailer".encode()
    path.write_bytes(raw)
    record = spans.extract_citation("quote.txt", "START", "END", str(tmp_path))
    assert record["file"]["path"] == str(path)
    assert record["file"]["hash"] == {"algorithm": "sha256", "hex": hashlib.sha256(raw).hexdigest()}
    span = record["span"]
    assert span["text"] == "START λ\r\nEND"
    assert raw[span["byte_start"]:span["byte_end"]].decode() == span["text"]
    assert raw.decode()[span["char_start"]:span["char_end"]] == span["text"]
    assert (span["line_start"], span["line_end"]) == (2, 3)
    path.write_text("changed")
    assert record["span"] == span


@pytest.mark.parametrize("kind", ["mutate", "replace"])
def test_detects_source_change(tmp_path, monkeypatch, kind):
    path = tmp_path / "source"
    path.write_text("PREFIX text SUFFIX")
    original = spans._scan_snapshot

    def mutate(*args):
        result = original(*args)
        if kind == "replace":
            other = tmp_path / "other"
            other.write_bytes(path.read_bytes())
            other.replace(path)
        else:
            path.write_text("PREFIX changed SUFFIX")
        return result

    monkeypatch.setattr(spans, "_scan_snapshot", mutate)
    with pytest.raises(spans.ClipError, match="changed"):
        spans.extract_citation(str(path), "PREFIX", "SUFFIX", str(tmp_path))


def test_invalid_utf8_even_outside_span(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"P content S\xff")
    assert spans._worker(str(path), "P", "S", str(tmp_path))["code"] == "invalid_utf8"


def test_size_limit(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_text("P content S")
    monkeypatch.setattr(spans, "MAX_SPAN_BYTES", 5)
    assert spans._worker(str(path), "P", "S", str(tmp_path))["code"] == "span_too_large"


def test_missing_and_nonregular(tmp_path):
    assert spans._worker(str(tmp_path / "missing"), "P", "S", str(tmp_path))["code"] == "io_error"
    assert spans._worker(str(tmp_path), "P", "S", str(tmp_path))["status"] == "error"


def test_lf_end_and_bom(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"\xef\xbb\xbfP\nS\n")
    record = spans.extract_citation(str(path), "\ufeffP", "S\n", str(tmp_path))
    assert record["span"]["char_start"] == record["span"]["byte_start"] == 0
    assert record["span"]["line_end"] == 2
