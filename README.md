# Span citations

`clip_span(path, prefix, suffix)` reads a disk file and returns a `span_*` buffer ID containing immutable JSON provenance. It is a common-scope Agent Zoo plugin, available to interactive agents and RLMs, with no third-party runtime dependencies beyond the host.

## Use

```python
clip_span(path="notes.txt", prefix="The experiment", suffix="was repeated.")
view(buffer="span_<returned-id>")
close_buffer(buffer="span_<returned-id>")
```

The tool accepts a file path, never editable buffer contents. Relative paths use the session's cwd captured at invocation; `~` is expanded. Input must be a regular UTF-8 text file. The excerpt includes both anchors and preserves original spelling, Unicode, whitespace, BOMs, and line endings. No fuzzy matching or normalization is performed.

A span starts at an exact prefix occurrence and ends after an exact suffix occurrence. The suffix must start at or after the prefix start and end at or after the prefix end; anchors may overlap or coincide. There must be exactly one such ordered pair in the entire file. Missing and ambiguous matches fail without creating a citation. A repeated anchor is acceptable only when the pair is unique.

## Provenance

Each JSON document has `schema: "azo.span-citation"`, `version: 1`, and `file`, `match`, and `span` objects. `file.path` is the resolved absolute path; `file.size_bytes` and `file.hash` describe the entire raw-byte snapshot. The hash has `algorithm: "sha256"` and its hexadecimal digest in `hex`. `match` records the supplied anchors, exact matching mode, and anchor inclusion. `span.text` contains the extracted text; `byte_start`/`byte_end` and `char_start`/`char_end` are zero-based, half-open offsets. Character offsets count Unicode code points. Line numbers are one-based and count LF delimiters, including the LF in CRLF.

Hashing, matching, and extraction use one private temporary snapshot. The scanner holds bounded-size chunks rather than loading the whole source into memory. It checks file identity, size, and modification metadata before and after reading and rejects detected changes. This is not an operating-system atomic snapshot against adversarial writers; the digest identifies the actual bytes read. SHA-256 is a content fingerprint, not a digital signature or proof of authorship.

Citation IDs live in a protected readonly namespace. `view`, `grep`, `buffer_info`, and copy-from operations can read them; edits, save, checkpoint restore, and overwrite-copy cannot modify them. Writable copies have ordinary `buf_*` IDs and are not trusted citations. Downstream consumers should resolve the original `span_*` ID from the producing session, not trust agent-supplied JSON or a copied buffer. Schema-return websocket clients can use their existing `buffers.read` operation to retrieve the original ID, paging as needed; no new websocket protocol is required. This plugin does not add semantic citation validation to arbitrary JSON schemas.

## Completion and lifetime

The tool waits up to 0.5 seconds for scanning, hashing, and JSON construction. If still running, it returns `status: "pending"` and a `clip_*` job ID. The normal harness interrupt polling later delivers exactly one completion containing the citation buffer ID or a structured error. The wait budget is not a hard real-time scheduling guarantee. Workers never mutate harness state. At most four outstanding jobs are accepted per runtime; anchors are limited to 64 KiB each and extracted spans to 16 MiB. Temporary disk space scales with the file size.

Completed records survive session save/load and are not idle-auto-closed. `close_buffer(buffer=...)` explicitly releases a citation; its old ID no longer resolves. Closing does not erase copies or prior transcript text. In-flight jobs do not survive backend restart: the restored session reports them as interrupted, without silently rereading the source. Undelivered completion notices are saved and requeued after reload, with transcript markers preventing duplicate delivery. Durability is limited to the most recent successful session save. Citation IDs are session-scoped references, not portable public URLs.

## Install and test

From an Agent Zoo environment, install this repository with `azo-plugin install /path/to/azo-plugin-span-citations`, then run `/reload` in the TUI. The plugin's `pixi.toml` places `src/span_citations.py` in common scope. No core harness or schema-return modifications are needed.

Run `pixi run test` from this repository. The test launcher uses the installed Agent Zoo Pixi environment, whose dependencies include pytest. Override `AZO_HOST_MANIFEST` or `AZO_HOST_ENVIRONMENT` for a different host. Tests cover matching, provenance, mutation rejection, persistence, deferred completion, pipeline composition, and schema-return buffer reads.
