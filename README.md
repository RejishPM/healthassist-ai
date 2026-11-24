# HealthAssist RAG MVP

A local FastAPI application using the existing PDF extractor/chunker,
Ollama embeddings, PostgreSQL/pgvector retrieval, and local Ollama generation.
`POST /documents` ingests a PDF; `POST /questions` returns source-grounded excerpts
with document, version, page, chunk, and similarity metadata.

The LLM selects verbatim passages rather than inventing a free-form answer.
The server verifies every quotation against the actual context sent to the model
and builds the answer and citations itself. Unknown sources or fabricated quotes
are withheld with HTTP 502. No matching context returns a clear abstention without
calling generation. This verifies provenance, not the clinical truth or relevance
of a source. Use synthetic documents for the demo.

## Reproduce the demo

Requirements: Python 3.12, Docker with Compose, and available ports 55433 and 11435.
Run from the repository root. All exposed services bind to loopback.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -c constraints.txt
# Optional: copy .env.example to .env only if you do not already have a .env.
export DATABASE_URL=postgresql://healthassist:demo-only-password@127.0.0.1:55433/healthassist
docker compose up -d --wait db
python -m scripts.init_db
```

Run the deterministic demo without downloading any models:

```bash
python -m scripts.demo --mock-ollama
```

It starts a temporary API and a clearly labeled mock Ollama HTTP fixture, generates
a synthetic PDF in memory, uploads it, retrieves it from the real database, and
checks the cited answer. It exits nonzero on failure and stops only its own API
and mock server. This is a plumbing check, **not actual LLM inference**. Mock
vectors use a distinct model name, so they never mix with real Ollama vectors.
The demo can be repeated; each upload gets a new document and unique chunk IDs.

For actual local inference (free, but model downloads require Internet access):

```bash
docker compose --profile llm up -d ollama
# Wait until this succeeds, then download both models:
curl --fail http://127.0.0.1:11435/api/version
docker compose exec -e OLLAMA_HOST=http://127.0.0.1:11434 ollama ollama pull nomic-embed-text
docker compose exec -e OLLAMA_HOST=http://127.0.0.1:11434 ollama ollama pull qwen2.5:0.5b
export OLLAMA_BASE_URL=http://127.0.0.1:11435
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
export OLLAMA_CHAT_MODEL=qwen2.5:0.5b
python -m scripts.demo
```

Model downloads use `registry.ollama.ai` and may redirect to model blob hosts;
restricted environments must allow the actual download destinations. The demo
fails visibly if a model cannot load or fails the quotation contract. It never
silently falls back to mocked output. A larger downloaded local chat model can
be configured if the small model cannot select valid excerpts. Changing embedding
models requires reingestion; the schema currently requires 768 dimensions.

Dependencies are constrained by `constraints.txt`; service images are pinned by
digest. Model tags can change: record `ollama list` model IDs when evaluating real
models. Database and model volumes persist across `docker compose --profile llm down`.
Use `down --volumes` only when you intentionally want to delete this demo's data.

## Use the API

With the database initialized and local models installed:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal:

```bash
curl --fail -X POST 'http://127.0.0.1:8000/documents?file_name=guide.pdf&version=v1' \
  -H 'Content-Type: application/pdf' --data-binary @/path/to/guide.pdf
curl --fail http://127.0.0.1:8000/questions \
  -H 'Content-Type: application/json' \
  -d '{"question":"When does the fictional Cedar clinic open?"}'
```

The upload body is raw PDF bytes, not multipart or a filesystem/remote URL.
Optional query parameters are `file_name`, `title`, and `version`. Defaults limit
uploads to 10 MiB, 100 pages, and 200 chunks; query strings and filenames are bounded.
Encrypted, invalid, empty, and image-only PDFs are rejected. Uploaded files are
removed after extraction. Extracted text and embeddings remain in PostgreSQL.
Repeated filenames create separate documents; versions are labels, and older
versions are not automatically deactivated. Retrieval respects `documents.is_active`.

Responses use `answered`, `no_relevant_context`, or `insufficient_evidence`.
Each answer quotation has a server-generated `[source_id]` reference and a matching
entry in `citations`. HTTP 422 denotes invalid input, 413 an oversized upload,
415 an unsupported media type, 502 a provider/grounding failure, and 503 a database
failure. Technical errors never masquerade as missing evidence. `/health` checks
PostgreSQL, not model availability; OpenAPI is available at `/docs`.

Configuration names and demo defaults are in `.env.example`. Retrieval defaults to
five chunks and a 0.5 cosine similarity threshold; this is **not clinically
calibrated**. `RETRIEVAL_MAX_QUERY_LENGTH` can reduce the query limit below the
API's hard cap of 2,000 characters. Ollama URLs must be local HTTP addresses and
cloud model tags/Bedrock are rejected by this API.

## Tests and CI

```bash
python -m unittest discover -s tests -v
# Add real PostgreSQL/pgvector integration (uses isolated, rolled-back schemas):
HEALTHASSIST_TEST_DATABASE_URL="$DATABASE_URL" python -m unittest discover -s tests -v
python -m pip check
```

Without the test database variable, database tests are explicitly skipped. With a
configured but unavailable database, tests fail. Unit tests mock PostgreSQL and
Ollama; database/API integration tests use real PostgreSQL and mocked Ollama HTTP
responses. CI runs both suites plus the deterministic HTTP demo on pull requests.
No paid APIs, real medical documents, or model downloads are needed in CI.
Do not run root `test_embedding.py` during discovery; it is a live provider demo.

## Limits

This is an unauthenticated, single-workspace local MVP. Do not expose it on a public
interface or treat uploads as approved clinical evidence. Production needs user
identity, authorization and dataset isolation, source review, upload/process limits,
operational monitoring, and clinical evaluation. PDF parsing runs in a worker thread
but is not a sandbox for hostile files. Extractive grounding prevents fabricated
quotations, not irrelevant excerpts, misleading sources, or prompt injection in
source content. There is no OCR, conversational memory, free-form clinical synthesis,
automatic version approval, or document deletion endpoint.

See `docs/mvp-validation.md` for actual validation results and outstanding blockers.
