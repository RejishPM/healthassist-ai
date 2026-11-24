# RAG MVP validation

Validated on 2026-10-15 with Python 3.12.14, PostgreSQL 16/pgvector, and the
existing `fix/rag-foundation-20261014` checkout. `main` was not modified.

## Implemented workflow

- FastAPI raw-PDF ingestion at `POST /documents`, validated streaming upload
  limits, temporary-file cleanup, extraction/chunking, local Ollama embeddings,
  and existing transactional PostgreSQL persistence.
- Standalone question answering at `POST /questions`: query validation, configured
  cosine threshold/top-k, compatible embedding model filtering, active documents,
  bounded context, local Ollama structured evidence selection, server-checked
  quotations, and source/page/version citations.
- Explicit abstention for absent/insufficient evidence; invalid provider output,
  unknown citations, or fabricated quotations are not returned as answers.
- Database pool startup/readiness/shutdown, `/health`, and OpenAPI upload schema.
- Generated synthetic PDF demo, local-only Compose services, separate mock/real
  provider modes, constrained dependency versions, and extended pull-request CI.

## Actual results

A fresh virtual environment was created and installed with:

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
```

| Check | Actual outcome |
| --- | --- |
| Full suite with `HEALTHASSIST_TEST_DATABASE_URL` | 49 tests executed, 49 passed, 0 skipped |
| Same suite without a test database URL | 45 passed, 4 database integration tests explicitly skipped |
| `python -m pip check` in clean environment | No broken requirements |
| `docker compose config --quiet` | Passed |
| Schema initialization with `python -m scripts.init_db`, repeated | Passed |
| `python -m scripts.demo --mock-ollama` with real HTTP/PDF/PostgreSQL | Passed, including repeated uploads/runs |
| `git diff --check` | Passed |
| Actual local Ollama model download | Blocked: registry proxy returned `403 Forbidden` |
| `python -m scripts.demo` with the running but model-empty Ollama | Exited 1; ingestion correctly returned HTTP 502 |
| Hosted GitHub Actions / PR metadata API | Unverified: API requests returned `Forbidden` |

The four integration tests cover fresh schemas and active filtering, rollback on
chunk collision, idempotent legacy-schema migration, and PDF HTTP ingestion through
real vector retrieval to a cited HTTP answer with a mocked Ollama HTTP provider.
Unit coverage includes upload validation, provider errors, temporary-file cleanup,
encrypted/empty PDFs, page/chunk limits, forged quotes/citations, bounded source
validation, and malformed/nonfinite/zero/wrong-dimension embeddings.

The mock demo generated a single-page synthetic Cedar clinic PDF, ingested it,
retrieved it from pgvector, and returned its exact text with `[1]`, the correct
filename, page 1, version, and persistent document/chunk identifiers. This validates
application plumbing and provenance; it does not validate real semantic embeddings
or real language-model behavior. No paid API was called.

## Review: security, correctness, compatibility

SQL remains parameterized and persistence remains transactional. Source filenames
are validated basenames, not paths; no remote-file ingestion is supported. Temporary
PDFs are deleted even on parser failure. Model/provider errors use generic HTTP
responses rather than exposing provider bodies or database connection details.
The API restricts Ollama to local service addresses, rejects cloud model tags and
Bedrock, and has no tools for the model to execute. Question and document content
are separated from system instructions. Grounding checks reject unsupported source
IDs and non-verbatim text; every returned answer passage has a citation.

Two correctness changes preserve existing signatures: prose chunks now stop at
page/source boundaries so page citations are accurate, and ingestion/retrieval
reject vectors incompatible with the fixed 768-dimensional schema before database
access. API uploads namespace logical IDs while keeping original source metadata;
existing CLI persistence behavior and schema remain compatible. Existing documents
do not require an identifier migration. Reingesting documents after the chunking
change is recommended when page-accurate citations are required.

This is a local, unauthenticated, single-workspace MVP. Uploads become active
immediately; there is no source approval, tenancy, automatic version retirement,
clinical evaluation, or public-service protection. Do not expose it publicly.
The default retrieval threshold is a demo choice, not a calibrated clinical value.
The LLM returns extractive evidence rather than unconstrained clinical synthesis.
A verified quote can still be irrelevant, misleading, malicious, or incomplete.
The PDF parser is not sandboxed; byte/page/chunk limits do not defeat every
malicious compressed PDF. Dependencies and images are pinned for repeatability,
not a claim of a completed supply-chain vulnerability audit.

## Remaining validation and release decision

Real-model quality and the successful real-Ollama demo remain unverified because
`registry.ollama.ai` is blocked and no models are cached. The environment draft
adds that domain while preserving `api.github.com`; model blob redirects may need
additional specific domains once observable. Review/save those network settings
and publish the environment, then pull both local models and run the real demo.
Do not disable TLS/checksum validation or substitute mocked output for this check.
Model tags are mutable; record their actual IDs when performing that validation.

CI now runs constrained installs, unit tests, PostgreSQL integration, and the
mock-provider HTTP demo. These are configured checks, not evidence of a hosted
successful run. Verify the latest hosted jobs and complete real-model validation
before approving the MVP for merge. No merge or deployment was performed.


## Real Ollama validation retry — 2026-10-15

This retry started from `2475b1c` on `fix/rag-foundation-20261014`.
Only documentation was changed. The existing local Compose services were started;
no cloud resources were deployed, and no paid APIs or mock fallback were used.

### Connectivity and exact download failure

`GET http://127.0.0.1:11435/api/version` succeeded:

```json
{"version":"0.6.8"}
```

`GET /api/tags` returned `{"models":[]}`. `ollama list` printed only its header:

```text
NAME    ID    SIZE    MODIFIED
```

Neither requested model is installed, so there are **no model IDs to report**.
Both real download commands were executed with the local CLI host explicitly set:

```bash
docker compose exec -T -e OLLAMA_HOST=http://127.0.0.1:11434 ollama ollama pull nomic-embed-text
docker compose exec -T -e OLLAMA_HOST=http://127.0.0.1:11434 ollama ollama pull qwen2.5:0.5b
```

Both exited 1 with these observed errors:

```text
Error: pull model manifest: Get "https://registry.ollama.ai/v2/library/nomic-embed-text/manifests/latest": Forbidden
Error: pull model manifest: Get "https://registry.ollama.ai/v2/library/qwen2.5/manifests/0.5b": Forbidden
```

A direct HTTPS request confirmed the precise failure: the HTTPS proxy rejected
CONNECT to **`registry.ollama.ai:443`**, before establishing a TLS connection to
the registry. `curl` exited 56 and reported:

```text
HTTP/1.1 403 Forbidden
content-length: 16
content-type: text/plain
server: envoy
connection: close

curl: (56) CONNECT tunnel failed, response 403
```

The curl target HTTP status was `000` because the tunnel was never established;
the **proxy CONNECT response** was 403. No model blob redirect was reached, so
there is no evidence of any additional blocked blob hostname. TLS verification
and model integrity checks were left enabled.

### Real demo and independent checks

Schema initialization succeeded, followed by the actual, unmocked command:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11435 \
OLLAMA_EMBEDDING_MODEL=nomic-embed-text \
OLLAMA_CHAT_MODEL=qwen2.5:0.5b \
python -m scripts.demo
```

`DATABASE_URL` pointed at the disposable local Compose demo database. The command
printed `LOCAL OLLAMA DEMO: requires downloaded embedding and chat models`, then
exited **1** because `/documents` returned **HTTP 502**. Direct provider probes
confirmed why:

```text
POST /api/embeddings -> HTTP 404
{"error":"model \"nomic-embed-text\" not found, try pulling it first"}
POST /api/chat -> HTTP 404
{"error":"model \"qwen2.5:0.5b\" not found, try pulling it first"}
```

The real demo did not generate embeddings or reach question answering. Actual
semantic retrieval, source quotations/citations, and both no-context and
insufficient-evidence abstention therefore remain **unverified with real models**.
The error was not converted into an abstention or silently replaced by mock output.

The database contained only the two retained `demo-mock-768` vectors (768 dimensions),
not real `nomic-embed-text` embeddings. These existing fixture vectors are excluded
by the real-model retrieval filter and are not counted as successful validation.

The independent test suite was rerun with a real PostgreSQL/pgvector connection:

```text
Ran 49 tests in 0.470s
OK
```

All 49 passed, zero skipped, including four database integration tests. These
tests mock Ollama, as documented; they do not resolve the real-model blocker.

### PR evidence and readiness decision

GitHub PR API access also failed: `gh pr view 1` returned `Forbidden`, and a direct
HTTPS probe to `api.github.com:443` received the same Envoy CONNECT 403. Hosted CI
and PR metadata remain unverified. The documentation is committed to the PR's
existing feature branch; posting a separate PR comment requires working API access.

The environment configuration already lists both `registry.ollama.ai` and
`api.github.com`, but the current runtime proxy still denies them. Restore runtime
HTTPS CONNECT access to those hosts rather than repeatedly adding duplicate rules.
Then retry both model downloads, inspect any newly observable blob-host denial,
record the installed model IDs, and run the real demo plus relevant/irrelevant
question and abstention checks. No assertions about real model quality can be made
from the successful mocked test suite.

**Recommendation: not ready for human merge approval.** The code can be reviewed,
but real-model grounding and abstention validation and the latest hosted CI results
are still outstanding. `main` is unchanged; no merge or deployment was performed.
