# PR #1 validation — 2026-10-14

Scope: `fix/rag-foundation-20261014`, reviewed base fixes through `dc8cf42`.

## Existing fixes

- `08233f6`: passes returned internal chunk IDs to embedding persistence; resolves
  the undefined variable without changing the public interface or logical IDs.
- `e0ee317`: adds `documents.is_active BOOLEAN NOT NULL DEFAULT TRUE` to both fresh
  schemas and legacy databases. Existing documents remain retrievable. Apply
  `app/db/schema.sql` to existing databases before using retrieval.
- `dc8cf42`: explicitly declares `requests`, already used by Ollama. No new paid
  provider calls or deployment steps are introduced.

The obsolete test keyword is now `document_chunk_ids`. Regression coverage checks
embedding/chunk alignment, metadata preservation, failure propagation, transaction
exit, retrieval validation, SQL parameters/filter/order, and result mapping.
Ollama HTTP responses, HTTP errors, and timeouts are mocked. Database integration
uses deterministic 768-dimensional embeddings and real PostgreSQL/pgvector;
it covers fresh schemas, legacy upgrades, rerunning schema SQL, NOT NULL/default
contracts, active filtering, and atomic rollback after duplicate chunks.

## Actual local results

Python 3.12.14, dependencies installed from `requirements.txt` in an isolated venv.

- `python -m unittest discover -s tests -v`: 20 discovered, 17 passed,
  3 explicitly skipped when `HEALTHASSIST_TEST_DATABASE_URL` was unset.
- With that variable set to a disposable local PostgreSQL 16/pgvector container:
  the same command ran 20 tests; all 20 passed, with zero skipped.
- `python -m pip check`: no broken requirements.
- `git diff --check`: passed.

Database image digest:
`sha256:7b822b0aac60967beb1ea5e576b8602c94c300a157d187f385ae3e0da199b90a`.
Tests isolate their schema in a transaction and roll back on close. Only use a
throwaway test database; integration setup needs extension/schema privileges.
No actual Ollama model or Bedrock API was called. Root `test_embedding.py` is a
live provider demo and is deliberately excluded from offline discovery.

## CI and security review

`.github/workflows/tests.yml` runs on pull requests and pushes to main/the feature
branch. Separate unit and database jobs use Python 3.12. The database job requires
the integration tests to execute, rather than silently accepting skipped tests.
Actions and the database image are pinned to verified revisions/digest. Workflow
permissions are read-only, and the database password is a disposable test value.
No repository or production secrets are needed; no privileged PR trigger is used.

Application SQL remains parameterized; writes remain transactional. Metadata and
embedding request exceptions propagate without logging clinical text. No runtime
application code or public signatures were changed during validation.

GitHub API requests for PR metadata and Actions runs returned `Forbidden` in this
environment. Consequently, local test success is verified, but hosted CI status
and PR description updates are not. The workflow file is not evidence of a green
GitHub Actions run. Verify both jobs on the latest PR commit before merging.

## Outstanding limitations and merge recommendation

- Python dependencies remain unpinned, so future installs can differ. A dependency
  lock is a separate follow-up rather than part of these focused fixes.
- `ADD COLUMN IF NOT EXISTS` handles the documented legacy schema without the
  column; it does not repair a manually created nullable/wrong-default column.
- Chunk IDs remain globally unique: reingesting the same logical IDs fails and
  rolls back. Document version/reingestion policy needs a later design decision.
- Real Ollama availability, model quality, PDF-to-database ingestion, and clinical
  answer quality are outside this focused persistence/retrieval validation.

The reviewed fixes are suitable to merge once both hosted CI jobs pass on the
latest commit and the target database schema is upgraded. Until that hosted
result is verified, hold merge approval. Do not merge automatically.
