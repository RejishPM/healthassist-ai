# HealthAssist development

Architecture and contracts: read `docs/02-architecture.md`,
`docs/03-database-design.md`, `docs/04-embedding-design.md`, and
`docs/05-retrieval-design.md`. PDF extraction/chunking lives in `app/ingestion`,
providers in `app/embedding`, PostgreSQL/pgvector persistence in `app/db`,
and retrieval models in `app/query`.

Use the existing checkout and requested feature branch; cloud tasks are isolated,
so do not create a worktree unless requested. Keep changes focused, preserve
parameterized SQL, transactions, metadata, and existing public interfaces.
Never commit credentials or clinical documents, call paid APIs, or deploy resources
without explicit authorization. Mock PostgreSQL and Ollama in unit tests.

Python 3.12: `python -m pip install -r requirements.txt`.
Run offline tests with `python -m unittest discover -s tests -v`.
`test_embedding.py` at the repository root is a live provider demo, not an offline
unit test; do not run it as part of discovery.
For database integration, set `HEALTHASSIST_TEST_DATABASE_URL` to a disposable
PostgreSQL database with pgvector installed, then run
`python -m unittest tests.test_database_integration -v`. Tests use isolated schemas
and roll back their changes; a configured but unavailable database must fail.

PRs should describe behavior, regression coverage, actual test results, skipped
integration checks, and security/backward compatibility risks. Require successful
unit and database CI jobs before merging. Do not modify main or merge without
human approval.
