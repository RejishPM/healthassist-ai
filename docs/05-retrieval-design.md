# HealthAssist Retrieval Design

## 1. Purpose

The retrieval layer is responsible for finding the most relevant approved clinical evidence for a user question.

It sits between the query-understanding layer and the LLM generation layer.

High-level flow:

```text
User Question
    ↓
Conversation Context
    ↓
Query Rewriter
    ↓
Standalone Retrieval Query
    ↓
Query Embedding
    ↓
Semantic Retrieval
    ↓
Similarity Threshold
    ↓
Top-K Relevant Chunks
    ↓
Retrieved Evidence
    ↓
Prompt / Context Builder
    ↓
LLM / LangGraph
```

The retrieval layer does not generate the final medical answer. Its job is to identify, rank, filter, and return grounded evidence with provenance and safety metadata.

---

## 2. Current Retrieval Strategy

The first implementation will use semantic vector search with PostgreSQL and pgvector.

Initial implementation:

```text
Semantic Retrieval
    ↓
Exact pgvector search
    ↓
Cosine distance
    ↓
Similarity threshold
    ↓
Top-K qualifying chunks
```

Hybrid retrieval, reranking, production-scale ANN indexing, and richer metadata filtering will be added incrementally after the base semantic retrieval flow is validated.

---

## 3. Query Understanding and Rewriting

Users often ask conversational follow-up questions that are not suitable for retrieval by themselves.

Example conversation:

```text
User:
What are the symptoms of diabetic ketoacidosis?

User:
What about treatment?
```

Embedding only:

```text
What about treatment?
```

would lose the clinical context.

The query-rewriting layer should convert the follow-up into a standalone retrieval query:

```text
What is the recommended treatment for diabetic ketoacidosis?
```

### Responsibilities of Query Rewriting

The query rewriter may:

* Resolve pronouns
* Resolve conversational references
* Expand supported abbreviations when context makes the meaning clear
* Normalize wording
* Convert follow-up questions into standalone retrieval queries
* Preserve the user's original intent

The query rewriter must not:

* Invent medical facts
* Introduce unsupported diagnoses
* Add assumptions that are not present in the conversation
* Change the user's clinical intent

---

## 4. Conversation Context

Conversation history may be useful in two separate places.

### 4.1 Retrieval Context

Relevant prior turns may be used to rewrite the current query.

Example:

```text
Previous question:
What are the symptoms of DKA?

Current question:
What about children?
```

Possible rewritten retrieval query:

```text
What are the symptoms of diabetic ketoacidosis in children?
```

The entire conversation should not automatically be embedded.

Instead:

```text
Conversation History
    ↓
Relevant Context Selection
    ↓
Query Rewriter
    ↓
Standalone Retrieval Query
```

This reduces noise and improves retrieval precision.

### 4.2 LLM Context

Relevant conversation history may also be provided later to the LLM to preserve conversational continuity.

Conceptually:

```text
Prompt Context
    ├── Current user question
    ├── Relevant conversation history
    ├── Retrieved clinical evidence
    └── Safety instructions
```

Conversation history provides context.

Retrieved approved clinical documents provide evidence.

Conversation history must never be treated as authoritative clinical evidence.

---

## 5. Query Input Validation

The retrieval layer must validate input before generating an embedding.

Initial validation should include:

* Reject empty input
* Reject whitespace-only input
* Enforce a configurable maximum query length
* Normalize input where appropriate
* Detect invalid or unsupported input patterns where necessary

Example:

```text
"       "
```

must not trigger an embedding request.

Extremely long queries should either be rejected or handled according to an explicit policy.

Language support should be validated against the selected embedding model before multilingual retrieval is advertised.

---

## 6. Query Embedding

The rewritten standalone query is converted into an embedding.

Current development configuration:

```text
Provider: Ollama
Model: nomic-embed-text
Embedding dimension: 768
```

Conceptually:

```text
Standalone Query
    ↓
Embedder abstraction
    ↓
OllamaEmbedder / BedrockEmbedder
    ↓
Query Vector
```

The retrieval layer should depend on the embedding abstraction rather than directly on a concrete provider.

Conceptually:

```text
SemanticRetriever
       ↓
    Embedder
      /   \
 Ollama   Bedrock
```

---

## 7. Embedding Model Compatibility

Query vectors and stored document vectors must be generated using compatible embedding models.

The retrieval query must therefore filter stored vectors using:

```text
provider
model
```

Example:

```sql
WHERE provider = 'ollama'
  AND model = 'nomic-embed-text'
```

Changing the configured embedding model without rebuilding stored embeddings can produce invalid semantic comparisons even if both models use the same vector dimension.

Therefore the system should never compare vectors purely because their dimensions match.

Future metadata may include:

```text
embedding_version
embedding_created_at
embedding_configuration
```

When an embedding model changes, affected documents should be re-embedded through a controlled migration or re-indexing process.

---

## 8. Vector Search

Stored document embeddings currently reside in:

```text
ollama_embeddings.embedding
```

The relationship is:

```text
ollama_embeddings.document_chunk_id
    ↓
document_chunks.id
```

The retrieval query will join:

* ollama_embeddings
* document_chunks
* documents

This allows the retriever to return:

* matching vector score
* chunk text
* page number
* source document
* document title
* version information
* safety-critical metadata

---

## 9. Similarity Metric

The initial implementation will use cosine distance with pgvector.

pgvector cosine search operator:

```text
<=>
```

returns cosine distance.

Smaller distance means a better match.

HealthAssist will expose similarity using:

```text
similarity = 1 - cosine_distance
```

Therefore:

```text
distance ↓ = better match

similarity ↑ = better match
```

Example:

```text
cosine distance = 0.12

similarity = 1 - 0.12
           = 0.88
```

All retrieval code, logs, tests, documentation, and configuration should use the same convention.

---

## 10. Top-K Retrieval

The retriever will rank candidate chunks by semantic similarity.

Initial configuration:

```text
RETRIEVAL_TOP_K=5
```

The value should be configuration-driven and not hard-coded.

Conceptually:

```text
Query Vector
    ↓
Compare against compatible stored vectors
    ↓
Rank by distance
    ↓
Apply similarity threshold
    ↓
Return at most Top-K results
```

---

## 11. Similarity Threshold

Top-K retrieval alone is unsafe because it will always return the nearest chunks even when none are actually relevant.

HealthAssist must therefore apply a minimum similarity threshold.

Planned configuration:

```text
RETRIEVAL_SIMILARITY_THRESHOLD=<validated value>
```

The threshold must eventually be selected using evaluation data rather than arbitrary intuition.

Conceptually:

```text
Candidate Chunk
    ↓
Similarity >= threshold?
    ├── Yes → retain
    └── No  → discard
```

---

## 12. No-Relevant-Context Behavior

The retrieval layer must explicitly distinguish between:

```text
SUCCESS
NO_RELEVANT_CONTEXT
ERROR
```

### SUCCESS

Relevant chunks were found.

### NO_RELEVANT_CONTEXT

The retrieval process completed correctly, but no chunks passed the configured relevance threshold.

Example conceptual response:

```text
results = []
has_relevant_context = false
status = NO_RELEVANT_CONTEXT
```

The downstream generation layer should then respond with behavior such as:

```text
I do not have sufficient information in the approved knowledge base to answer this question.
```

The system should not silently pass irrelevant context to the LLM.

### ERROR

A technical failure occurred, such as:

* embedding service unavailable
* database connection failure
* malformed vector
* query execution failure

An error must not be interpreted as "no relevant context."

---

## 13. Expected Retrieval Result

Each returned chunk should contain at least:

```text
document_chunk_id
chunk_id
text
page
source
document_title
document_version
safety_critical
distance
similarity
provider
model
```

Retrieval-level metadata should include:

```text
status
has_relevant_context
retrieved_count
top_k
similarity_threshold
```

---

## 14. Source Traceability

Every returned chunk must be traceable back to the approved source.

Relationship:

```text
Document
    ↓
Page
    ↓
Chunk
```

Example:

```text
Source:
Living_with_Diabetes_UK_Report.PDF

Page:
12

Chunk:
Living_with_Diabetes_UK_Report.PDF_p12_c3
```

This information will later be used to generate citations.

Traceability is a core requirement of HealthAssist.

---

## 15. Document Version Safety

Clinical guidance changes over time.

If multiple versions of the same guideline are stored, unrestricted retrieval could return outdated information.

Example:

```text
Diabetes Guideline v1
Diabetes Guideline v2
Diabetes Guideline v3
```

A semantically relevant chunk from v1 may still rank highly even though v3 is the approved current version.

Future document lifecycle metadata should include fields such as:

```text
version
status
effective_date
superseded_date
```

Clinical retrieval should eventually restrict searches to:

```text
status = active
```

or the latest approved version.

Until document lifecycle filtering is implemented, retrieval across multiple versions of the same clinical document is a known limitation.

---

## 16. Safety-Critical Metadata

The `document_chunks` table contains:

```text
safety_critical
```

This flag must be returned by retrieval.

Example safety-critical content may include:

* emergency escalation guidance
* severe hypoglycemia guidance
* diabetic ketoacidosis warnings
* contraindications
* high-risk medication instructions

Conceptually:

```text
Retrieved Evidence
    ↓
safety_critical = true?
    ↓
LangGraph workflow
    ↓
Additional safety / compliance checks
    ↓
Human-in-the-loop when required
```

The retrieval layer does not make the final safety decision.

It must preserve the safety metadata so downstream components can make that decision.

---

## 17. Embedding Provider Resilience

The embedding provider is an external dependency from the retriever's perspective.

Even a local Ollama service may be:

* unavailable
* restarting
* slow
* overloaded

Embedding calls therefore need explicit failure behavior.

Initial requirements:

* request timeout
* clear exception handling
* structured error logging
* bounded retries where appropriate

Retries should never be infinite.

A failed embedding request must result in an error state rather than a no-context state.

Future production deployments may use provider-level resilience and routing through the LLM/embedding gateway architecture.

---

## 18. Async Retrieval

The semantic retriever should be asynchronous.

HealthAssist is expected to integrate with:

* FastAPI
* PostgreSQL async connection pools
* LangGraph
* remote embedding providers

The interface should support a pattern such as:

```python
result = await retriever.retrieve(question)
```

Blocking provider implementations may temporarily be executed using a worker thread where necessary.

The retrieval layer should avoid blocking the application's async request path.

---

## 19. Retrieval Observability

Retrieval requests should generate structured diagnostic information.

Useful fields include:

```text
request_id
timestamp
embedding_provider
embedding_model
top_k
similarity_threshold
retrieved_document_chunk_ids
logical_chunk_ids
similarity_scores
source_documents
pages
safety_critical_flags
retrieval_status
latency
```

Raw healthcare or user content should not automatically be logged in production.

Logging must eventually comply with:

* PII/PHI handling requirements
* retention policies
* audit requirements
* access controls
* healthcare privacy regulations

The observability strategy should allow the team to answer:

```text
Why was this evidence retrieved?

What evidence supported this answer?

Which similarity scores were returned?

Did safety-critical content participate in the answer?
```

---

## 20. Retrieval Evaluation

Retrieval quality should be measured before introducing additional complexity.

The project should maintain a small evaluation dataset containing:

```text
Question
Expected document
Expected page or chunk
```

Example:

```text
Question:
What are the symptoms of diabetic ketoacidosis?

Expected Document:
Living_with_Diabetes_UK_Report.PDF

Expected:
Known DKA section/page/chunk
```

This dataset becomes a regression suite.

It should be rerun after changes such as:

```text
semantic retrieval
→ hybrid retrieval
→ reranking
→ embedding-model change
→ chunking-strategy change
```

Initial metrics may include:

* Hit Rate @ K
* Recall @ K
* Mean Reciprocal Rank

Precision metrics can be added once relevance labels are sufficiently complete.

---

## 21. Vector Index Strategy

The current dataset contains only a small number of vectors.

At the current scale, exact vector search is preferred.

Benefits:

* simple
* deterministic
* exact nearest-neighbor results
* suitable for retrieval-quality validation

Approximate nearest-neighbor indexing should be introduced only when scale creates a measurable latency requirement.

### Future HNSW Option

Important parameters include:

```text
m
ef_construction
ef_search
```

HNSW generally provides strong query performance and recall.

### Future IVFFlat Option

Important parameters include:

```text
lists
probes
```

IVFFlat provides configurable speed/recall trade-offs.

Index selection should be based on measured:

* corpus size
* retrieval latency
* recall
* memory
* ingestion cost

rather than selected prematurely.

---

## 22. Configuration

Retrieval configuration should be environment-driven from the start.

Example:

```text
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text

RETRIEVAL_TOP_K=5
RETRIEVAL_SIMILARITY_THRESHOLD=<validated value>
RETRIEVAL_MAX_QUERY_LENGTH=<configured value>
```

This allows retrieval behavior to change across:

* local development
* testing
* staging
* production

without modifying application code.

---

## 23. Responsibilities of the Retrieval Layer

The retrieval layer should:

* Accept a validated standalone retrieval query
* Generate a compatible query embedding
* Search stored embeddings
* Rank candidates
* Convert distance to similarity consistently
* Apply a similarity threshold
* Return at most Top-K qualifying chunks
* Preserve document provenance
* Preserve document version information
* Preserve safety-critical metadata
* Return explicit no-context behavior
* Produce useful diagnostic metadata

The retrieval layer should not:

* Generate the final clinical answer
* Treat conversation history as clinical evidence
* Make autonomous clinical decisions
* Perform human approval
* Modify source documents
* Invent context when retrieval fails

---

## 24. Planned Project Structure

Initial:

```text
app/
├── query/
│   ├── __init__.py
│   ├── conversation_context.py
│   └── query_rewriter.py
│
└── retrieval/
    ├── __init__.py
    └── semantic_retriever.py
```

Future:

```text
app/
└── retrieval/
    ├── semantic_retriever.py
    ├── keyword_retriever.py
    ├── hybrid_retriever.py
    └── reranker.py
```

These additional components should be introduced only when the simpler implementation has been validated.

---

## 25. Full Conversational Retrieval Flow

```text
User
 ↓
Current Question
 ↓
Conversation Context Manager
 ↓
Relevant History Selection
 ↓
Query Rewriter
 ↓
Standalone Retrieval Query
 ↓
Input Validation
 ↓
Embedder
 ↓
Query Vector
 ↓
Compatible Provider/Model Filter
 ↓
pgvector Cosine Search
 ↓
Distance
 ↓
Similarity = 1 - Distance
 ↓
Similarity Threshold
 ├── No Match
 │      ↓
 │   NO_RELEVANT_CONTEXT
 │
 └── Match
        ↓
      Top-K
        ↓
      Evidence + Provenance
        ↓
      Safety Metadata
        ↓
      Prompt / Context Builder
        ├── Current Question
        ├── Relevant Conversation Context
        └── Retrieved Approved Evidence
        ↓
      LangGraph / LLM
        ↓
      Guardrails / HITL
        ↓
      Grounded Answer + Citations
```

---

## 26. Initial Semantic Retriever Contract

The initial retriever should conceptually expose:

```python
result = await retriever.retrieve(query)
```

The implementation must:

1. Validate the query.
2. Generate a query embedding using the configured provider.
3. Search only embeddings generated by the compatible provider and model.
4. Calculate cosine distance.
5. Convert distance to the documented similarity convention.
6. Apply the configured similarity threshold.
7. Return at most Top-K qualifying chunks.
8. Preserve source, page, document version, and safety metadata.
9. Return an explicit no-context state when nothing qualifies.
10. Return an error state for technical failures.
11. Produce structured retrieval diagnostics.

---

## 27. Known Limitations of the Initial Implementation

The initial semantic retrieval implementation will not yet include:

* Hybrid semantic + keyword retrieval
* Reranking
* Automatic active-document-version filtering
* Complete multilingual support
* Production-scale ANN indexing
* Full conversational memory management
* Production-grade query rewriting
* Complete retrieval evaluation framework

These limitations are intentional.

The design will evolve incrementally after the base semantic retrieval flow is implemented, tested, and understood.

---

## 28. Current Status

The following stages are complete and validated:

```text
PDF ingestion
→ text extraction
→ chunking
→ embedding generation
→ PostgreSQL persistence
→ pgvector storage
```

Current database state:

```text
Documents: 1
Document Chunks: 25
Embeddings: 25
```

The relationship has been validated:

```text
documents.id
    ↓
document_chunks.document_id

document_chunks.id
    ↓
ollama_embeddings.document_chunk_id
```

The next implementation stage is:

```text
Query Understanding
        ↓
Semantic Retrieval
```

The first technical milestone is to successfully retrieve relevant chunks from the current 25 stored embeddings for a real healthcare question while correctly handling low-confidence or no-match queries.
