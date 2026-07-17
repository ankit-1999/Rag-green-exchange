# GreenGrid FastAPI LLD

This Low-Level Design describes only the FastAPI application modules and runtime interactions.

## 1. Application Layers

```mermaid
flowchart LR
  A[Routes Layer] --> B[Service Orchestration Layer]
  B --> C[Domain Services]
  B --> D[External Integrations]
  C --> E[Schemas]

  A1[document_routes.py] --> B
  A2[query_routes.py] --> B

  B1[document_service.py] --> C
  B2[rag_service.py] --> C

  C1[analytics_service.py]
  C2[prompt_service.py]
  C3[tool_registry.py]
  C4[chunker_service.py]

  D1[bedrock_service.py]
  D2[opensearch_service.py]
  D3[marketplace_api_service.py]
  D4[S3 via boto3]

  E1[document_schema.py]
  E2[query_schema.py]
```

---

## 2. Module Responsibilities

### 2.1 app/main.py
- Creates FastAPI app.
- Registers CORS and security header middleware.
- Includes routers.
- Runs startup index readiness check.

### 2.2 app/routes/query_routes.py
- Defines POST /query.
- Validates request with QueryRequest.
- Delegates to rag_service.answer_question().
- Converts service exceptions to HTTP errors.

### 2.3 app/routes/document_routes.py
- Defines POST /documents/upload, GET /documents, POST /documents/clear-index, GET /documents/{id}.
- Delegates ingestion and metadata operations to document_service.
- Converts service exceptions to HTTP errors.

### 2.4 app/services/rag_service.py
- Central query orchestrator.
- Runs planning, tool execution, analytics, retrieval, prompt building, rendering/generation.
- Returns QueryResponse.

### 2.5 app/services/document_service.py
- Central insert orchestrator.
- Fetches file from S3, extracts text, chunks text, embeds chunks, indexes chunks.
- Stores and serves in-memory document metadata.

### 2.6 Supporting services
- bedrock_service.py: embedding generation, planner generation, final answer generation.
- opensearch_service.py: index lifecycle, chunk indexing, vector retrieval, index clearing.
- analytics_service.py: deterministic analytics, prediction, recommendation outputs.
- prompt_service.py: grounded prompt assembly with strict output constraints.
- chunker_service.py: token-aware chunking.
- marketplace_api_service.py: read-only marketplace API execution.
- tool_registry.py: allowed tool definitions for planner and execution guardrails.

---

## 3. Key Data Contracts

### 3.1 Query contracts
Request: QueryRequest
- question: string (required)
- top_k: integer 1..20 (optional)

Response: QueryResponse
- answer: HTML string
- source_count: integer
- sources: list of QuerySource
- answer_mode: retrieval_only | retrieval_plus_api | api_only | insufficient_data
- api_facts_used: boolean
- api_summary: QueryApiSummary (optional)

### 3.2 Document insert contracts
Request: DocumentUploadRequest
- document_name
- document_type
- s3_uri

Response: DocumentUploadResponse
- document_id
- document_name
- status
- chunk_count
- message

---

## 4. LLD for Query Answer Generation

```mermaid
flowchart TD
  Q1[query_routes.query_documents] --> Q2[rag_service.answer_question]
  Q2 --> Q3[bedrock_service.plan_api_calls]
  Q3 --> Q4[tool call validation + normalization]
  Q4 --> Q5[marketplace_api_service.execute_tool_call]
  Q5 --> Q6[analytics_service.analyze_plan]
  Q6 --> Q7{Retrieval needed?}
  Q7 -- Yes --> Q8[embed question + OpenSearch similarity search]
  Q7 -- No --> Q9[skip retrieval]
  Q8 --> Q10[prompt_service.build_rag_prompt]
  Q9 --> Q10
  Q10 --> Q11{Known deterministic intent?}
  Q11 -- Yes --> Q12[rag_service deterministic renderer]
  Q11 -- No --> Q13[bedrock_service.generate_answer]
  Q12 --> Q14[QueryResponse]
  Q13 --> Q14
```

Detailed behavior:
1. Planner output decides intent, periods, and tool calls.
2. Tool execution is allowlisted and read-only.
3. Deterministic analytics enriches summary and prediction/recommendation metadata.
4. For known intents, deterministic HTML renderer is preferred.
5. Otherwise, prompt + LLM generation path is used.
6. Response always includes operation metadata in api_summary when available.

Example query walkthrough:
- Input question: Show next month shortage trend for solar and recommend seller action.
- Planner marks prediction/recommendation intent and required tools.
- Tool calls fetch live marketplace data.
- analytics_service computes trend and recommendation scores.
- rag_service selects shortage/recommendation renderer for final HTML answer.
- QueryResponse is returned with answer_mode and api_summary.

---

## 5. LLD for Insert Flow

```mermaid
flowchart TD
  D1[document_routes.upload_document] --> D2[document_service.ingest_document]
  D2 --> D3[opensearch_service.ensure_index_exists]
  D3 --> D4[_fetch_document_from_s3]
  D4 --> D5[_extract_text]
  D5 --> D6[chunker_service.chunk_text]
  D6 --> D7[for each chunk: bedrock_service.embed_text]
  D7 --> D8[opensearch_service.index_chunk]
  D8 --> D9[store DocumentMetadata]
  D9 --> D10[DocumentUploadResponse]
```

Insert flow notes:
- Document ID is deterministic from s3_uri hash.
- Chunk ID format: <document_id>_chunk_<index>.
- Partial indexing is supported when some chunks fail.
- Metadata is currently stored in in-memory store.

---

## 6. FastAPI Runtime Boundaries

Inside FastAPI app boundary:
- main.py, routes, schemas, orchestration services, deterministic business logic.

Outside FastAPI app boundary (called integrations):
- Bedrock APIs
- OpenSearch cluster
- S3 object store
- Marketplace external APIs

The application remains API-first with route thinness and service-heavy orchestration.