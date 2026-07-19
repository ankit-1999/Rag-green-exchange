# Green Marketplace - Architecture + HLD

## 1. Purpose and Scope
This document combines Architecture and High-Level Design for the Green Marketplace RAG/Analytics service.

It covers:
- Runtime components in this repository
- External platform dependencies (AWS, Bedrock, OpenSearch, marketplace API)
- Main request/data flows
- Module boundaries and responsibilities
- Deployment/runtime topology

---

## 2. System Context
Green Marketplace provides:
- Document ingestion and indexing for retrieval-augmented responses
- Query answering that combines deterministic marketplace analytics with LLM-generated or deterministic HTML output
- Forecasting/recommendation responses for selected intents

Primary API surfaces:
- `POST /documents/upload`
- `POST /query`
- `GET /health`
- `GET /`

---

## 3. Architecture Overview
```mermaid
flowchart LR
  U[User or External UI Client] --> API[FastAPI Service app/main.py]

  API --> DR[Document Routes app/routes/document_routes.py]
  API --> QR[Query Routes app/routes/query_routes.py]

  DR --> DS[document_service.py]
  DS --> CH[chunker_service.py]
  DS --> BR_EMB[bedrock_service.py Embed Titan]
  DS --> OS[opensearch_service.py]
  DS --> S3[(S3 Documents Bucket)]

  QR --> RAG[rag_service.py]
  RAG --> BR_PLAN[bedrock_service.py Planner Nova]
  RAG --> TOOLREG[tool_registry.py]
  RAG --> MKT[marketplace_api_service.py]
  RAG --> AN[analytics_service.py]
  RAG --> PR[prompt_service.py]
  RAG --> BR_ANS[bedrock_service.py Answer Nova]
  RAG --> OS

  BR_EMB --> BED[(Amazon Bedrock)]
  BR_PLAN --> BED
  BR_ANS --> BED
  OS --> OSD[(OpenSearch Domain)]
  MKT --> EXT[(External Marketplace API)]
```

---

## 4. Component Inventory (All Major Components)

### 4.1 Application Entry and Config
- `app/main.py`
  - FastAPI app construction
  - Middleware (CORS, Referrer-Policy)
  - Router registration
  - Startup lifecycle (`ensure_index_exists`)
  - Health/root endpoints
- `app/config.py`
  - Environment-backed settings
  - Bedrock model IDs and generation limits
  - OpenSearch endpoint/index/top-k
  - Marketplace API base URL/timeouts/page size
  - App metadata/security headers

### 4.2 API Layer
- `app/routes/document_routes.py`
  - Document upload/indexing endpoint(s)
- `app/routes/query_routes.py`
  - Query orchestration endpoint(s)

### 4.3 Schemas and Contracts
- `app/schemas/document_schema.py`
  - Request/response models for document workflows
- `app/schemas/query_schema.py`
  - Query request/response models
  - API summary structures, tool result shapes, periods, sources

### 4.4 Core Services
- `app/services/rag_service.py`
  - End-to-end query orchestration
  - Planner call, tool execution, analytics binding, response rendering
  - Deterministic renderers by intent and fallback handling
- `app/services/bedrock_service.py`
  - Embeddings generation
  - Planner prompt and planner output normalization/enforcement
  - Final answer generation
- `app/services/analytics_service.py`
  - Deterministic analytics by intent (supply, demand, balance, ratio, prices)
  - Forecasting (demand/price/shortage)
  - Recommendation scoring (seller/buyer)
  - Marketplace summary period analytics
- `app/services/marketplace_api_service.py`
  - Read-only external marketplace API execution
  - Paging/timeout handling and normalized results
- `app/services/tool_registry.py`
  - Allowed tool definitions and payload schemas
  - Planner tool catalog text
- `app/services/opensearch_service.py`
  - OpenSearch index management
  - Document chunk indexing and vector retrieval
- `app/services/document_service.py`
  - Ingestion orchestration: fetch, parse/chunk, embed, index
- `app/services/chunker_service.py`
  - Token-aware chunking with overlap
- `app/services/prompt_service.py`
  - RAG prompt assembly with API context and HTML-answer constraints

### 4.5 Deployment and Operations Assets
- `aws/rag-cloud-formation.yaml`
  - Infra provisioning (VPC/subnet/networking, EC2 runtime, OpenSearch, S3, IAM, monitoring)

### 4.6 External Dependencies (Runtime)
- AWS Bedrock
  - `amazon.titan-embed-text-v2:0` embeddings
  - `amazon.nova-micro-v1:0` planner/answer model
- OpenSearch (vector index + metadata)
- S3 (document source objects)
- External Marketplace API (listings/purchases)

### 4.7 Python Runtime Dependencies
From `requirements.txt`:
- FastAPI + Uvicorn
- Pydantic + pydantic-settings
- python-dotenv
- boto3
- opensearch-py + requests-aws4auth
- requests
- httpx

---

## 5. High-Level Flows

### 5.1 Document Ingestion Flow
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant DocRoute as document_routes.py
    participant DocSvc as document_service.py
    participant Chunk as chunker_service.py
    participant Bedrock as bedrock_service.py
    participant OS as opensearch_service.py
    participant S3 as S3

    Client->>DocRoute: POST /documents/upload
    DocRoute->>DocSvc: validate + process
    DocSvc->>S3: fetch document
    DocSvc->>Chunk: split into chunks
    DocSvc->>Bedrock: embed chunks
    DocSvc->>OS: index vectors + metadata
    OS-->>DocSvc: index result
    DocSvc-->>DocRoute: upload/index summary
    DocRoute-->>Client: response
```

### 5.2 Query/RAG + Analytics Flow
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant QueryRoute as query_routes.py
    participant RAG as rag_service.py
    participant Bedrock as bedrock_service.py
    participant ToolReg as tool_registry.py
    participant Mkt as marketplace_api_service.py
    participant An as analytics_service.py
    participant Prompt as prompt_service.py
    participant OS as opensearch_service.py

    Client->>QueryRoute: POST /query
    QueryRoute->>RAG: answer_question
    RAG->>Bedrock: plan_api_calls(question)
    Bedrock->>ToolReg: validate/enforce tools
    RAG->>Mkt: execute approved tool calls
    Mkt-->>RAG: normalized tool results
    RAG->>An: analyze_plan(plan, results)
    RAG->>OS: retrieve supporting chunks (intent-dependent)
    RAG->>Prompt: build_rag_prompt
    RAG->>Bedrock: generate_answer OR deterministic renderer
    Bedrock-->>RAG: HTML answer
    RAG-->>QueryRoute: QueryResponse
    QueryRoute-->>Client: response
```

---

## 6. HLD: Module Responsibilities and Boundaries

### 6.1 API Boundary
- Routes handle transport concerns only (request parsing/response delivery).
- Business logic lives in services.

### 6.2 Planner/Execution Boundary
- Planner can propose intent/tools but Python enforces:
  - Allowed tool names
  - Argument normalization
  - Required tool combinations for intents
  - Period/source/location constraints

### 6.3 Analytics Boundary
- `analytics_service.py` is deterministic and side-effect free.
- It consumes normalized tool results and emits computed structures used by renderer/LLM.

### 6.4 Rendering Boundary
- `rag_service.py` determines whether to:
  - Use deterministic renderer for known intents
  - Or use LLM answer generation
  - Or fallback response for insufficient/failed generation

### 6.5 Retrieval Boundary
- OpenSearch retrieval is optional for some intents (summary/demand-supply can be API-only paths).

---

## 7. Key Data Contracts
- Planner output: intent, tool calls, periods, groupings, metrics, flags
- Tool result envelope:
  - `tool`
  - `arguments`
  - `execution_status`
  - `record_count`
  - `data.records` (raw)
  - `data.sample_records` (compact)
  - `data.aggregates` (compact deterministic basis)
- Query response envelope:
  - HTML answer
  - answer mode
  - sources
  - API summary (intent, filters, analytics, predictions/recommendations)

---

## 8. Deployment Topology (HLD View)
- Runtime host: EC2 instance running FastAPI/Uvicorn service
- Vector/search store: OpenSearch domain
- Document storage: S3 bucket
- LLM/Embeddings: Bedrock APIs in configured region
- Monitoring: CloudWatch logs/alarms and budget alarms
- Network/security: VPC, subnet, SGs, IAM role/policies from CloudFormation

---

## 9. Security and Governance Controls
- Read-only marketplace tool execution constrained by tool registry
- API argument sanitization/normalization before external calls
- Environment-based configuration and endpoint validation
- Referrer-Policy response header middleware
- IAM-scoped AWS access for EC2 runtime

---

## 10. Design Notes and Trade-offs
- Deterministic analytics + deterministic renderers reduce hallucination risk for marketplace questions.
- Aggregate-driven computation improves resilience when record payloads are partial.
- Planner flexibility is retained, but intent/tool safety is enforced in code.
- HTML response strategy keeps UI rendering predictable across intents.

---

## 11. Future Evolution (Suggested)
- Add explicit versioned API contracts for tool result aggregates.
- Add cached analytics snapshots for repeated period queries.
- Add structured observability for intent/tool/latency per stage.
- Add test suites for all deterministic renderers and period filtering rules.

