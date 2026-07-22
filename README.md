# Rag-Green-Exchange

A Retrieval-Augmented Generation (RAG) powered AI service for the **Green Marketplace** — a blockchain-based platform for trading renewable energy credits. The system combines document-based knowledge retrieval with live marketplace data to deliver intelligent answers about electricity credits, marketplace trends, and trading recommendations.

---

## Overview

The service ingests regulatory documents, policies, and guides about renewable energy trading, indexes them as vector embeddings, and combines that knowledge with real-time marketplace data to answer complex questions with grounded, accurate responses.

---

## Capabilities

### Document Management
- **Upload & Index** — Ingest documents from Amazon S3 in TXT, PDF, or Markdown formats
- **Intelligent Chunking** — Split documents with configurable token size and overlap to preserve context
- **Semantic Embeddings** — Generate vector embeddings via Amazon Bedrock Titan (`amazon.titan-embed-text-v2:0`)
- **kNN Search** — Fast similarity retrieval from Amazon OpenSearch

### Live Marketplace Integration
- **Real-time Data Access** — Query live marketplace endpoints for current supply, demand, and pricing
- **Tool Execution** — Validated tool calls to marketplace backend:
  - `get_all_listings` — Historical supply, listing creation trends, asking prices
  - `get_active_listings` — Current available credits, market share by energy source
  - `get_all_purchases` — Realized demand, completed sales, selling prices
- **Source Normalization** — Standardizes energy source types: `SOLAR`, `WIND`, `HYDRO`, `BIOMASS`, `GEOTHERMAL`, `TIDAL`, `OTHER`

### Analytics & Intelligence
- **Supply Analysis** — Current supply mix percentages, supply by location, supply stability
- **Demand Analysis** — Historical demand, demand by source, demand-to-supply ratio
- **Pricing Intelligence** — Volume-weighted average prices, price volatility trends
- **Market Balance** — Shortage/surplus predictions and market equilibrium analysis
- **Forecasting**
  - Demand prediction (4-period weighted moving average)
  - Price prediction (monthly trend with weekly fallback)
  - Shortage prediction (supply forecast vs. demand)
- **Recommendations**
  - *Seller*: Opportunity scoring — demand growth (35%), demand-to-supply ratio (30%), price strength (20%), low saturation (15%)
  - *Buyer*: Listing scoring — price (40%), demand (30%), quantity (20%), recency (10%)

### Query Processing
- **Intent Recognition** — Planner determines user intent (supply / demand / price / prediction / recommendation)
- **Multi-Source Answers** — Combines RAG retrieval with live marketplace data
- **Confidence Assessment** — Tracks calculation confidence (high / medium / low / insufficient)
- **HTML Response Generation** — Returns polished, sanitized HTML fragments ready for frontend rendering

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/documents/upload` | Ingest document(s) from S3 |
| `GET` | `/documents` | List all indexed documents |
| `GET` | `/documents/{document_id}` | Get metadata for a specific document |
| `POST` | `/documents/clear-index` | Clear all indexed chunks from OpenSearch |
| `POST` | `/query` | Answer questions using indexed documents + marketplace data |

### Query Request / Response

**Request:**
```json
{
  "question": "What is the current solar credit supply trend?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "<HTML fragment>",
  "sources": [{ "chunk_text": "...", "score": 0.92, "document_id": "..." }],
  "api_facts_used": true,
  "api_summary": { "analytics": {}, "predictions": {}, "recommendations": {} },
  "answer_mode": "retrieval_plus_api"
}
```

Answer modes: `retrieval_only` | `retrieval_plus_api` | `api_only` | `insufficient_data`

---

## Architecture

```
app/
├── main.py                        # FastAPI app, CORS, lifespan, routers
├── config.py                      # Centralized config with validation
├── routes/
│   ├── document_routes.py         # Document upload, list, clear
│   └── query_routes.py            # RAG query endpoint
├── schemas/
│   ├── document_schema.py         # Pydantic models for document I/O
│   └── query_schema.py            # Query request/response models
└── services/
    ├── rag_service.py             # End-to-end query orchestration
    ├── bedrock_service.py         # LLM, embeddings, planning
    ├── opensearch_service.py      # Vector indexing and search
    ├── document_service.py        # Document ingestion pipeline
    ├── chunker_service.py         # Text splitting with overlap
    ├── marketplace_api_service.py # Live marketplace API client
    ├── analytics_service.py       # Deterministic analytics engine
    ├── prompt_service.py          # HTML answer prompt building
    ├── tool_registry.py           # Tool catalog for planner
    └── ui_constants.py            # UI display constants
```

### Data Flow

**Document Ingestion (Write Path)**
```
S3 Document → Fetch bytes → Extract text → Chunk → Embed (Bedrock) → Index (OpenSearch)
```

**Query Answering (Read Path)**
```
User Question → Embed → Retrieve RAG chunks → Plan API calls → Execute marketplace tools
             → Analyze results → Build prompt → Generate answer (Bedrock LLM) → Return HTML + sources
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic |
| LLM & Embeddings | Amazon Bedrock (`amazon.nova-micro-v1:0`, `amazon.titan-embed-text-v2:0`) |
| Vector Search | Amazon OpenSearch (kNN) |
| Document Storage | Amazon S3 |
| Auth | AWS SigV4 via EC2 instance roles |
| HTTP Client | httpx (with retries/timeouts) |
| AWS SDK | boto3 |

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (copy and edit as needed)
cp .env.example .env

# Run the service
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs` after startup.

