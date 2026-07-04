# GreenGrid RAG API - End-to-End Flow

## Purpose
This document explains the complete request lifecycle of the GreenGrid RAG system, from document ingestion to grounded AI answer generation.

The platform has two major runtime flows:
1. Ingestion flow (write path): takes a source document and builds vector-searchable chunks.
2. Query flow (read path): takes a user question, retrieves relevant chunks, and generates a grounded answer.

---

## High-Level Architecture Layers

### Layer 1: API Layer
Responsibilities:
1. Receive HTTP requests.
2. Validate request payloads.
3. Route to service orchestration.
4. Return structured JSON responses and proper status codes.

Main endpoints:
1. POST /documents/upload
2. GET /documents
3. GET /documents/{document_id}
4. POST /query
5. GET /health

### Layer 2: Processing Layer (Business Logic)
Responsibilities:
1. Parse and validate S3 metadata.
2. Extract and normalize text.
3. Chunk text with overlap.
4. Generate embeddings.
5. Build prompts and orchestrate RAG.

### Layer 3: Retrieval + Vector Layer (OpenSearch)
Responsibilities:
1. Store chunk records.
2. Store vector embeddings.
3. Perform kNN search over embedding vectors.
4. Return top-k relevant chunk matches with scores.

### Layer 4: AI Inference Layer (Bedrock)
Responsibilities:
1. Embedding generation with Titan model.
2. Final answer generation with Claude model.

### Layer 5: Storage Layer
Responsibilities:
1. Raw source documents in S3.
2. Indexed chunks and vectors in OpenSearch.
3. Runtime metadata cache in application memory.

---

## End-to-End Flow A: Document Ingestion (Chunking + Indexing)

### Step 1: Client uploads source file to S3
The file is stored in the configured bucket (for example, policy docs, guides, reports).

### Step 2: Client calls POST /documents/upload
Request payload includes:
1. document_name
2. document_type
3. s3_uri

### Step 3: API validation
The API validates request shape and required fields using schema models.

### Step 4: Service orchestration starts
The ingestion service generates a document_id and starts structured processing.

### Step 5: OpenSearch index readiness check
The system ensures the target index exists with kNN mapping and vector settings.

### Step 6: S3 fetch
The document bytes are fetched using AWS SDK and IAM role credentials.

### Step 7: Text extraction
For text-like files, bytes are decoded to UTF-8 text.
For PDF files, the current implementation uses placeholder behavior and logs extraction limitations.

### Step 8: Chunking layer
Text is split into overlapping chunks using token-based sizing approximated through character windows.

Why overlap is used:
1. Preserves context across chunk boundaries.
2. Improves retrieval relevance.

### Step 9: Embedding generation
Each chunk is sent to Bedrock Titan Embeddings model.
Output is a dense vector of configured dimension.

### Step 10: Vector indexing
Each chunk is indexed into OpenSearch with:
1. chunk metadata (ids, names, positions, S3 URI)
2. chunk text
3. embedding vector

### Step 11: Metadata response
System returns ingestion summary:
1. status (indexed, partial, failed)
2. chunk_count
3. document_id
4. message

---

## End-to-End Flow B: Query to AI Answer (RAG)

### Step 1: Client calls POST /query
Request payload includes:
1. question
2. optional top_k

### Step 2: API validation
Schema validation ensures non-empty question and valid optional controls.

### Step 3: Question embedding
The question text is embedded using Bedrock Titan Embeddings.

### Step 4: Retrieval from vector index
OpenSearch kNN query runs against embedding field in index.
Top-k nearest chunks are returned with relevance scores.

### Step 5: Prompt assembly layer
Prompt service builds a grounded prompt containing:
1. user question
2. selected context chunks
3. source references (document and chunk identifiers)
4. instruction to avoid hallucination and stay within context

### Step 6: AI response generation
Prompt is sent to Bedrock Claude model.
Claude returns a synthesized answer based on retrieved chunk context.

### Step 7: Citation packaging
Response includes:
1. final answer
2. source_count
3. sources list with chunk details, scores, and snippets

### Step 8: API returns final JSON
Client receives a grounded answer with traceable source evidence.

---

## Data Contracts Summary

### Ingestion Request
Fields:
1. document_name
2. document_type
3. s3_uri

### Ingestion Response
Fields:
1. document_id
2. document_name
3. status
4. chunk_count
5. message

### Query Request
Fields:
1. question
2. top_k (optional)

### Query Response
Fields:
1. answer
2. source_count
3. sources[]

Each source contains:
1. chunk_id
2. document_id
3. document_name
4. document_type
5. chunk_index
6. s3_uri
7. score
8. snippet

---

## How Chunking, RAG, and AI Response Work Together

### Chunking role
Chunking transforms long documents into retrieval-sized semantic units.
Without chunking, embedding quality and retrieval precision degrade on large texts.

### RAG role
RAG grounds model output by retrieving top relevant chunks first.
This reduces hallucination risk and makes outputs auditable.

### AI response role
The LLM does not answer from broad memory alone.
It is guided by retrieved evidence and prompt constraints.

Combined effect:
1. Better answer quality for domain questions.
2. Explainable outputs through citations.
3. Repeatable behavior for demos and audits.

---

## Operational Notes

### Where raw files are stored
S3 bucket configured by S3_BUCKET_NAME.

### Where vectors/chunks are stored
OpenSearch index configured by OPENSEARCH_INDEX_NAME.

### What is not persistent yet
Document metadata list endpoint currently uses in-memory store.
A process restart clears that runtime metadata cache.

---

## Current Capabilities
1. Ingestion from S3 to chunked vector index.
2. Query endpoint with retrieval + generation.
3. Source-cited response payload.
4. Basic route-level test coverage for query endpoint.

---

## Suggested Next Enhancements
1. Persist document metadata to a database (instead of in-memory cache).
2. Add retrieval-only debug endpoint (no LLM) for ranking analysis.
3. Add relevance threshold and fallback responses for low-confidence retrieval.
4. Add token/cost observability per request.
5. Add PDF-native extraction pipeline for production-grade ingestion.

---

## Detailed Code Walkthrough with Code (FastAPI Beginner Guide)

This section explains exactly where each step runs in code.

## 1) App Startup and Router Wiring

**File: `app/main.py`**

```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
```
- `logging` — Python standard library for writing log messages.
- `asynccontextmanager` — Converts an async generator function into a reusable context manager, used to wire startup/shutdown hooks.
- `FastAPI` — The core class that creates the web application object.
- `CORSMiddleware` — Middleware that adds CORS headers so browsers (and tools like Bruno) can call the API from any origin.

```python
from app.config import settings
from app.routes.document_routes import router as document_router
from app.routes.query_routes import router as query_router
from app.services import opensearch_service
```
- `settings` — Central config object. Every service reads from here.
- `document_router` and `query_router` — Router objects from the respective route files. They carry the endpoint definitions.
- `opensearch_service` — Imported here only for the startup index check.

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)
```
- `basicConfig` — Configures Python logging globally. All modules that call `logging.getLogger(__name__)` will use this format.
- `logger = logging.getLogger(__name__)` — Creates a logger specific to this file. `__name__` becomes `app.main`.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.APP_NAME)
    try:
        opensearch_service.ensure_index_exists()
    except Exception as exc:
        logger.error("OpenSearch startup check failed: %s", exc)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)
```
- `@asynccontextmanager` — Marks this function as a lifecycle hook.
- Code **before** `yield` runs on **startup**.
- Code **after** `yield` runs on **shutdown**.
- `ensure_index_exists()` — Creates OpenSearch index if not present. If it fails, exception is caught and logged. App still starts.
- `yield` — Tells FastAPI: startup is done, begin accepting requests.

```python
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)
```
- Creates the FastAPI app object.
- `title` and `version` appear in the `/docs` Swagger UI.
- `lifespan=lifespan` — Wires the startup/shutdown hook above.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- Every HTTP response will include CORS headers.
- `allow_origins=["*"]` — Any client can call this API. Restrict in production.

```python
app.include_router(document_router)
app.include_router(query_router)
```
- These two lines register all route functions from both route files into the app.
- Without this, the endpoints would not exist even if defined in route files.

```python
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "greengrid-rag-ai"}
```
- `@app.get("/health")` — Registers a GET endpoint at `/health`.
- `async def` — FastAPI allows async functions for non-blocking IO.
- Returns a plain dict. FastAPI automatically converts it to JSON response.

**File: `app/config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()
```
- `os` — Standard library to read environment variables.
- `load_dotenv()` — Reads `.env` file from working directory and loads values into the process environment. Called once at import time.

```python
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "GreenGrid Exchange RAG AI")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_EMBEDDING_MODEL_ID: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
    )
    BEDROCK_EMBEDDING_DIMENSION: int = int(
        os.getenv("BEDROCK_EMBEDDING_DIMENSION", "1024")
    )
    OPENSEARCH_ENDPOINT: str = os.getenv("OPENSEARCH_ENDPOINT", "https://localhost:9200")
    OPENSEARCH_INDEX_NAME: str = os.getenv("OPENSEARCH_INDEX_NAME", "greengrid-docs")
    OPENSEARCH_TOP_K: int = int(os.getenv("OPENSEARCH_TOP_K", "5"))
    OPENSEARCH_USE_AWS_AUTH: bool = (
        os.getenv("OPENSEARCH_USE_AWS_AUTH", "true").lower() == "true"
    )
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "greengrid-documents")
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
```
- Each attribute calls `os.getenv("KEY", "default")`.
- If `.env` has `AWS_REGION=us-east-1` then `settings.AWS_REGION` returns `us-east-1`.
- If the value is not set, the default string is used.
- `int(os.getenv(...))` — env vars are always strings; casting to int here ensures type safety.
- `OPENSEARCH_USE_AWS_AUTH` — converts the string `"true"` to Python bool `True`.

```python
settings = Settings()
```
- Creates one shared instance. All other files do `from app.config import settings` and share this object.

## 3) Request and Response Schemas

**File: `app/schemas/query_schema.py`**

```python
from pydantic import BaseModel, Field
from typing import List, Optional
```
- `BaseModel` — All schema classes inherit from this. It provides automatic validation and JSON parsing.
- `Field` — Used to set constraints like min_length, ge (greater or equal), le (less or equal).

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language user question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
```
- `question: str` — required field (indicated by `...`). FastAPI returns 422 if missing or too short.
- `min_length=3` — FastAPI validates this automatically before reaching your route function.
- `top_k: Optional[int]` — Optional field. If not sent, it will be `None`.
- `ge=1, le=20` — Accepted range for top_k is 1 to 20.

```python
class QuerySource(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    chunk_index: int
    s3_uri: str
    score: float
    snippet: str
```
- Each field maps to a value from an OpenSearch hit.
- `score: float` — Relevance score from kNN search.
- `snippet: str` — First 280 characters of chunk text used for display.

```python
class QueryResponse(BaseModel):
    answer: str
    source_count: int
    sources: List[QuerySource]
```
- This is what the API returns as JSON to the client.
- `sources: List[QuerySource]` — List of citation objects used to generate the answer.

## 4) Query API Route

**File: `app/routes/query_routes.py`**

```python
from fastapi import APIRouter, HTTPException, status
from app.schemas.query_schema import QueryRequest, QueryResponse
from app.services import rag_service
```
- `APIRouter` — A mini app that groups related routes. Prefix `/query` is applied to all routes inside.
- `HTTPException` — Used to return error responses with specific status codes and messages.
- `status` — Provides named constants like `status.HTTP_200_OK`.

```python
router = APIRouter(prefix="/query", tags=["RAG Query"])
```
- `prefix="/query"` — All routes defined in this router will start with `/query`.
- `tags=["RAG Query"]` — Groups endpoints under this label in the `/docs` Swagger UI.

```python
@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
async def query_documents(request: QueryRequest) -> QueryResponse:
```
- `@router.post("")` — Registers a POST endpoint at `/query` (empty string + prefix = `/query`).
- `response_model=QueryResponse` — FastAPI validates and filters the return value against this schema.
- `request: QueryRequest` — FastAPI automatically parses and validates the JSON body into this object.

```python
    try:
        return rag_service.answer_question(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
```
- `rag_service.answer_question(request)` — Calls the RAG orchestration service with the validated request.
- `except ValueError` — Raised for bad input (for example, empty embeddings). Returns 400 Bad Request.
- `except RuntimeError` — Raised when Bedrock or OpenSearch fails. Returns 502 Bad Gateway.
- `from exc` — Preserves the original exception chain in logs for debugging.

## 5) Chunking Layer

**File: `app/services/chunker_service.py`**

```python
def _estimate_tokens(text: str) -> int:
    return len(text) // 4
```
- Private helper (underscore prefix = not meant to be imported by other files).
- Estimates token count by dividing character count by 4.
- Approximation: English prose averages about 4 characters per token.

```python
def _tokens_to_chars(token_count: int) -> int:
    return token_count * 4
```
- Converts a token budget back to a character count for slicing text.

```python
def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE_TOKENS,
    overlap: int = settings.CHUNK_OVERLAP_TOKENS,
) -> List[str]:
```
- `chunk_size` and `overlap` use values from `settings` as defaults.
- `List[str]` — Return type annotation for static analysis and IDE support.

```python
    if not text or not text.strip():
        logger.warning("chunk_text received empty or blank text")
        return []
```
- Guards against empty/whitespace-only input.
- Returns empty list, which signals upstream to handle missing chunks.

```python
    chunk_chars = _tokens_to_chars(chunk_size)    # e.g. 500 tokens = 2000 chars
    overlap_chars = _tokens_to_chars(overlap)     # e.g. 50 tokens = 200 chars
    step = chunk_chars - overlap_chars            # e.g. 2000 - 200 = 1800 chars per step
```
- `chunk_chars` — How many characters in each chunk.
- `overlap_chars` — How many characters repeat at start of next chunk.
- `step` — How far the window advances each iteration.

```python
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
```
- Sliding window loop over the full text.
- `text[start:end]` — Slices the current window of characters.
- `.strip()` — Removes leading/trailing whitespace from each chunk.
- `if chunk:` — Skips empty windows (can happen at end of text).
- `start += step` — Advance window by step size (not full chunk size, so overlap is preserved).

## 8) Embedding and LLM Calls

**File: `app/services/bedrock_service.py`**

```python
def _get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
```
- Creates a Bedrock client using the default credential chain.
- On EC2, this picks up the IAM role automatically.
- `bedrock-runtime` is the API endpoint for model invocation (not model management).

```python
def embed_text(text: str) -> List[float]:
    client = _get_bedrock_client()
    body = json.dumps({"inputText": text})
    response = client.invoke_model(
        modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    embedding: List[float] = result["embedding"]
    return embedding
```
- `json.dumps` — Converts Python dict to JSON string for the API body.
- `invoke_model` — Calls the Bedrock API with the model ID and JSON payload.
- `response["body"].read()` — `body` is a streaming object; `.read()` reads all bytes.
- `result["embedding"]` — Titan returns a key named `embedding` with a list of floats.

```python
def generate_answer(prompt: str) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": settings.BEDROCK_LLM_MAX_TOKENS,
        "temperature": settings.BEDROCK_LLM_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = _get_bedrock_client().invoke_model(
        modelId=settings.BEDROCK_LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    answer: str = result["content"][0]["text"]
    return answer
```
- `anthropic_version` — Required header for Claude models on Bedrock.
- `messages` — Claude uses a chat messages format. `role: user` means the prompt is user-authored.
- `temperature: 0.0` — Lower temperature = more deterministic answers. Good for factual RAG.
- `result["content"][0]["text"]` — Claude returns a list of content blocks; first one is the text answer.

## 9) OpenSearch Layer (Vector DB and kNN Search)

**File: `app/services/opensearch_service.py`**

```python
def _get_client() -> OpenSearch:
    host_no_scheme = host.replace("https://", "").replace("http://", "")
    if settings.OPENSEARCH_USE_AWS_AUTH:
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, settings.AWS_REGION, "es")
        return OpenSearch(
            hosts=[{"host": host_no_scheme, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
```
- `AWSV4SignerAuth` — Signs each HTTP request with the IAM SigV4 signature so OpenSearch trusts the caller.
- `RequestsHttpConnection` — Uses the `requests` library as the HTTP backend.
- `verify_certs=True` — Validates the HTTPS certificate of the OpenSearch domain.

```python
INDEX_MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": settings.BEDROCK_EMBEDDING_DIMENSION,
                "method": {"name": "hnsw", "space_type": "l2", "engine": "nmslib"},
            },
        }
    },
}
```
- `knn: True` — Enables k-nearest-neighbours plugin for this index.
- `type: knn_vector` — Special field type that stores dense float vectors.
- `dimension` — Must match the output dimension of the embedding model (1024 for Titan V2).
- `hnsw` — Hierarchical Navigable Small World graph algorithm. Efficient for approximate nearest neighbour search.
- `space_type: l2` — Uses L2 (Euclidean) distance for similarity comparison.

```python
def ensure_index_exists() -> None:
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=INDEX_MAPPING)
```
- Idempotent check. Safe to call on every app start.
- Only creates index if it does not already exist.

```python
def index_chunk(...) -> None:
    doc = {
        "chunk_id": chunk_id,
        "text": text,
        "embedding": embedding,
        ...
    }
    client.index(index=settings.OPENSEARCH_INDEX_NAME, id=chunk_id, body=doc)
```
- Stores one chunk as an OpenSearch document.
- `id=chunk_id` — Uses chunk_id as the document primary key. Re-indexing same id overwrites it.

```python
def search_similar_chunks(query_embedding, top_k) -> List[Dict]:
    query_body = {
        "size": top_k,
        "query": {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}}
    }
    response = client.search(index=settings.OPENSEARCH_INDEX_NAME, body=query_body)
    hits = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        src["score"] = hit["_score"]
        hits.append(src)
    return hits
```
- `knn` query compares the query vector against all indexed `embedding` vectors.
- `k: top_k` — Returns top k most similar documents.
- `hit["_source"]` — The actual stored document fields (text, metadata).
- `hit["_score"]` — Similarity score. Higher means more similar.

## 7) RAG Orchestration Service (Core Read Path)

**File: `app/services/rag_service.py`**

```python
from app.config import settings
from app.schemas.query_schema import QueryRequest, QueryResponse, QuerySource
from app.services import bedrock_service, opensearch_service, prompt_service
```
- Imports all three dependent services needed for the RAG pipeline.

```python
def answer_question(request: QueryRequest) -> QueryResponse:
```
- Single entry point for the entire RAG flow.
- Accepts the validated `QueryRequest` object directly from the route layer.

```python
    top_k = request.top_k or settings.OPENSEARCH_TOP_K
```
- If client did not send `top_k`, fall back to the configured default (usually 5).
- `or` here acts as a null-coalescing: if `request.top_k` is `None`, use the right-hand value.

```python
    question_embedding = bedrock_service.embed_text(request.question)
```
- Converts the question string into a numeric vector using Titan model.
- This vector is used to search OpenSearch by similarity.

```python
    hits: List[Dict] = opensearch_service.search_similar_chunks(question_embedding, top_k=top_k)
```
- Runs kNN vector search in OpenSearch.
- Returns list of dicts. Each dict has text, metadata, and similarity score.

```python
    if not hits:
        return QueryResponse(
            answer="I could not find relevant indexed context...",
            source_count=0,
            sources=[],
        )
```
- Safe fallback: if no relevant chunks found, return a clean message instead of crashing.
- This happens when the question is about a topic not yet indexed.

```python
    prompt = prompt_service.build_rag_prompt(request.question, hits)
    answer = bedrock_service.generate_answer(prompt)
```
- `build_rag_prompt` builds the full structured prompt with context.
- `generate_answer` sends it to Claude and gets back a text answer.

```python
    sources: List[QuerySource] = []
    for hit in hits:
        sources.append(
            QuerySource(
                chunk_id=hit.get("chunk_id", ""),
                document_id=hit.get("document_id", ""),
                document_name=hit.get("document_name", ""),
                document_type=hit.get("document_type", ""),
                chunk_index=int(hit.get("chunk_index", 0)),
                s3_uri=hit.get("s3_uri", ""),
                score=float(hit.get("score", 0.0)),
                snippet=(hit.get("text", "")[:280]).replace("\n", " "),
            )
        )
```
- Converts raw OpenSearch hit dicts to typed `QuerySource` Pydantic objects.
- `hit.get("chunk_id", "")` — Safe read with fallback to empty string.
- `[:280]` — Truncates chunk text to 280 chars for the snippet display.
- `.replace("\n", " ")` — Removes newlines so snippet is single-line.

```python
    return QueryResponse(answer=answer, source_count=len(sources), sources=sources)
```
- Builds and returns the final response object.
- FastAPI will serialize this to JSON automatically.

## 6) Prompt Construction Layer

**File: `app/services/prompt_service.py`**

```python
def build_rag_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
```
- `retrieved_chunks` — List of dicts. Each dict is one OpenSearch hit.

```python
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[SOURCE {idx}]\n"
            f"document_name: {chunk.get('document_name', 'unknown')}\n"
            f"chunk_id: {chunk.get('chunk_id', 'unknown')}\n"
            f"chunk_index: {chunk.get('chunk_index', 'unknown')}\n"
            f"content:\n{chunk.get('text', '')}"
        )
```
- `enumerate(retrieved_chunks, start=1)` — Loops with a counter starting at 1 (for human-readable source numbering).
- `chunk.get('document_name', 'unknown')` — Safely reads each field from the hit dict. Falls back to `'unknown'` if key is missing.
- Each block clearly identifies source and content.

```python
    context_text = "\n\n".join(context_blocks)
```
- Joins all source blocks with two newlines as separator for readability in the prompt.

```python
    return (
        "You are a precise assistant for GreenGrid Exchange.\n"
        "Use only the provided CONTEXT to answer.\n"
        "If context is insufficient, say that explicitly and ask for more data.\n"
        "Do not fabricate policy or numeric values.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Return a concise answer in plain text."
    )
```
- System instruction section tells Claude what role to play and what constraints to follow.
- `Use only the provided CONTEXT` — This instruction is the key anti-hallucination guard.
- `Do not fabricate policy or numeric values` — Specific safety instruction for energy/policy domain.
- `QUESTION` and `CONTEXT` sections are clearly labeled for the model to parse structure.

## 10) Tests for Query Route

**File: `tests/test_query_route.py`**

```python
from fastapi.testclient import TestClient
from app.main import app
```
- `TestClient` — FastAPI test utility that sends requests directly to the app without running a real server.
- `from app.main import app` — Imports the real app object so routes and middleware are all active.

```python
def test_query_success(monkeypatch):
    def fake_answer_question(_request):
        return QueryResponse(answer="sample answer", source_count=1, sources=[...])
    monkeypatch.setattr(rag_service, "answer_question", fake_answer_question)
```
- `monkeypatch` — pytest fixture that temporarily replaces functions for testing.
- `monkeypatch.setattr(rag_service, "answer_question", fake_answer_question)` — Replaces the real RAG call with a fake function so the test does not call Bedrock or OpenSearch.

```python
    client = TestClient(app)
    res = client.post("/query", json={"question": "What is GreenGrid Exchange?", "top_k": 3})
    assert res.status_code == 200
    assert res.json()["answer"] == "sample answer"
```
- Sends a POST request to `/query`.
- Asserts that status code is 200 and JSON body contains expected answer.

```python
def test_query_empty_question_validation():
    res = client.post("/query", json={"question": ""})
    assert res.status_code == 422
```
- Empty string fails the `min_length=3` constraint on `QueryRequest.question`.
- FastAPI automatically returns 422 before the route function even runs.

---

## Debugging Guide by Symptom

### Symptom: /documents/upload returns 500
Check in order:
1. app/services/document_service.py (orchestration errors)
2. app/services/opensearch_service.py (index/connectivity/403)
3. app/services/bedrock_service.py (invoke model permissions)
4. EC2 journal logs for traceback

### Symptom: /query returns answer with no sources
Check in order:
1. query top_k value
2. search_similar_chunks output in opensearch_service
3. whether ingestion actually indexed chunks

### Symptom: 422 from FastAPI
Meaning:
1. Request body did not match schema.
2. Validate JSON fields and types against schema classes.
