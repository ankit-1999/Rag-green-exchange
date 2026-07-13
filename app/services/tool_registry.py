import json
from typing import Dict, List, Optional

# Central tool catalog used by:
# 1) LLM planner prompt generation
# 2) Backend tool-call validation/execution routing
_TOOL_CATALOG: List[Dict] = [
    {
        "name": "create_document",
        "purpose": "Ingest a document from S3 and index it for retrieval.",
        "when_to_use": [
            "Question asks to upload or register a new document",
            "Assistant needs to trigger ingestion before answering",
        ],
        "payload_schema": {
            "document_name": "string",
            "document_type": "string (POLICY|RULE|REPORT|GUIDE)",
            "s3_uri": "string (s3://bucket/key)",
        },
        "response_schema": {
            "document_id": "string",
            "document_name": "string",
            "status": "indexed|partial|failed",
            "chunk_count": "int",
            "message": "string",
        },
        "example_request": {
            "document_name": "policy.txt",
            "document_type": "GUIDE",
            "s3_uri": "s3://my-bucket/policy.txt",
        },
        "example_response": {
            "document_id": "doc_1234abcd",
            "document_name": "policy.txt",
            "status": "indexed",
            "chunk_count": 12,
            "message": "Document indexed successfully. 12/12 chunks stored.",
        },
    },
    {
        "name": "get_documents_summary",
        "purpose": "Fetch operational summary of currently ingested documents.",
        "when_to_use": [
            "Question asks for total document count",
            "Question asks for document type breakdown",
            "Question asks what documents are available",
        ],
        "payload_schema": {},
        "response_schema": {
            "total_documents": "int",
            "by_type": "object<string,int>",
            "sample_document_names": "list<string>",
        },
        "example_request": {},
        "example_response": {
            "total_documents": 12,
            "by_type": {"GUIDE": 8, "REPORT": 4},
            "sample_document_names": ["doc_a.txt", "doc_b.pdf"],
        },
    },
    {
        "name": "list_documents",
        "purpose": "List ingested document metadata entries.",
        "when_to_use": [
            "Question asks to show all documents",
            "Need document IDs for follow-up lookup",
        ],
        "payload_schema": {
            "limit": "optional int",
        },
        "response_schema": {
            "documents": "list<object>",
        },
        "example_request": {"limit": 10},
        "example_response": {
            "documents": [
                {
                    "document_id": "doc_1234abcd",
                    "document_name": "policy.txt",
                    "document_type": "GUIDE",
                    "s3_uri": "s3://bucket/key",
                    "chunk_count": 8,
                }
            ]
        },
    },
    {
        "name": "get_document",
        "purpose": "Get metadata of one document by document_id.",
        "when_to_use": [
            "Question asks details of specific document",
            "Need to verify document exists",
        ],
        "payload_schema": {
            "document_id": "string",
        },
        "response_schema": {
            "document": "object|null",
        },
        "example_request": {"document_id": "doc_1234abcd"},
        "example_response": {
            "document": {
                "document_id": "doc_1234abcd",
                "document_name": "policy.txt",
                "document_type": "GUIDE",
                "s3_uri": "s3://bucket/key",
                "chunk_count": 8,
            }
        },
    },
]


def get_tool_catalog() -> List[Dict]:
    return list(_TOOL_CATALOG)


def get_tool_by_name(name: str) -> Optional[Dict]:
    for tool in _TOOL_CATALOG:
        if tool["name"] == name:
            return tool
    return None


def get_allowed_tool_names() -> List[str]:
    return [tool["name"] for tool in _TOOL_CATALOG]


def build_planner_tools_text() -> str:
    """Render tool catalog as compact JSON for planner prompt context."""
    return json.dumps(_TOOL_CATALOG, ensure_ascii=True, indent=2)
