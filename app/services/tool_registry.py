import json
from typing import Dict, List, Optional

# Central tool catalog used by:
# 1) LLM planner prompt generation
# 2) Backend tool-call validation/execution routing
_TOOL_CATALOG: List[Dict] = [
    {
        "name": "create_user",
        "purpose": "Create a new marketplace user.",
        "when_to_use": [
            "Question asks to register/create a user",
            "Assistant needs a user profile before credit operations",
        ],
        "payload_schema": {
            "first_name": "string",
            "last_name": "string",
            "age": "int (1..120)",
            "city": "string",
            "role": "seller|buyer|admin",
        },
        "response_schema": {
            "user_id": "string",
            "first_name": "string",
            "last_name": "string",
            "age": "int",
            "city": "string",
            "role": "string",
            "created_at": "datetime",
        },
        "example_request": {
            "first_name": "Ankit",
            "last_name": "Singh",
            "age": 27,
            "city": "Noida",
            "role": "seller",
        },
        "example_response": {
            "user_id": "user_ab12cd34",
            "first_name": "Ankit",
            "last_name": "Singh",
            "age": 27,
            "city": "Noida",
            "role": "seller",
            "created_at": "2026-07-07T05:30:00+00:00",
        },
    },
    {
        "name": "get_user",
        "purpose": "Fetch a single user's details by user_id.",
        "when_to_use": [
            "Question asks user details",
            "Need to verify a user exists before transfer/create actions",
        ],
        "payload_schema": {
            "user_id": "string",
        },
        "response_schema": {
            "user": "object|null",
        },
        "example_request": {"user_id": "user_ab12cd34"},
        "example_response": {
            "user": {
                "user_id": "user_ab12cd34",
                "first_name": "Ankit",
                "last_name": "Singh",
                "age": 27,
                "city": "Noida",
                "role": "seller",
                "created_at": "2026-07-07T05:30:00+00:00",
            }
        },
    },
    {
        "name": "list_users",
        "purpose": "List all users in the application store.",
        "when_to_use": [
            "Question asks to show all users",
            "Need user selection context for transfer",
        ],
        "payload_schema": {},
        "response_schema": {
            "users": "list<object>",
        },
        "example_request": {},
        "example_response": {
            "users": [
                {
                    "user_id": "user_ab12cd34",
                    "first_name": "Ankit",
                    "last_name": "Singh",
                    "age": 27,
                    "city": "Noida",
                    "role": "seller",
                    "created_at": "2026-07-07T05:30:00+00:00",
                }
            ]
        },
    },
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
    {
        "name": "create_credit",
        "purpose": "Create a new credit owned by a user.",
        "when_to_use": [
            "Question asks to create/list a new credit listing",
            "Need credit object for later transfer",
        ],
        "payload_schema": {
            "user_id": "string",
            "credit_type": "solar|wind|coal",
            "price": "float > 0",
        },
        "response_schema": {
            "credit_id": "string",
            "credit_code": "string",
            "user_id": "string",
            "credit_type": "string",
            "price": "float",
            "created_at": "datetime",
        },
        "example_request": {
            "user_id": "user_ab12cd34",
            "credit_type": "solar",
            "price": 120.5,
        },
        "example_response": {
            "credit_id": "credit_ab12cd34",
            "credit_code": "EC-101",
            "user_id": "user_ab12cd34",
            "credit_type": "solar",
            "price": 120.5,
            "created_at": "2026-07-07T05:30:00+00:00",
        },
    },
    {
        "name": "list_credits",
        "purpose": "List all credit listings.",
        "when_to_use": [
            "Question asks all listed credits",
            "Need set of credits for comparisons",
        ],
        "payload_schema": {},
        "response_schema": {
            "credits": "list<object>",
        },
        "example_request": {},
        "example_response": {
            "credits": [
                {
                    "credit_id": "credit_ab12cd34",
                    "credit_code": "EC-101",
                    "user_id": "user_ab12cd34",
                    "credit_type": "solar",
                    "price": 120.5,
                    "created_at": "2026-07-07T05:30:00+00:00",
                }
            ]
        },
    },
    {
        "name": "get_credit_details",
        "purpose": "Fetch live ownership and metadata for a specific credit.",
        "when_to_use": [
            "Question asks who owns a credit",
            "Question asks when a credit was created",
            "Question asks credit price/type",
        ],
        "payload_schema": {
            "credit_reference": "string (EC-101 or internal credit_id like credit_ab12cd34)",
        },
        "response_schema": {
            "credit_reference": "string",
            "owner_user_id": "string",
            "credit_type": "string",
            "credit_price": "float",
            "credit_created_at": "datetime",
        },
        "example_request": {"credit_reference": "EC-101"},
        "example_response": {
            "credit_reference": "EC-101",
            "owner_user_id": "user_1234abcd",
            "credit_type": "solar",
            "credit_price": 120.5,
            "credit_created_at": "2026-07-05T10:15:20+00:00",
        },
    },
    {
        "name": "list_credit_audit",
        "purpose": "List all credit create/transfer operations.",
        "when_to_use": [
            "Question asks audit history",
            "Need transfer trail for credits",
        ],
        "payload_schema": {
            "limit": "optional int",
        },
        "response_schema": {
            "audit_records": "list<object>",
        },
        "example_request": {"limit": 10},
        "example_response": {
            "audit_records": [
                {
                    "event_id": "evt_ab12cd34",
                    "operation": "transfer",
                    "credit_id": "credit_ab12cd34",
                    "source_user_id": "user_1",
                    "destination_user_id": "user_2",
                    "created_at": "2026-07-07T05:31:00+00:00",
                    "details": "Transferred ownership from user_1 to user_2",
                }
            ]
        },
    },
    {
        "name": "transfer_credit",
        "purpose": "Transfer a credit from source user to destination user.",
        "when_to_use": [
            "Question asks to transfer a credit",
            "Ownership change operation is requested",
        ],
        "payload_schema": {
            "credit_id": "string",
            "source_user_id": "string",
            "destination_user_id": "string",
        },
        "response_schema": {
            "credit_id": "string",
            "credit_code": "string",
            "user_id": "string (new owner)",
            "credit_type": "string",
            "price": "float",
            "created_at": "datetime",
        },
        "example_request": {
            "credit_id": "credit_ab12cd34",
            "source_user_id": "user_1",
            "destination_user_id": "user_2",
        },
        "example_response": {
            "credit_id": "credit_ab12cd34",
            "credit_code": "EC-101",
            "user_id": "user_2",
            "credit_type": "solar",
            "price": 120.5,
            "created_at": "2026-07-07T05:30:00+00:00",
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
