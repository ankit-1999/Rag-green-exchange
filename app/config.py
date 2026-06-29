import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Application
    APP_NAME: str = os.getenv("APP_NAME", "GreenGrid Exchange RAG AI")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # AWS
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # Amazon Bedrock
    BEDROCK_EMBEDDING_MODEL_ID: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
    )
    # Titan Text Embeddings V2 outputs 1024 dimensions by default
    BEDROCK_EMBEDDING_DIMENSION: int = int(
        os.getenv("BEDROCK_EMBEDDING_DIMENSION", "1024")
    )
    BEDROCK_LLM_MODEL_ID: str = os.getenv(
        "BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
    )
    BEDROCK_LLM_MAX_TOKENS: int = int(os.getenv("BEDROCK_LLM_MAX_TOKENS", "1024"))
    BEDROCK_LLM_TEMPERATURE: float = float(
        os.getenv("BEDROCK_LLM_TEMPERATURE", "0.0")
    )

    # Amazon OpenSearch Service
    # Format: https://<domain-endpoint>  (no trailing slash)
    OPENSEARCH_ENDPOINT: str = os.getenv(
        "OPENSEARCH_ENDPOINT", "https://localhost:9200"
    )
    OPENSEARCH_INDEX_NAME: str = os.getenv(
        "OPENSEARCH_INDEX_NAME", "greengrid-docs"
    )
    OPENSEARCH_TOP_K: int = int(os.getenv("OPENSEARCH_TOP_K", "5"))
    # Set to true when using AWS managed OpenSearch with IAM SigV4 auth
    OPENSEARCH_USE_AWS_AUTH: bool = (
        os.getenv("OPENSEARCH_USE_AWS_AUTH", "true").lower() == "true"
    )

    # Amazon S3
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "greengrid-documents")

    # Chunking
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))


settings = Settings()
