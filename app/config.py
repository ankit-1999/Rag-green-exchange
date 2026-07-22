"""
Application configuration for the Green Marketplace RAG AI service.

All values are loaded from environment variables with validated defaults.
Marketplace API paths reference public read-only endpoints only.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    return os.getenv(name, str(default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _int(
    name: str,
    default: int,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Read and validate an integer environment variable."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer."
        ) from exc

    if value < minimum:
        raise ValueError(
            f"Environment variable {name} must be >= {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"Environment variable {name} must be <= {maximum}."
        )

    return value


def _get_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read and validate a floating-point environment variable."""
    raw_value = os.getenv(name, str(default))

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be numeric."
        ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"Environment variable {name} must be >= {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"Environment variable {name} must be <= {maximum}."
        )

    return value


def _normalize_base_url(value: str, variable_name: str) -> str:
    """
    Validate and normalize an HTTP or HTTPS base URL.

    The returned URL never ends with a slash, allowing endpoint paths to be
    joined consistently.
    """
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"{variable_name} must use http or https."
        )

    if not parsed.netloc:
        raise ValueError(
            f"{variable_name} must contain a valid host."
        )

    return normalized


def _path(name: str, default: str) -> str:
    """Read and normalize an API path so it always starts with a slash."""
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"Environment variable {name} cannot be empty.")
    return value if value.startswith("/") else f"/{value}"


class Settings:
    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    APP_NAME = os.getenv(
        "APP_NAME",
        "Green Marketplace RAG AI",
    )
    APP_VERSION = os.getenv(
        "APP_VERSION",
        "2.1.0",
    )
    DEBUG = _bool("DEBUG", False)
    REFERRER_POLICY = os.getenv(
        "REFERRER_POLICY",
        "no-referrer",
    ).strip()

    # -----------------------------------------------------------------------
    # AWS and Amazon Bedrock
    # -----------------------------------------------------------------------

    AWS_REGION = os.getenv(
        "AWS_REGION",
        "us-east-1",
    )
    BEDROCK_EMBEDDING_MODEL_ID = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID",
        "amazon.titan-embed-text-v2:0",
    )
    BEDROCK_EMBEDDING_DIMENSION = _int(
        "BEDROCK_EMBEDDING_DIMENSION",
        1024,
        256,
        1024,
    )
    BEDROCK_LLM_MODEL_ID = os.getenv(
        "BEDROCK_LLM_MODEL_ID",
        "amazon.nova-micro-v1:0",
    )
    BEDROCK_LLM_MAX_TOKENS = _int(
        "BEDROCK_LLM_MAX_TOKENS",
        3200,
        128,
        4096,
    )
    BEDROCK_LLM_TEMPERATURE = _get_float(
        "BEDROCK_LLM_TEMPERATURE",
        0.0,
        0.0,
        1.0,
    )
    BEDROCK_PLANNER_MAX_TOKENS = _int(
        "BEDROCK_PLANNER_MAX_TOKENS",
        700,
        200,
        2048,
    )
    BEDROCK_PLANNER_TEMPERATURE = _get_float(
        "BEDROCK_PLANNER_TEMPERATURE",
        0.0,
        0.0,
        1.0,
    )

    # -----------------------------------------------------------------------
    # OpenSearch and document ingestion
    # -----------------------------------------------------------------------

    OPENSEARCH_ENDPOINT = _normalize_base_url(
        os.getenv(
            "OPENSEARCH_ENDPOINT",
            "https://localhost:9200",
        ),
        "OPENSEARCH_ENDPOINT",
    )
    OPENSEARCH_INDEX_NAME = os.getenv(
        "OPENSEARCH_INDEX_NAME",
        "greengrid-docs",
    )
    OPENSEARCH_TOP_K = _int(
        "OPENSEARCH_TOP_K",
        8,
        1,
        20,
    )
    OPENSEARCH_USE_AWS_AUTH = _bool(
        "OPENSEARCH_USE_AWS_AUTH",
        True,
    )
    S3_BUCKET_NAME = os.getenv(
        "S3_BUCKET_NAME",
        "greengrid-documents",
    )
    CHUNK_SIZE_TOKENS = _int(
        "CHUNK_SIZE_TOKENS",
        500,
        100,
        2000,
    )
    CHUNK_OVERLAP_TOKENS = _int(
        "CHUNK_OVERLAP_TOKENS",
        100,
        0,
        500,
    )

    # -----------------------------------------------------------------------
    # Public marketplace API
    # -----------------------------------------------------------------------

    MARKETPLACE_API_BASE_URL = _normalize_base_url(
        os.getenv(
            "MARKETPLACE_API_BASE_URL",
            "https://greengridenergyexchange.onrender.com",
        ),
        "MARKETPLACE_API_BASE_URL",
    )
    MARKETPLACE_API_CONNECT_TIMEOUT_SECONDS = _get_float(
        "MARKETPLACE_API_CONNECT_TIMEOUT_SECONDS",
        10.0,
        1.0,
        60.0,
    )
    MARKETPLACE_API_READ_TIMEOUT_SECONDS = _get_float(
        "MARKETPLACE_API_READ_TIMEOUT_SECONDS",
        45.0,
        5.0,
        120.0,
    )
    MARKETPLACE_API_TIMEOUT_SECONDS = _get_float(
        "MARKETPLACE_API_TIMEOUT_SECONDS",
        60.0,
        5.0,
        180.0,
    )
    MARKETPLACE_API_PAGE_SIZE = _int(
        "MARKETPLACE_API_PAGE_SIZE",
        200,
        1,
        200,
    )
    MARKETPLACE_API_MAX_PAGES = _int(
        "MARKETPLACE_API_MAX_PAGES",
        100,
        1,
        500,
    )
    MARKETPLACE_API_MAX_RETRIES = _int(
        "MARKETPLACE_API_MAX_RETRIES",
        2,
        0,
        5,
    )
    MARKETPLACE_API_VERIFY_SSL = _bool(
        "MARKETPLACE_API_VERIFY_SSL",
        True,
    )
    MARKETPLACE_API_USER_AGENT = os.getenv(
        "MARKETPLACE_API_USER_AGENT",
        "GreenGrid-RAG-AI/2.1",
    )

    MARKETPLACE_ALL_LISTINGS_PATH = _path(
        "MARKETPLACE_ALL_LISTINGS_PATH",
        "/api/v1/public/listings",
    )
    MARKETPLACE_ACTIVE_LISTINGS_PATH = _path(
        "MARKETPLACE_ACTIVE_LISTINGS_PATH",
        "/api/v1/public/listings/active",
    )
    MARKETPLACE_ALL_PURCHASES_PATH = _path(
        "MARKETPLACE_ALL_PURCHASES_PATH",
        "/api/v1/public/purchases",
    )

    # -----------------------------------------------------------------------
    # Deterministic analytics and prediction thresholds
    # -----------------------------------------------------------------------

    ANALYTICS_MIN_PURCHASE_RECORDS = _int(
        "ANALYTICS_MIN_PURCHASE_RECORDS",
        30,
        1,
    )
    ANALYTICS_MIN_HISTORY_PERIODS = _int(
        "ANALYTICS_MIN_HISTORY_PERIODS",
        6,
        2,
    )
    ANALYTICS_DEFAULT_HISTORY_DAYS = _int(
        "ANALYTICS_DEFAULT_HISTORY_DAYS",
        180,
        28,
        1095,
    )
    ANALYTICS_LLM_SAMPLE_RECORDS = _int(
        "ANALYTICS_LLM_SAMPLE_RECORDS",
        3,
        0,
        10,
    )
    ANALYTICS_RECOMMENDATION_TIE_THRESHOLD = _get_float(
        "ANALYTICS_RECOMMENDATION_TIE_THRESHOLD",
        0.05,
        0.0,
        1.0,
    )

    # -----------------------------------------------------------------------
    # Green Marketplace marketplace enums
    # -----------------------------------------------------------------------

    SUPPORTED_ENERGY_SOURCES = (
        "SOLAR",
        "WIND",
        "HYDRO",
        "BIOMASS",
        "GEOTHERMAL",
        "TIDAL",
        "OTHER",
    )

    SUPPORTED_LISTING_STATUSES = (
        "ACTIVE",
        "SOLD",
        "EXPIRED",
        "CANCELLED",
    )

    SUPPORTED_PURCHASE_STATUSES = (
        "ACTIVE",
        "PENDING",
        "COMPLETED",
        "CONSUMED",
        "CANCELLED",
        "FAILED",
    )


settings = Settings()
