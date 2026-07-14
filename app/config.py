"""
config.py
---------

Central application configuration loaded from environment variables.

The AI/RAG service and GreenGrid marketplace backend run in separate
environments. The AI service accesses only approved public read-only GET APIs
from the marketplace backend.

No static AWS credentials or marketplace secrets are stored here.
"""

import os
from typing import Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv # type: ignore

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable safely."""
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _get_int(
    name: str,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Read and validate an integer environment variable."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer."
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


def _normalize_base_url(value: str) -> str:
    """
    Validate and normalize the marketplace backend base URL.

    The returned URL never ends with a slash so endpoint paths can be joined
    consistently:

        f"{settings.MARKETPLACE_API_BASE_URL}/api/v1/public/listings"
    """
    normalized = value.strip().rstrip("/")

    parsed = urlparse(normalized)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "MARKETPLACE_API_BASE_URL must use http or https."
        )

    if not parsed.netloc:
        raise ValueError(
            "MARKETPLACE_API_BASE_URL must contain a valid host."
        )

    return normalized


class Settings:
    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    APP_NAME: str = os.getenv(
        "APP_NAME",
        "GreenGrid Exchange RAG AI",
    )

    APP_VERSION: str = os.getenv(
        "APP_VERSION",
        "2.0.0",
    )

    DEBUG: bool = _get_bool(
        "DEBUG",
        False,
    )

    # -----------------------------------------------------------------------
    # AWS
    # -----------------------------------------------------------------------

    AWS_REGION: str = os.getenv(
        "AWS_REGION",
        "us-east-1",
    )

    # -----------------------------------------------------------------------
    # Amazon Bedrock
    # -----------------------------------------------------------------------

    BEDROCK_EMBEDDING_MODEL_ID: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID",
        "amazon.titan-embed-text-v2:0",
    )

    BEDROCK_EMBEDDING_DIMENSION: int = _get_int(
        "BEDROCK_EMBEDDING_DIMENSION",
        1024,
        minimum=256,
        maximum=1024,
    )

    BEDROCK_LLM_MODEL_ID: str = os.getenv(
        "BEDROCK_LLM_MODEL_ID",
        "amazon.nova-micro-v1:0",
    )

    BEDROCK_LLM_MAX_TOKENS: int = _get_int(
        "BEDROCK_LLM_MAX_TOKENS",
        1024,
        minimum=128,
        maximum=4096,
    )

    BEDROCK_LLM_TEMPERATURE: float = _get_float(
        "BEDROCK_LLM_TEMPERATURE",
        0.0,
        minimum=0.0,
        maximum=1.0,
    )

    # Separate token limit for the tool planner.
    # The planner returns structured JSON and does not need the full answer
    # token allowance.
    BEDROCK_PLANNER_MAX_TOKENS: int = _get_int(
        "BEDROCK_PLANNER_MAX_TOKENS",
        900,
        minimum=200,
        maximum=2048,
    )

    BEDROCK_PLANNER_TEMPERATURE: float = _get_float(
        "BEDROCK_PLANNER_TEMPERATURE",
        0.0,
        minimum=0.0,
        maximum=1.0,
    )

    # -----------------------------------------------------------------------
    # Amazon OpenSearch Service
    # -----------------------------------------------------------------------

    # Format:
    # https://<domain-endpoint>
    #
    # Do not include a trailing slash.
    OPENSEARCH_ENDPOINT: str = os.getenv(
        "OPENSEARCH_ENDPOINT",
        "https://localhost:9200",
    ).strip().rstrip("/")

    OPENSEARCH_INDEX_NAME: str = os.getenv(
        "OPENSEARCH_INDEX_NAME",
        "greengrid-docs",
    )

    OPENSEARCH_TOP_K: int = _get_int(
        "OPENSEARCH_TOP_K",
        5,
        minimum=1,
        maximum=20,
    )

    # Enable this when using AWS-managed OpenSearch with IAM SigV4
    # authentication.
    OPENSEARCH_USE_AWS_AUTH: bool = _get_bool(
        "OPENSEARCH_USE_AWS_AUTH",
        True,
    )

    # -----------------------------------------------------------------------
    # Amazon S3
    # -----------------------------------------------------------------------

    S3_BUCKET_NAME: str = os.getenv(
        "S3_BUCKET_NAME",
        "greengrid-documents",
    )

    # -----------------------------------------------------------------------
    # Document Chunking
    # -----------------------------------------------------------------------

    CHUNK_SIZE_TOKENS: int = _get_int(
        "CHUNK_SIZE_TOKENS",
        500,
        minimum=100,
        maximum=2000,
    )

    CHUNK_OVERLAP_TOKENS: int = _get_int(
        "CHUNK_OVERLAP_TOKENS",
        50,
        minimum=0,
        maximum=500,
    )

    # -----------------------------------------------------------------------
    # GreenGrid Marketplace Backend
    # -----------------------------------------------------------------------

    MARKETPLACE_API_BASE_URL: str = _normalize_base_url(
        os.getenv(
            "MARKETPLACE_API_BASE_URL",
            "https://greengridenergyexchange.onrender.com",
        )
    )

    # Connection timeout covers DNS resolution and TCP/TLS connection.
    MARKETPLACE_API_CONNECT_TIMEOUT_SECONDS: float = _get_float(
        "MARKETPLACE_API_CONNECT_TIMEOUT_SECONDS",
        10.0,
        minimum=1.0,
        maximum=60.0,
    )

    # Read timeout allows the Render backend time to wake from a cold start
    # and return a response.
    MARKETPLACE_API_READ_TIMEOUT_SECONDS: float = _get_float(
        "MARKETPLACE_API_READ_TIMEOUT_SECONDS",
        45.0,
        minimum=5.0,
        maximum=120.0,
    )

    # Total request timeout used by the marketplace API client.
    MARKETPLACE_API_TIMEOUT_SECONDS: float = _get_float(
        "MARKETPLACE_API_TIMEOUT_SECONDS",
        60.0,
        minimum=5.0,
        maximum=180.0,
    )

    # Maximum records requested from the backend in one page.
    MARKETPLACE_API_PAGE_SIZE: int = _get_int(
        "MARKETPLACE_API_PAGE_SIZE",
        100,
        minimum=1,
        maximum=100,
    )

    # Safety limit that prevents an unbounded pagination loop.
    MARKETPLACE_API_MAX_PAGES: int = _get_int(
        "MARKETPLACE_API_MAX_PAGES",
        100,
        minimum=1,
        maximum=500,
    )

    # Number of retries for transient connection failures and HTTP 5xx
    # responses. Keep this low to avoid delaying chatbot responses.
    MARKETPLACE_API_MAX_RETRIES: int = _get_int(
        "MARKETPLACE_API_MAX_RETRIES",
        2,
        minimum=0,
        maximum=5,
    )

    # TLS verification should remain enabled in production.
    MARKETPLACE_API_VERIFY_SSL: bool = _get_bool(
        "MARKETPLACE_API_VERIFY_SSL",
        True,
    )

    # User-Agent allows the backend to distinguish calls originating from
    # the GreenGrid AI service.
    MARKETPLACE_API_USER_AGENT: str = os.getenv(
        "MARKETPLACE_API_USER_AGENT",
        "GreenGrid-RAG-AI/2.0",
    )

    # -----------------------------------------------------------------------
    # Public Read-Only Marketplace API Paths
    # -----------------------------------------------------------------------

    MARKETPLACE_ALL_LISTINGS_PATH: str = os.getenv(
        "MARKETPLACE_ALL_LISTINGS_PATH",
        "/api/v1/public/listings",
    )

    MARKETPLACE_ACTIVE_LISTINGS_PATH: str = os.getenv(
        "MARKETPLACE_ACTIVE_LISTINGS_PATH",
        "/api/v1/listings/active",
    )

    MARKETPLACE_ALL_PURCHASES_PATH: str = os.getenv(
        "MARKETPLACE_ALL_PURCHASES_PATH",
        "/api/v1/public/purchases",
    )

    # -----------------------------------------------------------------------
    # Analytics and Prediction Configuration
    # -----------------------------------------------------------------------

    # Minimum number of completed marketplace purchases required before a
    # prediction can be described as reasonably supported.
    ANALYTICS_MIN_PURCHASE_RECORDS: int = _get_int(
        "ANALYTICS_MIN_PURCHASE_RECORDS",
        30,
        minimum=1,
        maximum=10000,
    )

    # Minimum number of historical weekly periods required for forecasting.
    ANALYTICS_MIN_HISTORY_PERIODS: int = _get_int(
        "ANALYTICS_MIN_HISTORY_PERIODS",
        6,
        minimum=2,
        maximum=104,
    )

    # Default history used when the user requests a prediction without
    # specifying a historical period.
    ANALYTICS_DEFAULT_HISTORY_DAYS: int = _get_int(
        "ANALYTICS_DEFAULT_HISTORY_DAYS",
        180,
        minimum=28,
        maximum=1095,
    )

    # Short-term recommendation window.
    ANALYTICS_RECENT_WINDOW_DAYS: int = _get_int(
        "ANALYTICS_RECENT_WINDOW_DAYS",
        7,
        minimum=1,
        maximum=90,
    )

    # Supporting trend window for questions such as:
    # "Should I list Solar or Wind credits this week?"
    ANALYTICS_SUPPORTING_WINDOW_DAYS: int = _get_int(
        "ANALYTICS_SUPPORTING_WINDOW_DAYS",
        28,
        minimum=7,
        maximum=365,
    )

    # Default forecast horizon when the user asks about next week but does
    # not provide an exact number of days.
    ANALYTICS_DEFAULT_WEEKLY_FORECAST_DAYS: int = _get_int(
        "ANALYTICS_DEFAULT_WEEKLY_FORECAST_DAYS",
        7,
        minimum=1,
        maximum=31,
    )

    # Maximum number of raw records that may be included in the final LLM
    # prompt. Full datasets remain available to deterministic analytics code.
    ANALYTICS_LLM_SAMPLE_RECORDS: int = _get_int(
        "ANALYTICS_LLM_SAMPLE_RECORDS",
        5,
        minimum=0,
        maximum=20,
    )

    # If two recommendation scores differ by less than this threshold,
    # the AI should report that there is no strong preference.
    ANALYTICS_RECOMMENDATION_TIE_THRESHOLD: float = _get_float(
        "ANALYTICS_RECOMMENDATION_TIE_THRESHOLD",
        0.05,
        minimum=0.0,
        maximum=1.0,
    )

    # -----------------------------------------------------------------------
    # Supported Marketplace Values
    # -----------------------------------------------------------------------

    SUPPORTED_ENERGY_SOURCES: Tuple[str, ...] = (
        "SOLAR",
        "WIND",
        "HYDRO",
    )

    SUPPORTED_LISTING_STATUSES: Tuple[str, ...] = (
        "active",
        "sold",
        "expired",
        "cancelled",
    )

    # Confirm these values against the actual backend purchase-status enum.
    SUPPORTED_PURCHASE_STATUSES: Tuple[str, ...] = (
        "active",
        "pending",
        "completed",
        "consumed",
        "cancelled",
        "failed",
    )


settings = Settings()