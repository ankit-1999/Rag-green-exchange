"""
marketplace_api_service.py
--------------------------

Read-only HTTP client for the GreenGrid marketplace backend.

Responsibilities:
- Map approved LLM tool names to public GET endpoints.
- Validate and forward supported query filters.
- Retrieve all pages when complete marketplace data is required.
- Handle plain-list and wrapped-list API response formats.
- Deduplicate records by stable ID.
- Normalize key marketplace fields used by analytics.
- Return structured execution metadata to the RAG orchestrator.

The service never calls POST, PATCH, DELETE, authenticated, or blockchain APIs.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx

from app.config import settings
from app.services import tool_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported read-only tools and endpoint paths
# ---------------------------------------------------------------------------

_TOOL_ENDPOINTS: Dict[str, str] = {
    "get_all_listings": settings.MARKETPLACE_ALL_LISTINGS_PATH,
    "get_active_listings": settings.MARKETPLACE_ACTIVE_LISTINGS_PATH,
    "get_all_purchases": settings.MARKETPLACE_ALL_PURCHASES_PATH,
}

_TOOL_COLLECTION_KEYS: Dict[str, Tuple[str, ...]] = {
    "get_all_listings": ("listings", "items", "results", "data"),
    "get_active_listings": ("listings", "items", "results", "data"),
    "get_all_purchases": ("purchases", "items", "results", "data"),
}

_TOOL_ALLOWED_FILTERS: Dict[str, set[str]] = {
    "get_all_listings": {
        "energy_source",
        "location",
        "status",
        "created_from",
        "created_to",
        "skip",
        "limit",
    },
    "get_active_listings": {
        "energy_source",
        "location",
        "min_price_per_kwh",
        "max_price_per_kwh",
        "min_energy_kwh",
        "sort_by",
        "sort_order",
        "skip",
        "limit",
    },
    "get_all_purchases": {
        "energy_source",
        "location",
        "status",
        "completed_from",
        "completed_to",
        "skip",
        "limit",
    },
}

_SUPPORTED_ENERGY_SOURCES = {"SOLAR", "WIND", "HYDRO"}
_SUPPORTED_LISTING_STATUSES = {"active", "sold", "expired", "cancelled"}
_SUPPORTED_PURCHASE_STATUSES = {
    "active",
    "pending",
    "completed",
    "consumed",
    "cancelled",
    "failed",
}
_SUPPORTED_SORT_FIELDS = {
    "price_per_kwh",
    "energy_kwh",
    "created_at",
    "expires_at",
}
_SUPPORTED_SORT_ORDERS = {"asc", "desc"}

_ENERGY_SOURCE_ALIASES = {
    "solar": "SOLAR",
    "solar_energy": "SOLAR",
    "solar power": "SOLAR",
    "wind": "WIND",
    "wind_energy": "WIND",
    "wind power": "WIND",
    "hydro": "HYDRO",
    "hydropower": "HYDRO",
    "hydro_energy": "HYDRO",
    "hydro power": "HYDRO",
    # Backward compatibility for older data and planner output.
    "small_hydro": "HYDRO",
    "small hydro": "HYDRO",
    "small-hydro": "HYDRO",
}


class MarketplaceApiError(RuntimeError):
    """Base error raised for marketplace API failures."""


class MarketplaceApiResponseError(MarketplaceApiError):
    """Raised when a successful HTTP response has an unsupported shape."""


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


def execute_tool_call(tool_name: str, arguments: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute one approved public marketplace GET tool.

    Parameters
    ----------
    tool_name:
        Registered read-only marketplace tool name.
    arguments:
        Planner-generated filters. Unsupported filters are removed.

    Returns
    -------
    Dict[str, Any]
        Structured result compatible with QueryToolResult:
        {
          "tool": str,
          "data": {
            "records": list[dict],
            "sample_records": list[dict],
            "response_metadata": dict
          },
          "arguments": dict,
          "record_count": int,
          "pages_fetched": int,
          "endpoint": str,
          "execution_status": "success"|"partial"|"empty"|"failed",
          "error": str|null
        }
    """
    started_at = time.monotonic()
    endpoint = _get_endpoint(tool_name)
    validated_arguments = _validate_arguments(tool_name, arguments or {})

    try:
        records, pages_fetched, partial, warnings = _fetch_all_pages(
            tool_name=tool_name,
            endpoint=endpoint,
            arguments=validated_arguments,
        )
        normalized_records = _normalize_records(tool_name, records)
        unique_records = _deduplicate_records(normalized_records)

        if partial:
            execution_status = "partial"
        elif not unique_records:
            execution_status = "empty"
        else:
            execution_status = "success"

        duration_ms = round((time.monotonic() - started_at) * 1000, 2)
        sample_size = settings.ANALYTICS_LLM_SAMPLE_RECORDS

        return {
            "tool": tool_name,
            "data": {
                "records": unique_records,
                "sample_records": unique_records[:sample_size],
                "response_metadata": {
                    "warnings": warnings,
                    "duration_ms": duration_ms,
                    "base_url": settings.MARKETPLACE_API_BASE_URL,
                },
            },
            "arguments": validated_arguments,
            "record_count": len(unique_records),
            "pages_fetched": pages_fetched,
            "endpoint": endpoint,
            "execution_status": execution_status,
            "error": "; ".join(warnings) if partial and warnings else None,
        }

    except Exception as exc:
        duration_ms = round((time.monotonic() - started_at) * 1000, 2)
        logger.exception("Marketplace tool execution failed: tool=%s", tool_name)
        return {
            "tool": tool_name,
            "data": {
                "records": [],
                "sample_records": [],
                "response_metadata": {
                    "warnings": [],
                    "duration_ms": duration_ms,
                    "base_url": settings.MARKETPLACE_API_BASE_URL,
                },
            },
            "arguments": validated_arguments,
            "record_count": 0,
            "pages_fetched": 0,
            "endpoint": endpoint,
            "execution_status": "failed",
            "error": _safe_error_message(exc),
        }


def get_records(tool_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract normalized records from an execute_tool_call result."""
    data = tool_result.get("data", {})
    if not isinstance(data, Mapping):
        return []

    records = data.get("records", [])
    if not isinstance(records, list):
        return []

    return [record for record in records if isinstance(record, dict)]


# ---------------------------------------------------------------------------
# Tool and input validation
# ---------------------------------------------------------------------------


def _get_endpoint(tool_name: str) -> str:
    """Return the configured path for an approved marketplace tool."""
    if tool_registry.get_tool_by_name(tool_name) is None:
        raise MarketplaceApiError(f"Tool is not registered: {tool_name}")

    endpoint = _TOOL_ENDPOINTS.get(tool_name)
    if endpoint is None:
        raise MarketplaceApiError(
            f"Tool has no marketplace GET executor: {tool_name}"
        )

    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    return endpoint


def _validate_arguments(tool_name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate planner-generated API filters using an explicit allowlist."""
    allowed_filters = _TOOL_ALLOWED_FILTERS.get(tool_name)
    if allowed_filters is None:
        raise MarketplaceApiError(f"Unsupported marketplace tool: {tool_name}")

    cleaned: Dict[str, Any] = {}

    for key, value in arguments.items():
        if key not in allowed_filters or value is None or value == "":
            continue

        if key == "energy_source":
            normalized_source = _normalize_energy_source(value)
            if normalized_source:
                cleaned[key] = normalized_source

        elif key == "location":
            normalized_location = str(value).strip()
            if normalized_location:
                cleaned[key] = normalized_location[:200]

        elif key == "status":
            normalized_status = str(value).strip().lower()
            allowed_statuses = (
                _SUPPORTED_PURCHASE_STATUSES
                if tool_name == "get_all_purchases"
                else _SUPPORTED_LISTING_STATUSES
            )
            if normalized_status in allowed_statuses:
                cleaned[key] = normalized_status

        elif key in {
            "created_from",
            "created_to",
            "completed_from",
            "completed_to",
        }:
            normalized_date = _normalize_iso_date_or_datetime(value)
            if normalized_date:
                cleaned[key] = normalized_date

        elif key in {"min_price_per_kwh", "max_price_per_kwh"}:
            positive_decimal = _positive_decimal(value)
            if positive_decimal is not None:
                cleaned[key] = positive_decimal

        elif key == "min_energy_kwh":
            positive_int = _positive_int(value)
            if positive_int is not None:
                cleaned[key] = positive_int

        elif key == "sort_by":
            normalized_sort = str(value).strip().lower()
            if normalized_sort in _SUPPORTED_SORT_FIELDS:
                cleaned[key] = normalized_sort

        elif key == "sort_order":
            normalized_order = str(value).strip().lower()
            if normalized_order in _SUPPORTED_SORT_ORDERS:
                cleaned[key] = normalized_order

        elif key == "skip":
            non_negative_int = _non_negative_int(value)
            if non_negative_int is not None:
                cleaned[key] = non_negative_int

        elif key == "limit":
            positive_int = _positive_int(value)
            if positive_int is not None:
                cleaned[key] = min(positive_int, settings.MARKETPLACE_API_PAGE_SIZE)

    cleaned.setdefault("skip", 0)
    cleaned.setdefault("limit", settings.MARKETPLACE_API_PAGE_SIZE)

    if tool_name == "get_all_purchases":
        cleaned.setdefault("status", "completed")

    return cleaned


def _normalize_energy_source(value: Any) -> Optional[str]:
    """Normalize energy source to SOLAR, WIND, or HYDRO."""
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    normalized_key = raw.lower().replace("-", "_")
    normalized_key = " ".join(normalized_key.split())

    direct = raw.upper().replace("-", "_").replace(" ", "_")
    if direct in _SUPPORTED_ENERGY_SOURCES:
        return direct

    return _ENERGY_SOURCE_ALIASES.get(normalized_key) or _ENERGY_SOURCE_ALIASES.get(raw.lower())


def _normalize_iso_date_or_datetime(value: Any) -> Optional[str]:
    """Accept an ISO date or ISO datetime and return its original normalized text."""
    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return None

    try:
        date.fromisoformat(candidate)
        return candidate
    except ValueError:
        pass

    try:
        datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        return candidate
    except ValueError:
        return None


def _positive_decimal(value: Any) -> Optional[str]:
    """Return a normalized positive decimal string."""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None

    if number <= 0:
        return None

    return format(number, "f")


def _positive_int(value: Any) -> Optional[int]:
    """Return a positive integer or None."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _non_negative_int(value: Any) -> Optional[int]:
    """Return a non-negative integer or None."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


# ---------------------------------------------------------------------------
# HTTP and pagination
# ---------------------------------------------------------------------------


def _fetch_all_pages(
    tool_name: str,
    endpoint: str,
    arguments: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], int, bool, List[str]]:
    """Retrieve pages until the final page or configured safety limit."""
    base_skip = int(arguments.get("skip", 0))
    page_size = min(
        int(arguments.get("limit", settings.MARKETPLACE_API_PAGE_SIZE)),
        settings.MARKETPLACE_API_PAGE_SIZE,
    )

    base_params = {
        key: value
        for key, value in arguments.items()
        if key not in {"skip", "limit"}
    }

    all_records: List[Dict[str, Any]] = []
    pages_fetched = 0
    partial = False
    warnings: List[str] = []

    timeout = httpx.Timeout(
        timeout=settings.MARKETPLACE_API_TIMEOUT_SECONDS,
        connect=settings.MARKETPLACE_API_CONNECT_TIMEOUT_SECONDS,
        read=settings.MARKETPLACE_API_READ_TIMEOUT_SECONDS,
    )

    headers = {
        "Accept": "application/json",
        "User-Agent": settings.MARKETPLACE_API_USER_AGENT,
    }

    transport = httpx.HTTPTransport(
        retries=settings.MARKETPLACE_API_MAX_RETRIES,
        verify=settings.MARKETPLACE_API_VERIFY_SSL,
    )

    with httpx.Client(
        base_url=settings.MARKETPLACE_API_BASE_URL,
        timeout=timeout,
        headers=headers,
        transport=transport,
        follow_redirects=True,
    ) as client:
        for page_index in range(settings.MARKETPLACE_API_MAX_PAGES):
            params = dict(base_params)
            params["skip"] = base_skip + (page_index * page_size)
            params["limit"] = page_size

            try:
                payload = _request_json(
                    client=client,
                    endpoint=endpoint,
                    params=params,
                )
            except MarketplaceApiError as exc:
                if pages_fetched == 0:
                    raise
                partial = True
                warnings.append(
                    f"Pagination stopped after {pages_fetched} page(s): {_safe_error_message(exc)}"
                )
                break

            page_records, pagination_metadata = _extract_collection(
                tool_name=tool_name,
                payload=payload,
            )
            pages_fetched += 1
            all_records.extend(page_records)

            if _is_last_page(
                page_records=page_records,
                page_size=page_size,
                metadata=pagination_metadata,
                current_skip=params["skip"],
            ):
                break
        else:
            partial = True
            warnings.append(
                "Pagination reached MARKETPLACE_API_MAX_PAGES before a final page was detected."
            )

    return all_records, pages_fetched, partial, warnings


def _request_json(
    client: httpx.Client,
    endpoint: str,
    params: Mapping[str, Any],
) -> Any:
    """Execute one GET request and return parsed JSON."""
    url = f"{settings.MARKETPLACE_API_BASE_URL}{endpoint}"
    logger.info("Marketplace GET: url=%s params=%s", url, dict(params))

    try:
        response = client.get(endpoint, params=params)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise MarketplaceApiError(
            f"Marketplace API timed out for {endpoint}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        response_preview = exc.response.text[:300].replace("\n", " ")
        raise MarketplaceApiError(
            f"Marketplace API returned HTTP {status_code} for {endpoint}: {response_preview}"
        ) from exc
    except httpx.HTTPError as exc:
        raise MarketplaceApiError(
            f"Marketplace API request failed for {endpoint}: {exc}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        preview = response.text[:300].replace("\n", " ")
        raise MarketplaceApiResponseError(
            f"Marketplace API returned invalid JSON for {endpoint}: {preview}"
        ) from exc


def _extract_collection(
    tool_name: str,
    payload: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract a record array from plain or wrapped API response formats."""
    if isinstance(payload, list):
        return _dict_records(payload), {}

    if not isinstance(payload, dict):
        raise MarketplaceApiResponseError(
            f"Expected list or object response for {tool_name}; received {type(payload).__name__}."
        )

    for key in _TOOL_COLLECTION_KEYS[tool_name]:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            metadata = {
                key_name: value
                for key_name, value in payload.items()
                if key_name != key
            }
            return _dict_records(candidate), metadata

    # Some APIs wrap the collection one level deeper under data.
    nested_data = payload.get("data")
    if isinstance(nested_data, dict):
        for key in _TOOL_COLLECTION_KEYS[tool_name]:
            candidate = nested_data.get(key)
            if isinstance(candidate, list):
                metadata = {
                    key_name: value
                    for key_name, value in nested_data.items()
                    if key_name != key
                }
                return _dict_records(candidate), metadata

    raise MarketplaceApiResponseError(
        f"Could not find a record collection in the {tool_name} response."
    )


def _dict_records(values: Sequence[Any]) -> List[Dict[str, Any]]:
    """Keep dictionary records and ignore malformed array entries."""
    records = [dict(value) for value in values if isinstance(value, Mapping)]
    if len(records) != len(values):
        logger.warning(
            "Ignored %d non-object records in marketplace response.",
            len(values) - len(records),
        )
    return records


def _is_last_page(
    page_records: Sequence[Mapping[str, Any]],
    page_size: int,
    metadata: Mapping[str, Any],
    current_skip: int,
) -> bool:
    """Determine whether the current response is the final page."""
    if not page_records:
        return True

    if len(page_records) < page_size:
        return True

    has_more = metadata.get("has_more")
    if isinstance(has_more, bool):
        return not has_more

    next_value = metadata.get("next") or metadata.get("next_cursor")
    if "next" in metadata or "next_cursor" in metadata:
        return next_value in {None, "", False}

    total = metadata.get("total") or metadata.get("total_count")
    if isinstance(total, int):
        return current_skip + len(page_records) >= total

    return False


# ---------------------------------------------------------------------------
# Record normalization and deduplication
# ---------------------------------------------------------------------------


def _normalize_records(tool_name: str, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize fields needed by analytics while preserving backend fields."""
    normalized: List[Dict[str, Any]] = []

    for raw_record in records:
        record = dict(raw_record)

        source_value = (
            record.get("energy_source")
            or record.get("source")
            or record.get("credit_type")
        )
        normalized_source = _normalize_energy_source(source_value)
        if normalized_source:
            record["energy_source"] = normalized_source

        if "energy_kwh" not in record:
            quantity = record.get("quantity_kwh") or record.get("quantity")
            if quantity is not None:
                record["energy_kwh"] = quantity

        if "price_per_kwh" not in record:
            price = record.get("price") or record.get("unit_price")
            if price is not None:
                record["price_per_kwh"] = price

        if "credit_reference" not in record:
            credit_reference = (
                record.get("credit_id")
                or record.get("reference")
                or record.get("credit_ref")
            )
            if credit_reference is not None:
                record["credit_reference"] = credit_reference

        if tool_name == "get_all_purchases" and "location" not in record:
            listing_location = record.get("listing_location")
            if listing_location is not None:
                record["location"] = listing_location

        normalized.append(record)

    return normalized


def _deduplicate_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate records by stable ID while preserving original order."""
    unique: List[Dict[str, Any]] = []
    seen = set()

    for index, raw_record in enumerate(records):
        record = dict(raw_record)
        record_id = (
            record.get("id")
            or record.get("purchase_id")
            or record.get("listing_id")
            or record.get("credit_reference")
        )

        if record_id is None:
            # Preserve records without stable IDs rather than discarding data.
            fingerprint = ("missing_id", index, repr(sorted(record.items())))
        else:
            fingerprint = ("stable_id", str(record_id))

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        unique.append(record)

    return unique


def _safe_error_message(exc: Exception) -> str:
    """Return a bounded error message safe for logs and API summaries."""
    message = str(exc).strip() or type(exc).__name__
    return message[:500]
