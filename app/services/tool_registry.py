import json
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Central read-only tool catalog
# ---------------------------------------------------------------------------
#
# Used by:
# 1. LLM planner prompt generation
# 2. Backend tool-call validation
# 3. Backend API execution routing
#
# IMPORTANT:
# - The AI planner may only select tools listed here.
# - All marketplace tools are read-only GET operations.
# - The AI must not create, update, cancel, delete, or purchase anything.
# - Predictions and analytics are calculated after API data is returned.
# - Nova Micro selects tools and explains results; it must not invent numbers.
# ---------------------------------------------------------------------------

_TOOL_CATALOG: List[Dict] = [
    # -----------------------------------------------------------------------
    # PUBLIC MARKETPLACE LISTING TOOLS
    # -----------------------------------------------------------------------
    {
        "name": "get_all_listings",
        "http_method": "GET",
        "endpoint": "/api/v1/public/listings",
        "purpose": (
            "Retrieve marketplace listings for historical supply, listing "
            "status, source, location, asking-price, stability, and "
            "demand-to-supply analytics."
        ),
        "when_to_use": [
            "Question asks about listings created during a historical period",
            "Question asks for historical supply",
            "Question asks to compare listed supply by renewable source",
            "Question asks about active, sold, expired, or cancelled listings",
            "Question asks for asking-price history",
            "Question asks which source had the most stable newly listed supply",
            "Question asks for demand-to-supply ratio",
            "Question requests demand, supply, or price prediction",
            "Question asks whether Solar or Wind credits should be listed",
            "Question asks for historical marketplace trends",
        ],
        "do_not_use_when": [
            "Question only asks for credits currently available to buy; "
            "use get_active_listings instead",
            "Question asks only for completed purchase demand; "
            "use get_all_purchases instead",
            "Question is purely conceptual and can be answered from RAG",
        ],
        "payload_schema": {
            "energy_source": (
                "optional string; allowed values: "
                "SOLAR|WIND|SMALL_HYDRO"
            ),
            "location": (
                "optional string; case-insensitive location filter"
            ),
            "status": (
                "optional string; allowed values: "
                "active|sold|expired|cancelled"
            ),
            "created_from": (
                "optional ISO-8601 date or datetime; "
                "inclusive listing-creation lower boundary"
            ),
            "created_to": (
                "optional ISO-8601 date or datetime; "
                "inclusive listing-creation upper boundary"
            ),
            "skip": "optional int >= 0; default 0",
            "limit": "optional int from 1 to 100; default 100",
        },
        "response_schema": {
            "listings": [
                {
                    "id": "string UUID",
                    "credit_reference": "string; example EC-105",
                    "seller_id": "string UUID or anonymized seller ID",
                    "energy_source": (
                        "string: SOLAR|WIND|SMALL_HYDRO"
                    ),
                    "energy_kwh": "int",
                    "price_per_kwh": "decimal string",
                    "total_price": "decimal string or number",
                    "title": "string",
                    "description": "string|null",
                    "location": "string|null",
                    "status": (
                        "string: active|sold|expired|cancelled"
                    ),
                    "is_available": "bool",
                    "created_at": "ISO-8601 datetime",
                    "updated_at": "ISO-8601 datetime",
                    "expires_at": "ISO-8601 datetime|null",
                }
            ],
        },
        "analytics_supported": [
            "Historical listed supply by source",
            "Historical listed supply by location",
            "Newly listed kWh by period",
            "Listing-status distribution",
            "Historical asking-price analysis",
            "Supply trend by source",
            "Newly listed supply stability",
            "Demand-to-supply ratio denominator",
            "Demand and price prediction features",
            "Solar-versus-Wind listing recommendation features",
        ],
        "example_request": {
            "energy_source": "SOLAR",
            "location": "Noida",
            "status": "active",
            "created_from": "2026-06-01",
            "created_to": "2026-06-30",
            "skip": 0,
            "limit": 100,
        },
        "example_response": {
            "listings": [
                {
                    "id": "e8b82f52-beea-439d-9ebe-1d717c890107",
                    "credit_reference": "EC-105",
                    "seller_id": "seller_01",
                    "energy_source": "SOLAR",
                    "energy_kwh": 100,
                    "price_per_kwh": "0.000500",
                    "total_price": "0.050000",
                    "title": "Solar - 100 kWh Available",
                    "description": "Verified renewable electricity",
                    "location": "Noida, Uttar Pradesh, India",
                    "status": "active",
                    "is_available": True,
                    "created_at": "2026-06-10T10:00:00Z",
                    "updated_at": "2026-06-10T10:00:00Z",
                    "expires_at": "2026-07-10T10:00:00Z",
                }
            ]
        },
        "pagination_rule": (
            "Fetch all pages before claiming total, all, market-wide, "
            "highest, lowest, percentage, ratio, stability, or trend. "
            "Increase skip by limit until fewer than limit records return."
        ),
    },
    {
        "name": "get_active_listings",
        "http_method": "GET",
        "endpoint": "/api/v1/listings/active",
        "purpose": (
            "Retrieve listings that are currently active, available, and "
            "eligible for purchase."
        ),
        "when_to_use": [
            "Question asks which credits are currently available",
            "Question asks about current active supply",
            "Question asks which renewable source has highest available supply",
            "Question asks for active supply percentages by source",
            "Question asks for available credits in a location",
            "Question asks which location has greatest source-level supply",
            "Question asks for cheapest available listings",
            "Question asks for listings within a price range",
            "Question asks for listings with a minimum quantity",
            "Question asks for current marketplace recommendations",
            "Question asks for current supply saturation",
            "Question asks for current inventory used in a shortage forecast",
        ],
        "do_not_use_when": [
            "Question asks for completed purchases or realized demand",
            "Question asks for historical listed supply over a period",
            "Question asks for sold, expired, or cancelled listings",
            "Question is purely conceptual and can be answered from RAG",
        ],
        "payload_schema": {
            "energy_source": (
                "optional string; allowed values: "
                "SOLAR|WIND|SMALL_HYDRO"
            ),
            "location": (
                "optional string; case-insensitive location filter"
            ),
            "min_price_per_kwh": (
                "optional positive decimal"
            ),
            "max_price_per_kwh": (
                "optional positive decimal"
            ),
            "min_energy_kwh": (
                "optional positive int"
            ),
            "sort_by": (
                "optional string; allowed values: "
                "price_per_kwh|energy_kwh|created_at|expires_at"
            ),
            "sort_order": (
                "optional string; allowed values: asc|desc"
            ),
            "skip": "optional int >= 0; default 0",
            "limit": "optional int from 1 to 100; default 100",
        },
        "response_schema": {
            "listings": [
                {
                    "id": "string UUID",
                    "credit_reference": "string; example EC-105",
                    "energy_source": (
                        "string: SOLAR|WIND|SMALL_HYDRO"
                    ),
                    "energy_kwh": "int",
                    "price_per_kwh": "decimal string",
                    "total_price": "decimal string or number",
                    "title": "string",
                    "location": "string|null",
                    "status": "string; expected active",
                    "is_available": "bool; expected true",
                    "created_at": "ISO-8601 datetime",
                    "expires_at": "ISO-8601 datetime|null",
                }
            ],
        },
        "analytics_supported": [
            "Current available supply by source",
            "Current supply mix percentages",
            "Current active listings by location",
            "Lowest-priced available listing",
            "Highest-volume available listing",
            "Active supply saturation",
            "Current inventory for shortage prediction",
            "Buyer recommendation candidates",
        ],
        "example_request": {
            "energy_source": "WIND",
            "location": "Gujarat",
            "max_price_per_kwh": 0.0006,
            "min_energy_kwh": 100,
            "sort_by": "price_per_kwh",
            "sort_order": "asc",
            "skip": 0,
            "limit": 100,
        },
        "example_response": {
            "listings": [
                {
                    "id": "84dbe4ba-64fc-487e-bab3-1f0891f23545",
                    "credit_reference": "EC-118",
                    "energy_source": "WIND",
                    "energy_kwh": 250,
                    "price_per_kwh": "0.000440",
                    "total_price": "0.110000",
                    "title": "Wind - 250 kWh Available",
                    "location": "Gujarat, India",
                    "status": "active",
                    "is_available": True,
                    "created_at": "2026-07-01T10:00:00Z",
                    "expires_at": "2026-08-01T10:00:00Z",
                }
            ]
        },
        "pagination_rule": (
            "Fetch all pages before calculating total active supply, "
            "source percentages, location rankings, or recommendations. "
            "For top-N listing questions, server sorting and a limited "
            "result may be used if the API guarantees complete ordering."
        ),
    },

    # -----------------------------------------------------------------------
    # PUBLIC PURCHASE TOOL
    # -----------------------------------------------------------------------
    {
        "name": "get_all_purchases",
        "http_method": "GET",
        "endpoint": "/api/v1/public/purchases",
        "purpose": (
            "Retrieve marketplace-wide purchase records for realized demand, "
            "realized selling-price, demand trend, price volatility, "
            "forecasting, and recommendations."
        ),
        "when_to_use": [
            "Question asks about demand",
            "Question asks about completed purchase volume",
            "Question asks which source had highest demand",
            "Question asks which source had highest average selling price",
            "Question asks for realized selling-price history",
            "Question asks for demand-to-supply ratio",
            "Question asks about historical demand by location",
            "Question asks about price volatility",
            "Question asks for demand or price prediction",
            "Question asks whether a location may face a future shortage",
            "Question asks whether Solar or Wind credits should be listed",
            "Question asks for a recommendation based on historical demand",
        ],
        "do_not_use_when": [
            "Question asks only for current available listings",
            "Question asks only for static marketplace rules",
            "Question asks only for historical listing supply",
        ],
        "payload_schema": {
            "energy_source": (
                "optional string; allowed values: "
                "SOLAR|WIND|SMALL_HYDRO"
            ),
            "location": (
                "optional string; normalized listing location"
            ),
            "status": (
                "optional string; use completed for realized demand "
                "and realized selling-price analytics"
            ),
            "completed_from": (
                "optional ISO-8601 date or datetime; "
                "inclusive completion lower boundary"
            ),
            "completed_to": (
                "optional ISO-8601 date or datetime; "
                "inclusive completion upper boundary"
            ),
            "skip": "optional int >= 0; default 0",
            "limit": "optional int from 1 to 100; default 100",
        },
        "response_schema": {
            "purchases": [
                {
                    "id": "string UUID",
                    "credit_reference": "string; example EC-105",
                    "listing_id": "string UUID",
                    "energy_source": (
                        "string: SOLAR|WIND|SMALL_HYDRO"
                    ),
                    "location": "string|null",
                    "energy_kwh": "int",
                    "price_per_kwh": "decimal string",
                    "total_price": "decimal string",
                    "status": "string",
                    "created_at": "ISO-8601 datetime",
                    "completed_at": "ISO-8601 datetime|null",
                    "consumed_at": "ISO-8601 datetime|null",
                }
            ],
        },
        "analytics_supported": [
            "Realized demand by source",
            "Realized demand by location",
            "Completed purchase volume",
            "Source demand share",
            "Volume-weighted realized selling price",
            "Historical demand trend",
            "Historical realized-price trend",
            "Demand-to-supply ratio numerator",
            "Price volatility",
            "Demand prediction features",
            "Price prediction features",
            "Shortage prediction features",
            "Seller and buyer recommendation features",
        ],
        "example_request": {
            "energy_source": "SOLAR",
            "location": "Noida",
            "status": "completed",
            "completed_from": "2026-06-01",
            "completed_to": "2026-06-30",
            "skip": 0,
            "limit": 100,
        },
        "example_response": {
            "purchases": [
                {
                    "id": "66c098e1-d5eb-4d21-b2b1-696c4a79a1f4",
                    "credit_reference": "EC-105",
                    "listing_id": "e8b82f52-beea-439d-9ebe-1d717c890107",
                    "energy_source": "SOLAR",
                    "location": "Noida, Uttar Pradesh, India",
                    "energy_kwh": 50,
                    "price_per_kwh": "0.000480",
                    "total_price": "0.024000",
                    "status": "completed",
                    "created_at": "2026-06-15T10:00:00Z",
                    "completed_at": "2026-06-15T10:05:00Z",
                    "consumed_at": None,
                }
            ]
        },
        "pagination_rule": (
            "Fetch all pages before calculating market-wide demand, "
            "average selling price, demand share, volatility, trend, "
            "prediction, or recommendation."
        ),
        "privacy_rule": (
            "The public response must not expose buyer email, seller email, "
            "phone number, full address, wallet address, passwords, tokens, "
            "private credentials, or other sensitive personal information."
        ),
    },
]


def get_tool_catalog() -> List[dict]:
    """Return a shallow copy of the read-only tool catalog."""
    return list(_TOOL_CATALOG)


def get_tool_by_name(name: str) -> Optional[dict]:
    """Return one tool definition by exact tool name."""
    for tool in _TOOL_CATALOG:
        if tool["name"] == name:
            return tool
    return None


def get_allowed_tool_names() -> List[str]:
    return [tool["name"] for tool in _TOOL_CATALOG]


def build_planner_tools_text() -> str:
    """Render tool catalog as compact JSON for planner prompt context."""
    return json.dumps(_TOOL_CATALOG, ensure_ascii=True, indent=2)
