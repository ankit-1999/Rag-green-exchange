"""
tool_registry.py
----------------

Central read-only tool catalog for the Green Marketplace AI service.

Used by:
1. LLM planner prompt generation
2. Backend tool-call validation
3. Marketplace API execution routing

Important constraints:
- The AI planner may select only tools registered in this file.
- Every marketplace tool is a public read-only GET operation.
- The AI must never create, update, cancel, delete, or purchase anything.
- Analytics and predictions are calculated only after live API data is returned.
- Nova Micro may select tools and explain deterministic results, but it must not
  invent marketplace values.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


SUPPORTED_ENERGY_SOURCES: tuple[str, ...] = (
    "SOLAR",
    "WIND",
    "HYDRO",
    "BIOMASS",
    "GEOTHERMAL",
    "TIDAL",
    "OTHER",
)

SUPPORTED_ENERGY_SOURCES_TEXT = "|".join(SUPPORTED_ENERGY_SOURCES)


# ---------------------------------------------------------------------------
# Central read-only tool catalog
# ---------------------------------------------------------------------------

_TOOL_CATALOG: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # PUBLIC MARKETPLACE LISTING TOOLS
    # -----------------------------------------------------------------------
    {
        "name": "get_all_listings",
        "http_method": "GET",
        "endpoint": "/api/v1/public/listings",
        "authentication_required": False,
        "purpose": (
            "Retrieve marketplace listings and backend supply aggregates for "
            "historical supply, listing status, renewable source, location, "
            "asking-price, stability, market-balance, demand-to-supply, "
            "prediction, and recommendation analytics."
        ),
        "when_to_use": [
            "Question asks about listings created during a historical period",
            "Question asks for historical listed supply",
            "Question asks to compare listed supply by renewable source",
            "Question asks about ACTIVE, SOLD, EXPIRED, or CANCELLED listings",
            "Question asks for asking-price history",
            "Question asks which source had the most stable newly listed supply",
            "Question asks for demand-to-supply ratio",
            "Question asks for historical shortage, surplus, or market balance",
            "Question requests demand, supply, shortage, or price prediction inputs",
            "Question asks which renewable credits should be listed",
            "Question asks for historical marketplace trends",
        ],
        "do_not_use_when": [
            "Question only asks for credits currently available to buy; use get_active_listings instead",
            "Question asks only for completed purchase demand; use get_all_purchases instead",
            "Question is purely conceptual and can be answered from RAG",
        ],
        "payload_schema": {
            "energy_source": (
                "optional string; allowed values: "
                f"{SUPPORTED_ENERGY_SOURCES_TEXT}"
            ),
            "location": (
                "optional string; case-insensitive partial city or region match"
            ),
            "status": (
                "optional string; allowed API values: "
                "ACTIVE|SOLD|EXPIRED|CANCELLED"
            ),
            "created_from": (
                "optional ISO-8601 date or datetime; inclusive listing-creation "
                "lower boundary"
            ),
            "created_to": (
                "optional ISO-8601 date or datetime; inclusive listing-creation "
                "upper boundary"
            ),
            "skip": "optional int >= 0; default 0",
            "limit": "optional int from 1 to 200; default 50",
        },
        "response_schema": {
            "meta": {
                "total": "int; total matching listings across all pages",
                "skip": "int; pagination offset",
                "limit": "int; requested page size",
                "filters_applied": {
                    "energy_source": "string|null",
                    "location": "string|null",
                    "status": "string|null",
                    "created_from": "ISO-8601 datetime|null",
                    "created_to": "ISO-8601 datetime|null",
                },
                "ai_context": "string; backend guidance for AI analytics",
            },
            "supply_by_source": {
                "EnergySource.<SOURCE>": {
                    "ListingStatus.<STATUS>": {
                        "count": "int",
                        "total_kwh": "number",
                        "avg_price_per_kwh": "number|null",
                        "min_price_per_kwh": "number|null",
                        "max_price_per_kwh": "number|null",
                    }
                }
            },
            "listings": [
                {
                    "id": "string UUID",
                    "seller_id": "string UUID or anonymized seller reference",
                    "energy_source": (
                        "string: " f"{SUPPORTED_ENERGY_SOURCES_TEXT}"
                    ),
                    "energy_kwh": "number",
                    "price_per_kwh": "number",
                    "location": "string|null",
                    "status": "string: ACTIVE|SOLD|EXPIRED|CANCELLED",
                    "blockchain_verified": "bool",
                    "created_at": "ISO-8601 datetime",
                    "expires_at": "ISO-8601 datetime|null",
                }
            ],
        },
        "aggregate_fields": [
            "supply_by_source",
        ],
        "analytics_supported": [
            "Historical listed supply by source",
            "Historical listed supply by location",
            "Newly listed kWh by period",
            "Listing-status distribution",
            "Historical asking-price analysis",
            "Supply trend by source",
            "Newly listed supply stability",
            "Demand-to-supply ratio denominator",
            "Historical shortage, surplus, and market balance",
            "Demand and price prediction features",
            "Seller recommendation features across supported sources",
        ],
        "example_request": {
            "energy_source": "BIOMASS",
            "location": "Noida",
            "status": "ACTIVE",
            "created_from": "2026-01-01T00:00:00Z",
            "created_to": "2026-07-15T23:59:59Z",
            "skip": 0,
            "limit": 200,
        },
        "example_response": {
            "meta": {
                "total": 1,
                "skip": 0,
                "limit": 200,
                "filters_applied": {
                    "energy_source": "BIOMASS",
                    "location": "Noida",
                    "status": "ACTIVE",
                    "created_from": "2026-01-01T00:00:00Z",
                    "created_to": "2026-07-15T23:59:59Z",
                },
                "ai_context": (
                    "supply_by_source aggregates all matching listing statuses"
                ),
            },
            "supply_by_source": {
                "EnergySource.BIOMASS": {
                    "ListingStatus.ACTIVE": {
                        "count": 1,
                        "total_kwh": 100,
                        "avg_price_per_kwh": 1,
                        "min_price_per_kwh": 1,
                        "max_price_per_kwh": 1,
                    }
                }
            },
            "listings": [
                {
                    "id": "e8b82f52-beea-439d-9ebe-1d717c890107",
                    "seller_id": "seller_01",
                    "energy_source": "BIOMASS",
                    "energy_kwh": 100,
                    "price_per_kwh": 1,
                    "location": "Noida",
                    "status": "ACTIVE",
                    "blockchain_verified": True,
                    "created_at": "2026-07-10T10:00:00Z",
                    "expires_at": "2026-08-10T10:00:00Z",
                }
            ],
        },
        "pagination_rule": (
            "Use meta.total with skip and limit. Fetch all pages before "
            "claiming total, all, market-wide, highest, lowest, percentage, "
            "ratio, stability, trend, prediction, or recommendation results. "
            "Increase skip by limit until the total is reached or fewer than "
            "limit records are returned."
        ),
    },
    {
        "name": "get_active_listings",
        "http_method": "GET",
        "endpoint": "/api/v1/public/listings/active",
        "authentication_required": False,
        "purpose": (
            "Retrieve listings that are currently active and available. The "
            "response includes source-level market share and location-level "
            "supply breakdowns optimized for AI answers."
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
            "Question asks for current buyer recommendation candidates",
            "Question asks for current supply saturation",
            "Question asks for current inventory used in a shortage forecast",
            "Question asks which renewable source should be listed now",
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
                f"{SUPPORTED_ENERGY_SOURCES_TEXT}"
            ),
            "location": (
                "optional string; case-insensitive partial city or region match"
            ),
            "min_price_per_kwh": "optional positive number; minimum ETH per kWh",
            "max_price_per_kwh": "optional positive number; maximum ETH per kWh",
            "min_energy_kwh": "optional positive number; minimum available kWh",
            "sort_by": (
                "optional string; allowed values: "
                "created_at|price_per_kwh|energy_kwh; default created_at"
            ),
            "sort_order": "optional string; allowed values: asc|desc; default desc",
            "skip": "optional int >= 0; default 0",
            "limit": "optional int from 1 to 200; default 50",
        },
        "response_schema": {
            "meta": {
                "total_active_listings": "int",
                "total_active_kwh": "number",
                "skip": "int",
                "limit": "int",
                "as_of": "ISO-8601 datetime",
                "ai_context": "string; backend guidance for AI analytics",
            },
            "source_breakdown": {
                "EnergySource.<SOURCE>": {
                    "active_listings": "int",
                    "total_kwh_available": "number",
                    "market_share_pct": "number",
                    "avg_price_per_kwh": "number|null",
                    "min_price_per_kwh": "number|null",
                    "max_price_per_kwh": "number|null",
                }
            },
            "location_breakdown": {
                "<location>": {
                    "EnergySource.<SOURCE>": {
                        "listings": "int",
                        "total_kwh": "number",
                    }
                }
            },
            "listings": [
                {
                    "id": "string UUID",
                    "energy_source": (
                        "string: " f"{SUPPORTED_ENERGY_SOURCES_TEXT}"
                    ),
                    "energy_kwh": "number",
                    "price_per_kwh": "number",
                    "location": "string|null",
                    "blockchain_verified": "bool",
                    "created_at": "ISO-8601 datetime",
                    "expires_at": "ISO-8601 datetime|null",
                }
            ],
        },
        "aggregate_fields": [
            "source_breakdown",
            "location_breakdown",
        ],
        "analytics_supported": [
            "Current available supply by source",
            "Current supply mix percentages across all supported sources",
            "Current active listings by location",
            "Highest-supply location for a selected source",
            "Lowest-priced available listing",
            "Highest-volume available listing",
            "Active supply saturation",
            "Current inventory for shortage prediction",
            "Buyer recommendation candidates",
        ],
        "example_request": {
            "energy_source": "GEOTHERMAL",
            "location": "Gujarat",
            "max_price_per_kwh": 2,
            "min_energy_kwh": 100,
            "sort_by": "price_per_kwh",
            "sort_order": "asc",
            "skip": 0,
            "limit": 200,
        },
        "example_response": {
            "meta": {
                "total_active_listings": 1,
                "total_active_kwh": 250,
                "skip": 0,
                "limit": 200,
                "as_of": "2026-07-15T07:25:39Z",
                "ai_context": (
                    "source_breakdown supplies market share; location_breakdown "
                    "supports location answers"
                ),
            },
            "source_breakdown": {
                "EnergySource.GEOTHERMAL": {
                    "active_listings": 1,
                    "total_kwh_available": 250,
                    "market_share_pct": 100,
                    "avg_price_per_kwh": 1,
                    "min_price_per_kwh": 1,
                    "max_price_per_kwh": 1,
                }
            },
            "location_breakdown": {
                "Gujarat": {
                    "EnergySource.GEOTHERMAL": {
                        "listings": 1,
                        "total_kwh": 250,
                    }
                }
            },
            "listings": [
                {
                    "id": "84dbe4ba-64fc-487e-bab3-1f0891f23545",
                    "energy_source": "GEOTHERMAL",
                    "energy_kwh": 250,
                    "price_per_kwh": 1,
                    "location": "Gujarat",
                    "blockchain_verified": True,
                    "created_at": "2026-07-15T10:00:00Z",
                    "expires_at": "2026-08-01T10:00:00Z",
                }
            ],
        },
        "pagination_rule": (
            "Use meta.total_active_listings with skip and limit. Fetch all "
            "pages before calculating total active supply, source percentages, "
            "location rankings, or recommendations. For top-N questions, "
            "server sorting and a limited result may be used only when the API "
            "guarantees complete ordering."
        ),
    },

    # -----------------------------------------------------------------------
    # PUBLIC PURCHASE TOOL
    # -----------------------------------------------------------------------
    {
        "name": "get_all_purchases",
        "http_method": "GET",
        "endpoint": "/api/v1/public/purchases",
        "authentication_required": False,
        "purpose": (
            "Retrieve marketplace-wide purchase records and backend demand "
            "aggregates for realized demand, selling-price, location demand, "
            "price volatility, forecasting, and recommendations."
        ),
        "when_to_use": [
            "Question asks about realized demand",
            "Question asks about completed purchase volume",
            "Question asks which source had highest demand",
            "Question asks which source had highest average selling price",
            "Question asks for realized selling-price history",
            "Question asks for demand-to-supply ratio",
            "Question asks about historical demand by location",
            "Question asks about price volatility",
            "Question asks for demand or price prediction",
            "Question asks whether a location may face a future shortage",
            "Question asks which renewable credits should be listed",
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
                f"{SUPPORTED_ENERGY_SOURCES_TEXT}"
            ),
            "location": (
                "optional string; case-insensitive partial seller-location match"
            ),
            "status": (
                "optional string; use COMPLETED for realized demand and "
                "realized selling-price analytics"
            ),
            "completed_from": (
                "optional ISO-8601 date or datetime; inclusive completion "
                "lower boundary"
            ),
            "completed_to": (
                "optional ISO-8601 date or datetime; inclusive completion "
                "upper boundary"
            ),
            "group_by_month": (
                "optional bool; when true, return monthly price trend by "
                "renewable source for next-month price forecasting"
            ),
            "skip": "optional int >= 0; default 0",
            "limit": "optional int from 1 to 200; default 50",
        },
        "response_schema": {
            "meta": {
                "total": "int; total matching purchases across all pages",
                "skip": "int",
                "limit": "int",
                "filters_applied": {
                    "energy_source": "string|null",
                    "location": "string|null",
                    "status": "string|null",
                    "completed_from": "ISO-8601 datetime|null",
                    "completed_to": "ISO-8601 datetime|null",
                },
                "ai_context": "string; backend guidance for AI analytics",
            },
            "demand_by_source": {
                "EnergySource.<SOURCE>": {
                    "total_purchases": "int",
                    "total_kwh_sold": "number",
                    "demand_share_pct": "number",
                    "avg_price_per_kwh": "number|null",
                    "min_price_per_kwh": "number|null",
                    "max_price_per_kwh": "number|null",
                    "price_volatility": "number|null",
                    "total_revenue_eth": "number",
                }
            },
            "location_demand_breakdown": {
                "<location>": {
                    "EnergySource.<SOURCE>": {
                        "purchases": "int",
                        "kwh_sold": "number",
                    }
                }
            },
            "monthly_price_trend": (
                "object containing monthly source buckets when "
                "group_by_month=true; otherwise an instructional string"
            ),
            "purchases": [
                {
                    "id": "string UUID",
                    "listing_id": "string UUID",
                    "energy_source": (
                        "string: " f"{SUPPORTED_ENERGY_SOURCES_TEXT}"
                    ),
                    "location": "string|null",
                    "energy_kwh": "number",
                    "price_per_kwh": "number",
                    "total_price": "number",
                    "status": "string such as PENDING or COMPLETED",
                    "created_at": "ISO-8601 datetime",
                    "completed_at": "ISO-8601 datetime|null",
                }
            ],
        },
        "aggregate_fields": [
            "demand_by_source",
            "location_demand_breakdown",
            "monthly_price_trend",
        ],
        "analytics_supported": [
            "Realized demand by source",
            "Realized demand by location",
            "Completed purchase volume",
            "Source demand share",
            "Volume-weighted realized selling price",
            "Historical demand trend",
            "Historical realized-price trend",
            "Demand-to-supply ratio numerator",
            "Backend-computed price volatility",
            "Demand prediction features across supported sources",
            "Price prediction features across supported sources",
            "Shortage prediction features",
            "Seller and buyer recommendation features",
        ],
        "example_request": {
            "energy_source": "TIDAL",
            "location": "Chennai",
            "status": "COMPLETED",
            "completed_from": "2026-01-01T00:00:00Z",
            "completed_to": "2026-07-15T23:59:59Z",
            "group_by_month": True,
            "skip": 0,
            "limit": 200,
        },
        "example_response": {
            "meta": {
                "total": 1,
                "skip": 0,
                "limit": 200,
                "filters_applied": {
                    "energy_source": "TIDAL",
                    "location": "Chennai",
                    "status": "COMPLETED",
                    "completed_from": "2026-01-01T00:00:00Z",
                    "completed_to": "2026-07-15T23:59:59Z",
                },
                "ai_context": (
                    "demand_by_source supports demand, price, and volatility "
                    "analytics"
                ),
            },
            "demand_by_source": {
                "EnergySource.TIDAL": {
                    "total_purchases": 1,
                    "total_kwh_sold": 50,
                    "demand_share_pct": 100,
                    "avg_price_per_kwh": 1,
                    "min_price_per_kwh": 1,
                    "max_price_per_kwh": 1,
                    "price_volatility": 0,
                    "total_revenue_eth": 50,
                }
            },
            "location_demand_breakdown": {
                "Chennai": {
                    "EnergySource.TIDAL": {
                        "purchases": 1,
                        "kwh_sold": 50,
                    }
                }
            },
            "monthly_price_trend": {
                "2026-07": {
                    "EnergySource.TIDAL": {
                        "avg_price": 1,
                        "total_kwh": 50,
                        "purchase_count": 1,
                    }
                }
            },
            "purchases": [
                {
                    "id": "66c098e1-d5eb-4d21-b2b1-696c4a79a1f4",
                    "listing_id": "e8b82f52-beea-439d-9ebe-1d717c890107",
                    "energy_source": "TIDAL",
                    "location": "Chennai",
                    "energy_kwh": 50,
                    "price_per_kwh": 1,
                    "total_price": 50,
                    "status": "COMPLETED",
                    "created_at": "2026-07-15T10:00:00Z",
                    "completed_at": "2026-07-15T10:05:00Z",
                }
            ],
        },
        "pagination_rule": (
            "Use meta.total with skip and limit. Fetch all pages before "
            "calculating market-wide demand, average selling price, demand "
            "share, volatility, trend, prediction, or recommendation results. "
            "Preserve aggregate sections from the first response page."
        ),
        "privacy_rule": (
            "The public response must not expose buyer email, seller email, "
            "phone number, full address, wallet secrets, passwords, tokens, "
            "private credentials, or other sensitive personal information."
        ),
    },
]


def get_tool_catalog() -> List[Dict[str, Any]]:
    """Return a shallow copy of the complete read-only tool catalog."""
    return list(_TOOL_CATALOG)


def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return one registered tool definition by exact tool name."""
    normalized_name = str(name or "").strip()
    for tool in _TOOL_CATALOG:
        if tool["name"] == normalized_name:
            return tool
    return None


def get_allowed_tool_names() -> List[str]:
    """Return all tool names that the planner may select."""
    return [tool["name"] for tool in _TOOL_CATALOG]


def build_planner_tools_text() -> str:
    """Render the complete catalog as formatted JSON for debugging or docs."""
    return json.dumps(
        _TOOL_CATALOG,
        ensure_ascii=True,
        indent=2,
    )


def build_compact_planner_tools_text() -> str:
    """
    Render a token-efficient planner catalog.

    The LLM planner needs tool names, endpoints, purposes, and filter names. Full
    response schemas, examples, analytics notes, and pagination rules remain in
    the central catalog for validation, documentation, and troubleshooting but
    are intentionally excluded from each planner request.
    """
    compact_catalog = [
        {
            "name": tool["name"],
            "endpoint": tool["endpoint"],
            "purpose": tool["purpose"],
            "filters": list(tool.get("payload_schema", {}).keys()),
        }
        for tool in _TOOL_CATALOG
    ]

    return json.dumps(
        compact_catalog,
        ensure_ascii=True,
        separators=(",", ":"),
    )

