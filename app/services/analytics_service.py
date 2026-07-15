"""
analytics_service.py
--------------------

Deterministic analytics, forecasting, and recommendation engine for GreenGrid
Exchange.

This module does not call APIs, OpenSearch, or Amazon Bedrock. It receives the
validated tool plan and normalized marketplace tool results, performs all
calculations, and returns compact structured results for the final LLM answer.

Supported renewable sources are loaded centrally from app.config and currently
include SOLAR, WIND, HYDRO, BIOMASS, GEOTHERMAL, TIDAL, and OTHER.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_SOURCES: Tuple[str, ...] = tuple(
    getattr(
        settings,
        "SUPPORTED_ENERGY_SOURCES",
        ("SOLAR", "WIND", "HYDRO", "BIOMASS", "GEOTHERMAL", "TIDAL", "OTHER"),
    )
)
COMPLETED_PURCHASE_STATUSES = {"completed", "consumed"}
ACTIVE_LISTING_STATUS = "active"

SOURCE_ALIASES = {
    "solar": "SOLAR", "solar_energy": "SOLAR", "solar energy": "SOLAR", "solar power": "SOLAR",
    "wind": "WIND", "wind_energy": "WIND", "wind energy": "WIND", "wind power": "WIND",
    "hydro": "HYDRO", "hydropower": "HYDRO", "hydro_energy": "HYDRO", "hydro energy": "HYDRO", "hydro power": "HYDRO",
    "biomass": "BIOMASS", "bio mass": "BIOMASS", "biomass_energy": "BIOMASS", "biomass energy": "BIOMASS", "bioenergy": "BIOMASS",
    "geothermal": "GEOTHERMAL", "geothermal_energy": "GEOTHERMAL", "geothermal energy": "GEOTHERMAL", "geothermal power": "GEOTHERMAL",
    "tidal": "TIDAL", "tidal_energy": "TIDAL", "tidal energy": "TIDAL", "tidal power": "TIDAL",
    "other": "OTHER", "other_renewable": "OTHER", "other renewable": "OTHER", "other source": "OTHER",
    "small_hydro": "HYDRO", "small hydro": "HYDRO", "small-hydro": "HYDRO",
}

PREDICTION_INTENTS = {"demand_prediction", "price_prediction", "shortage_prediction"}
RECOMMENDATION_INTENTS = {"seller_recommendation", "buyer_recommendation"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_plan(
    plan: Mapping[str, Any],
    tool_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Execute deterministic analysis for the validated planner intent."""
    intent = str(plan.get("intent", "none") or "none").strip().lower()
    datasets = _extract_datasets(tool_results)
    aggregates = _extract_aggregates(tool_results)
    limitations = _tool_limitations(tool_results)
    records_analyzed = {
        "all_listings": len(datasets["all_listings"]),
        "active_listings": len(datasets["active_listings"]),
        "purchases": len(datasets["purchases"]),
    }

    analytics_result: Dict[str, Any] = {}
    prediction_result: Optional[Dict[str, Any]] = None
    recommendation_result: Optional[Dict[str, Any]] = None
    confidence: Optional[str] = None
    calculation_method: Optional[str] = None

    try:
        if intent == "current_supply":
            analytics_result = _current_supply(datasets["active_listings"], aggregates["active_listings"])
            calculation_method = "active source totals from source_breakdown, with raw-record fallback"
            confidence = _descriptive_confidence(datasets["active_listings"], limitations, aggregates["active_listings"])

        elif intent == "supply_mix":
            analytics_result = _supply_mix(datasets["active_listings"], aggregates["active_listings"])
            calculation_method = "active source kWh divided by total active kWh"
            confidence = _descriptive_confidence(datasets["active_listings"], limitations, aggregates["active_listings"])

        elif intent == "supply_by_location":
            analytics_result = _supply_by_location(datasets["active_listings"], aggregates["active_listings"], plan)
            calculation_method = "active kWh grouped by location, optionally filtered by source"
            confidence = _descriptive_confidence(datasets["active_listings"], limitations, aggregates["active_listings"])

        elif intent == "marketplace_summary":
            analytics_result = _marketplace_summary(datasets=datasets, aggregates=aggregates)
            calculation_method = (
                "current active inventory plus same-day listing and completed-purchase "
                "activity, grouped by source and location"
            )
            confidence = _combined_descriptive_confidence(datasets, limitations, aggregates)

        elif intent == "historical_supply":
            analytics_result = _historical_supply(datasets["all_listings"], aggregates["all_listings"])
            calculation_method = "listed kWh grouped by source from supply_by_source, with raw-record fallback"
            confidence = _descriptive_confidence(datasets["all_listings"], limitations, aggregates["all_listings"])

        elif intent == "historical_demand":
            analytics_result = _historical_demand(datasets["purchases"], aggregates["purchases"])
            calculation_method = "completed purchase kWh grouped by source"
            confidence = _descriptive_confidence(datasets["purchases"], limitations, aggregates["purchases"])

        elif intent == "demand_and_supply":
            analytics_result = _demand_and_supply(
                datasets["all_listings"],
                datasets["purchases"],
                aggregates["all_listings"],
                aggregates["purchases"],
            )
            calculation_method = (
                "total supply = remaining listing kWh + sold completed-purchase kWh; "
                "realized demand = completed-purchase kWh"
            )
            confidence = _combined_descriptive_confidence(datasets, limitations, aggregates)

        elif intent == "average_selling_price":
            analytics_result = _average_selling_price(datasets["purchases"], aggregates["purchases"])
            calculation_method = "volume-weighted completed selling price by source"
            confidence = _descriptive_confidence(datasets["purchases"], limitations, aggregates["purchases"])

        elif intent == "demand_supply_ratio":
            analytics_result = _demand_supply_ratio(datasets["all_listings"], datasets["purchases"], aggregates["all_listings"], aggregates["purchases"])
            calculation_method = "completed purchase kWh divided by listed kWh"
            confidence = _combined_descriptive_confidence(datasets, limitations, aggregates)

        elif intent == "market_balance":
            analytics_result = _market_balance(datasets["all_listings"], datasets["purchases"], aggregates["all_listings"], aggregates["purchases"])
            calculation_method = "listed supply kWh minus completed demand kWh"
            confidence = _combined_descriptive_confidence(datasets, limitations, aggregates)

        elif intent == "supply_stability":
            analytics_result = _supply_stability(datasets["all_listings"])
            calculation_method = "weekly listed-supply coefficient of variation"
            confidence = _time_series_confidence(datasets["all_listings"], "created_at", limitations)
            limitations.append("Supply stability measures newly listed weekly supply, not historical daily active inventory.")

        elif intent == "price_volatility":
            analytics_result = _price_volatility(datasets["purchases"], aggregates["purchases"])
            calculation_method = "backend price volatility where available, otherwise weekly weighted-price coefficient of variation"
            confidence = _time_series_confidence(datasets["purchases"], "completed_at", limitations)

        elif intent == "demand_prediction":
            analytics_result = {
                "historical_demand": _historical_demand(datasets["purchases"], aggregates["purchases"]),
                "historical_supply": _historical_supply(datasets["all_listings"], aggregates["all_listings"]),
                "demand_supply_ratio": _demand_supply_ratio(datasets["all_listings"], datasets["purchases"], aggregates["all_listings"], aggregates["purchases"]),
            }
            prediction_result, confidence, extra = _predict_demand(datasets["purchases"], plan)
            limitations.extend(extra)
            calculation_method = "recursive four-period weighted moving-average demand forecast"

        elif intent == "price_prediction":
            analytics_result = _average_selling_price(datasets["purchases"], aggregates["purchases"])
            prediction_result, confidence, extra = _predict_price(datasets["purchases"], aggregates["purchases"], plan)
            limitations.extend(extra)
            calculation_method = "monthly API trend forecast, with weekly raw-record fallback"

        elif intent == "shortage_prediction":
            analytics_result = {
                "historical_market_balance": _market_balance(datasets["all_listings"], datasets["purchases"], aggregates["all_listings"], aggregates["purchases"]),
                "current_active_supply": _current_supply(datasets["active_listings"], aggregates["active_listings"]),
            }
            prediction_result, confidence, extra = _predict_shortage(datasets["all_listings"], datasets["active_listings"], datasets["purchases"], plan)
            limitations.extend(extra)
            calculation_method = "forecast newly listed supply plus current inventory minus forecast completed demand"

        elif intent == "seller_recommendation":
            analytics_result = {
                "current_supply": _current_supply(datasets["active_listings"], aggregates["active_listings"]),
                "historical_demand": _historical_demand(datasets["purchases"], aggregates["purchases"]),
                "demand_supply_ratio": _demand_supply_ratio(datasets["all_listings"], datasets["purchases"], aggregates["all_listings"], aggregates["purchases"]),
                "average_selling_price": _average_selling_price(datasets["purchases"], aggregates["purchases"]),
            }
            recommendation_result, confidence, extra = _seller_recommendation(datasets["all_listings"], datasets["active_listings"], datasets["purchases"])
            limitations.extend(extra)
            calculation_method = "opportunity score: demand growth 35%, demand-to-supply 30%, realized-price strength 20%, low active saturation 15%"

        elif intent == "buyer_recommendation":
            analytics_result = {
                "current_supply": _current_supply(datasets["active_listings"], aggregates["active_listings"]),
                "historical_demand": _historical_demand(datasets["purchases"], aggregates["purchases"]),
            }
            recommendation_result, confidence, extra = _buyer_recommendation(datasets["active_listings"], datasets["purchases"])
            limitations.extend(extra)
            calculation_method = "listing score: price 40%, historical demand 30%, quantity 20%, recency 10%"

        else:
            analytics_result = _generic_dataset_summary(datasets)
            calculation_method = "record summary"
            confidence = _combined_descriptive_confidence(datasets, limitations, aggregates)

    except Exception as exc:
        logger.exception("Analytics processing failed for intent=%s", intent)
        limitations.append(f"Analytics calculation failed: {type(exc).__name__}")
        confidence = "insufficient_data"
        analytics_result = {}
        prediction_result = None
        recommendation_result = None

    return {
        "intent": intent,
        "analytics_result": _json_safe(analytics_result),
        "prediction_result": _json_safe(prediction_result) if prediction_result else None,
        "recommendation_result": _json_safe(recommendation_result) if recommendation_result else None,
        "confidence": confidence,
        "calculation_method": calculation_method,
        "limitations": _unique_strings(limitations),
        "records_analyzed": records_analyzed,
    }


# ---------------------------------------------------------------------------
# Dataset and aggregate extraction
# ---------------------------------------------------------------------------


def _extract_datasets(tool_results: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    datasets = {"all_listings": [], "active_listings": [], "purchases": []}
    for result in tool_results:
        tool = str(result.get("tool", ""))
        records = _records_from_result(result)
        if tool == "get_all_listings":
            datasets["all_listings"].extend(_clean_listings(records))
        elif tool == "get_active_listings":
            datasets["active_listings"].extend(_clean_listings(records, active_only=True))
        elif tool == "get_all_purchases":
            datasets["purchases"].extend(_clean_purchases(records))
    return {key: _deduplicate(records) for key, records in datasets.items()}


def _extract_aggregates(tool_results: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = {"all_listings": {}, "active_listings": {}, "purchases": {}}
    for tool_result in tool_results:
        data = tool_result.get("data", {})
        aggregates = data.get("aggregates", {}) if isinstance(data, Mapping) else {}
        if not isinstance(aggregates, Mapping):
            continue
        tool = tool_result.get("tool")
        if tool == "get_all_listings":
            result["all_listings"] = dict(aggregates)
        elif tool == "get_active_listings":
            result["active_listings"] = dict(aggregates)
        elif tool == "get_all_purchases":
            result["purchases"] = dict(aggregates)
    return result


def _records_from_result(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data = result.get("data", {})
    records = data.get("records", []) if isinstance(data, Mapping) else []
    return [dict(record) for record in records if isinstance(record, Mapping)] if isinstance(records, list) else []


def _clean_listings(records: Sequence[Mapping[str, Any]], active_only: bool = False) -> List[Dict[str, Any]]:
    cleaned = []
    for raw in records:
        source = _source(raw.get("energy_source") or raw.get("source"))
        energy = _number(raw.get("energy_kwh"))
        if source is None or energy is None or energy <= 0:
            continue
        status = str(raw.get("status", "ACTIVE" if active_only else "")).strip().lower()
        is_available = raw.get("is_available", True if active_only else None)
        if active_only and not (status == ACTIVE_LISTING_STATUS and is_available is not False):
            continue
        record = dict(raw)
        record.update(
            energy_source=source,
            energy_kwh=energy,
            price_per_kwh=_number(raw.get("price_per_kwh")),
            status=status,
            location=_normalize_location(raw.get("location")),
            created_at=_datetime(raw.get("created_at")),
            expires_at=_datetime(raw.get("expires_at")),
        )
        cleaned.append(record)
    return cleaned


def _clean_purchases(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for raw in records:
        source = _source(raw.get("energy_source") or raw.get("source"))
        energy = _number(raw.get("energy_kwh"))
        status = str(raw.get("status", "")).strip().lower()
        completed_at = _datetime(raw.get("completed_at"))
        if source is None or energy is None or energy <= 0:
            continue
        if status not in COMPLETED_PURCHASE_STATUSES or completed_at is None:
            continue
        record = dict(raw)
        record.update(
            energy_source=source,
            energy_kwh=energy,
            price_per_kwh=_number(raw.get("price_per_kwh")),
            status=status,
            location=_normalize_location(raw.get("location")),
            completed_at=completed_at,
        )
        cleaned.append(record)
    return cleaned


# ---------------------------------------------------------------------------
# Descriptive analytics
# ---------------------------------------------------------------------------


def _marketplace_summary(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    aggregates: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build current inventory plus today's listing and completed-demand summary."""
    active_records = datasets.get("active_listings", [])
    listing_records = datasets.get("all_listings", [])
    purchase_records = datasets.get("purchases", [])
    active_aggregates = aggregates.get("active_listings", {})
    listing_aggregates = aggregates.get("all_listings", {})
    purchase_aggregates = aggregates.get("purchases", {})

    current_supply = _current_supply(active_records, active_aggregates)
    supply_mix = _supply_mix(active_records, active_aggregates)
    new_supply = _historical_supply(listing_records, listing_aggregates)
    completed_demand = _historical_demand(purchase_records, purchase_aggregates)
    realized_prices = _average_selling_price(purchase_records, purchase_aggregates)
    balance = _market_balance(listing_records, purchase_records, listing_aggregates, purchase_aggregates)

    active_breakdown = active_aggregates.get("source_breakdown", {})
    source_statistics: Dict[str, Dict[str, Any]] = {}
    for source in SUPPORTED_SOURCES:
        stats = active_breakdown.get(source, {}) if isinstance(active_breakdown, Mapping) else {}
        source_statistics[source] = {
            "active_listings": int(_number(stats.get("active_listings")) or 0) if isinstance(stats, Mapping) else 0,
            "available_supply_kwh": current_supply["supply_by_source_kwh"].get(source, 0.0),
            "market_share_pct": supply_mix["supply_mix_percentage"].get(source, 0.0),
            "average_asking_price_per_kwh": _round(_number(stats.get("avg_price_per_kwh")), 8) if isinstance(stats, Mapping) else None,
            "newly_listed_kwh_today": new_supply["listed_supply_kwh_by_source"].get(source, 0.0),
            "completed_demand_kwh_today": completed_demand["demand_kwh_by_source"].get(source, 0.0),
            "average_realized_price_per_kwh_today": realized_prices["weighted_average_selling_price_by_source"].get(source),
            "market_balance_kwh_today": balance["market_balance_kwh_by_source"].get(source, 0.0),
        }

    location_supply = _location_totals_from_aggregate(active_aggregates.get("location_breakdown", {}), "total_kwh")
    location_demand = _location_totals_from_aggregate(purchase_aggregates.get("location_demand_breakdown", {}), "kwh_sold")
    top_supply_location, top_supply_value = _positive_max_item(location_supply)
    top_demand_location, top_demand_value = _positive_max_item(location_demand)

    return {
        "current_inventory": current_supply,
        "supply_mix": supply_mix,
        "today_listing_activity": {
            "new_listing_count": len(listing_records),
            "newly_listed_supply_kwh": new_supply["total_listed_supply_kwh"],
            "newly_listed_supply_by_source_kwh": new_supply["listed_supply_kwh_by_source"],
        },
        "today_purchase_activity": {
            "completed_purchase_count": len(purchase_records),
            "completed_demand_kwh": completed_demand["total_completed_demand_kwh"],
            "completed_demand_by_source_kwh": completed_demand["demand_kwh_by_source"],
            "weighted_average_realized_price_by_source": realized_prices["weighted_average_selling_price_by_source"],
        },
        "source_statistics": source_statistics,
        "today_market_balance": balance,
        "location_highlights": {
            "highest_active_supply_location": top_supply_location,
            "highest_active_supply_kwh": _round(top_supply_value),
            "highest_completed_demand_location": top_demand_location,
            "highest_completed_demand_kwh": _round(top_demand_value),
        },
    }


def _location_totals_from_aggregate(value: Any, metric: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    if not isinstance(value, Mapping):
        return totals
    for location, source_map in value.items():
        if not isinstance(source_map, Mapping):
            continue
        totals[str(location)] = sum(
            _number(stats.get(metric)) or 0.0
            for stats in source_map.values()
            if isinstance(stats, Mapping)
        )
    return totals


def _active_totals(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, float]:
    breakdown = aggregates.get("source_breakdown", {}) if isinstance(aggregates, Mapping) else {}
    if isinstance(breakdown, Mapping) and breakdown:
        return {source: _round(_number((breakdown.get(source) or {}).get("total_kwh_available")) or 0.0) or 0.0 for source in SUPPORTED_SOURCES}
    return _sum_energy_by_source(records)


def _listed_totals(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, float]:
    supply = aggregates.get("supply_by_source", {}) if isinstance(aggregates, Mapping) else {}
    if isinstance(supply, Mapping) and supply:
        totals = {source: 0.0 for source in SUPPORTED_SOURCES}
        for source in SUPPORTED_SOURCES:
            statuses = supply.get(source, {})
            if isinstance(statuses, Mapping):
                totals[source] = sum(_number(stats.get("total_kwh")) or 0.0 for stats in statuses.values() if isinstance(stats, Mapping))
        return {source: _round(value) or 0.0 for source, value in totals.items()}
    return _sum_energy_by_source(records)


def _demand_totals(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, float]:
    demand = aggregates.get("demand_by_source", {}) if isinstance(aggregates, Mapping) else {}
    if isinstance(demand, Mapping) and demand:
        return {source: _round(_number((demand.get(source) or {}).get("total_kwh_sold")) or 0.0) or 0.0 for source in SUPPORTED_SOURCES}
    return _sum_energy_by_source(records)


def _current_supply(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, Any]:
    totals = _active_totals(records, aggregates)
    winner, value = _positive_max_item(totals)
    return {"supply_by_source_kwh": totals, "total_active_supply_kwh": _round(sum(totals.values())), "highest_supply_source": winner, "highest_supply_kwh": _round(value), "listing_count": len(records)}


def _supply_mix(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, Any]:
    totals = _active_totals(records, aggregates)
    grand_total = sum(totals.values())
    breakdown = aggregates.get("source_breakdown", {}) if isinstance(aggregates, Mapping) else {}
    percentages = {}
    for source in SUPPORTED_SOURCES:
        backend = _number((breakdown.get(source) or {}).get("market_share_pct")) if isinstance(breakdown, Mapping) else None
        percentages[source] = _round(backend if backend is not None else ((totals[source] / grand_total) * 100 if grand_total else 0.0), 2)
    return {"active_supply_kwh_by_source": totals, "total_active_supply_kwh": _round(grand_total), "supply_mix_percentage": percentages, "basis": "active energy_kwh, not listing count"}


def _supply_by_location(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    requested_source = _plan_source(plan)
    values: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    breakdown = aggregates.get("location_breakdown", {}) if isinstance(aggregates, Mapping) else {}
    if isinstance(breakdown, Mapping) and breakdown:
        for location, source_map in breakdown.items():
            if not isinstance(source_map, Mapping):
                continue
            for source, stats in source_map.items():
                if source not in SUPPORTED_SOURCES or (requested_source and source != requested_source) or not isinstance(stats, Mapping):
                    continue
                values[str(location)] += _number(stats.get("total_kwh")) or 0.0
                counts[str(location)] += int(_number(stats.get("listings")) or 0)
    else:
        for record in records:
            if requested_source and record.get("energy_source") != requested_source:
                continue
            location = record.get("location")
            if location:
                values[str(location)] += float(record["energy_kwh"])
                counts[str(location)] += 1
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return {
        "energy_source": requested_source,
        "supply_kwh_by_location": {location: _round(value) for location, value in ranked},
        "listing_count_by_location": dict(counts),
        "highest_supply_location": ranked[0][0] if ranked and ranked[0][1] > 0 else None,
        "highest_supply_kwh": _round(ranked[0][1]) if ranked and ranked[0][1] > 0 else None,
    }


def _historical_supply(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, Any]:
    totals = _listed_totals(records, aggregates)
    winner, value = _positive_max_item(totals)
    return {"listed_supply_kwh_by_source": totals, "total_listed_supply_kwh": _round(sum(totals.values())), "highest_listed_supply_source": winner, "highest_listed_supply_kwh": _round(value), "listing_count": len(records)}


def _historical_demand(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, Any]:
    totals = _demand_totals(records, aggregates)
    grand_total = sum(totals.values())
    demand = aggregates.get("demand_by_source", {}) if isinstance(aggregates, Mapping) else {}
    counts = {}
    percentages = {}
    for source in SUPPORTED_SOURCES:
        stats = demand.get(source, {}) if isinstance(demand, Mapping) else {}
        counts[source] = int(_number(stats.get("total_purchases")) or sum(1 for record in records if record.get("energy_source") == source)) if isinstance(stats, Mapping) else sum(1 for record in records if record.get("energy_source") == source)
        backend_share = _number(stats.get("demand_share_pct")) if isinstance(stats, Mapping) else None
        percentages[source] = _round(backend_share if backend_share is not None else ((totals[source] / grand_total) * 100 if grand_total else 0.0), 2)
    winner, value = _positive_max_item(totals)
    return {"demand_kwh_by_source": totals, "demand_share_percentage": percentages, "purchase_count_by_source": counts, "total_completed_demand_kwh": _round(grand_total), "highest_demand_source": winner, "highest_demand_kwh": _round(value)}


def _average_selling_price(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, Any]:
    demand = aggregates.get("demand_by_source", {}) if isinstance(aggregates, Mapping) else {}
    prices: Dict[str, Optional[float]] = {}
    volumes = _demand_totals(records, aggregates)
    for source in SUPPORTED_SOURCES:
        stats = demand.get(source, {}) if isinstance(demand, Mapping) else {}
        backend = _number(stats.get("avg_price_per_kwh")) if isinstance(stats, Mapping) else None
        prices[source] = backend if backend is not None else _weighted_average_price([record for record in records if record.get("energy_source") == source])
    valid = {source: value for source, value in prices.items() if value is not None}
    winner, value = _max_item(valid)
    return {"weighted_average_selling_price_by_source": {source: _round(price, 8) for source, price in prices.items()}, "sold_volume_kwh_by_source": volumes, "highest_average_price_source": winner, "highest_average_price_per_kwh": _round(value, 8), "price_basis": "completed purchase volume-weighted realized price"}


def _demand_and_supply(
    listings: Sequence[Mapping[str, Any]],
    purchases: Sequence[Mapping[str, Any]],
    listing_aggregates: Mapping[str, Any],
    purchase_aggregates: Mapping[str, Any],
) -> Dict[str, Any]:
    """Calculate remaining supply, sold supply/demand, and total issued supply.

    GreenGrid API semantics:
    - get_all_listings contains credits that remain unsold for the scope.
    - get_all_purchases contains sold credits.
    - total supply therefore equals remaining listing kWh plus sold kWh.
    - realized demand equals sold/completed purchase kWh.
    """
    remaining_supply = _listed_totals(listings, listing_aggregates)
    sold_supply_and_demand = _demand_totals(purchases, purchase_aggregates)
    total_supply = {
        source: _round(remaining_supply[source] + sold_supply_and_demand[source]) or 0.0
        for source in SUPPORTED_SOURCES
    }
    remaining_share = {
        source: _round((remaining_supply[source] / total_supply[source]) * 100, 2) if total_supply[source] > 0 else 0.0
        for source in SUPPORTED_SOURCES
    }
    sold_share = {
        source: _round((sold_supply_and_demand[source] / total_supply[source]) * 100, 2) if total_supply[source] > 0 else 0.0
        for source in SUPPORTED_SOURCES
    }
    leader, leader_value = _positive_max_item(total_supply)
    demand_leader, demand_leader_value = _positive_max_item(sold_supply_and_demand)
    return {
        "remaining_supply_kwh_by_source": remaining_supply,
        "sold_supply_kwh_by_source": sold_supply_and_demand,
        "realized_demand_kwh_by_source": sold_supply_and_demand,
        "total_supply_kwh_by_source": total_supply,
        "remaining_supply_percentage_by_source": remaining_share,
        "sold_percentage_by_source": sold_share,
        "highest_total_supply_source": leader,
        "highest_total_supply_kwh": _round(leader_value),
        "highest_demand_source": demand_leader,
        "highest_demand_kwh": _round(demand_leader_value),
        "supply_definition": "total supply = remaining credits from get_all_listings + sold credits from completed get_all_purchases",
        "demand_definition": "realized demand = sold credits from completed purchases",
    }


def _demand_supply_ratio(listings, purchases, listing_aggregates, purchase_aggregates) -> Dict[str, Any]:
    supply = _listed_totals(listings, listing_aggregates)
    demand = _demand_totals(purchases, purchase_aggregates)
    ratios = {source: _round(demand[source] / supply[source], 4) if supply[source] > 0 else None for source in SUPPORTED_SOURCES}
    winner, value = _max_item({source: ratio for source, ratio in ratios.items() if ratio is not None})
    return {"listed_supply_kwh_by_source": supply, "completed_demand_kwh_by_source": demand, "demand_supply_ratio_by_source": ratios, "highest_ratio_source": winner, "highest_ratio": _round(value, 4)}


def _market_balance(listings, purchases, listing_aggregates, purchase_aggregates) -> Dict[str, Any]:
    supply = _listed_totals(listings, listing_aggregates)
    demand = _demand_totals(purchases, purchase_aggregates)
    balances = {source: _round(supply[source] - demand[source]) or 0.0 for source in SUPPORTED_SOURCES}
    conditions = {source: "surplus" if value > 0 else "shortage" if value < 0 else "balanced" for source, value in balances.items()}
    return {"listed_supply_kwh_by_source": supply, "completed_demand_kwh_by_source": demand, "market_balance_kwh_by_source": balances, "market_condition_by_source": conditions}


def _supply_stability(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    weekly = _weekly_series(records, "created_at", "energy_kwh", False)
    coefficients = {source: _coefficient_of_variation([value for _, value in weekly[source]]) for source in SUPPORTED_SOURCES}
    winner, value = _min_item({source: coefficient for source, coefficient in coefficients.items() if coefficient is not None})
    return {"weekly_listed_supply": {source: [{"week": week, "listed_kwh": _round(value)} for week, value in weekly[source]] for source in SUPPORTED_SOURCES}, "supply_stability_coefficient_by_source": coefficients, "most_stable_supply_source": winner, "lowest_stability_coefficient": _round(value, 4)}


def _price_volatility(records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]) -> Dict[str, Any]:
    weekly = _weekly_series(records, "completed_at", "price_per_kwh", True)
    demand = aggregates.get("demand_by_source", {}) if isinstance(aggregates, Mapping) else {}
    coefficients = {}
    for source in SUPPORTED_SOURCES:
        stats = demand.get(source, {}) if isinstance(demand, Mapping) else {}
        backend = _number(stats.get("price_volatility")) if isinstance(stats, Mapping) else None
        coefficients[source] = _round(backend, 4) if backend is not None else _coefficient_of_variation([value for _, value in weekly[source]])
    winner, value = _max_item({source: coefficient for source, coefficient in coefficients.items() if coefficient is not None})
    return {"weekly_weighted_realized_prices": {source: [{"week": week, "weighted_price_per_kwh": _round(value, 8)} for week, value in weekly[source]] for source in SUPPORTED_SOURCES}, "price_volatility_coefficient_by_source": coefficients, "highest_price_volatility_source": winner, "highest_volatility_coefficient": _round(value, 4)}


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def _predict_demand(purchases: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> Tuple[Dict[str, Any], str, List[str]]:
    weekly = _weekly_energy_by_source(purchases, "completed_at")
    periods = _forecast_periods(plan)
    results = {source: _recursive_forecast([value for _, value in weekly[source]], periods, price=False) for source in SUPPORTED_SOURCES}
    eligible = {source: result.get("predicted_forecast_period_kwh") for source, result in results.items() if result.get("predicted_forecast_period_kwh") is not None}
    winner, value = _max_item(eligible)
    history_periods = max((len(weekly[source]) for source in SUPPORTED_SOURCES), default=0)
    confidence = _prediction_confidence(len(purchases), history_periods)
    limits = ["Future marketplace demand is not guaranteed."]
    if confidence == "insufficient_data":
        limits.insert(0, _minimum_prediction_data_message())
    return {"metric": "completed_demand_kwh", "model": "recursive_four_period_weighted_moving_average", "forecast_weeks": periods, "predictions_by_source": results, "predicted_highest_demand_source": winner, "predicted_highest_demand_kwh": _round(value), "historical_purchase_records": len(purchases), "historical_periods": history_periods}, confidence, limits


def _predict_price(purchases: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any], plan: Mapping[str, Any]) -> Tuple[Dict[str, Any], str, List[str]]:
    monthly = _parse_monthly_price_trend(aggregates.get("monthly_price_trend") if isinstance(aggregates, Mapping) else None)
    use_monthly = any(monthly[source] for source in SUPPORTED_SOURCES)
    if use_monthly:
        results = {source: _recursive_forecast([value for _, value in monthly[source]], 1, price=True) for source in SUPPORTED_SOURCES}
        periods = max((len(monthly[source]) for source in SUPPORTED_SOURCES), default=0)
        model = "monthly_api_price_trend_weighted_moving_average"
    else:
        weekly = _weekly_series(purchases, "completed_at", "price_per_kwh", True)
        forecast_periods = _forecast_periods(plan)
        results = {source: _recursive_forecast([value for _, value in weekly[source]], forecast_periods, price=True) for source in SUPPORTED_SOURCES}
        periods = max((len(weekly[source]) for source in SUPPORTED_SOURCES), default=0)
        model = "weekly_weighted_moving_average"
    eligible = {source: result.get("predicted_forecast_period_price_per_kwh") for source, result in results.items() if result.get("predicted_forecast_period_price_per_kwh") is not None}
    winner, value = _max_item(eligible)
    confidence = _prediction_confidence(len(purchases), periods)
    limits = ["Predicted prices are estimates and are not guaranteed."]
    if confidence == "insufficient_data":
        limits.insert(0, "Insufficient historical completed-price data for a reliable forecast.")
    return {"metric": "realized_price_per_kwh", "model": model, "used_backend_monthly_price_trend": use_monthly, "predictions_by_source": results, "predicted_highest_price_source": winner, "predicted_highest_price_per_kwh": _round(value, 8), "historical_purchase_records": len(purchases), "historical_periods": periods}, confidence, limits


def _predict_shortage(listings, active_listings, purchases, plan) -> Tuple[Dict[str, Any], str, List[str]]:
    source = _plan_source(plan)
    location = _plan_location(plan)
    if not source:
        return {"energy_source": None, "location": location, "shortage_expected": None}, "insufficient_data", ["A specific energy source is required for shortage prediction."]
    filtered_listings = _filter_records(listings, source, location)
    filtered_active = _filter_records(active_listings, source, location)
    filtered_purchases = _filter_records(purchases, source, location)
    periods = _forecast_periods(plan)
    supply_values = [value for _, value in _weekly_energy_by_source(filtered_listings, "created_at")[source]]
    demand_values = [value for _, value in _weekly_energy_by_source(filtered_purchases, "completed_at")[source]]
    supply_forecast = _recursive_forecast(supply_values, periods, False)
    demand_forecast = _recursive_forecast(demand_values, periods, False)
    predicted_supply = supply_forecast.get("predicted_forecast_period_kwh")
    predicted_demand = demand_forecast.get("predicted_forecast_period_kwh")
    active_inventory = sum(float(record["energy_kwh"]) for record in filtered_active)
    gap = predicted_supply + active_inventory - predicted_demand if predicted_supply is not None and predicted_demand is not None else None
    confidence = _prediction_confidence(len(filtered_purchases), min(len(supply_values), len(demand_values)))
    limits = ["Projected shortage or surplus is an estimate, not a guarantee."]
    if confidence == "insufficient_data":
        limits.insert(0, "Insufficient matching location and source history for a reliable shortage forecast.")
    return {"metric": "projected_market_balance_kwh", "energy_source": source, "location": location, "forecast_weeks": periods, "predicted_new_supply_kwh": _round(predicted_supply), "current_active_supply_kwh": _round(active_inventory), "predicted_demand_kwh": _round(predicted_demand), "projected_gap_kwh": _round(gap), "shortage_expected": gap < 0 if gap is not None else None, "model": "recursive_four_period_weighted_moving_average"}, confidence, limits


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def _seller_recommendation(listings, active_listings, purchases) -> Tuple[Dict[str, Any], str, List[str]]:
    supply = _sum_energy_by_source(listings)
    active_supply = _sum_energy_by_source(active_listings)
    demand = _sum_energy_by_source(purchases)
    prices = {source: _weighted_average_price([record for record in purchases if record.get("energy_source") == source]) or 0.0 for source in SUPPORTED_SOURCES}
    growth = _recent_growth_by_source(purchases, "completed_at")
    ratios = {source: demand[source] / supply[source] if supply[source] > 0 else 0.0 for source in SUPPORTED_SOURCES}
    growth_scores = _normalize_metric(growth, True)
    ratio_scores = _normalize_metric(ratios, True)
    price_scores = _normalize_metric(prices, True)
    saturation_scores = _normalize_metric(active_supply, False)
    scores = {}
    factors = {}
    for source in SUPPORTED_SOURCES:
        scores[source] = _round(0.35 * growth_scores[source] + 0.30 * ratio_scores[source] + 0.20 * price_scores[source] + 0.15 * saturation_scores[source], 4)
        factors[source] = {"demand_growth": _round(growth[source], 4), "demand_supply_ratio": _round(ratios[source], 4), "weighted_realized_price": _round(prices[source], 8), "active_supply_kwh": _round(active_supply[source])}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    no_preference = len(ranked) >= 2 and abs(ranked[0][1] - ranked[1][1]) < settings.ANALYTICS_RECOMMENDATION_TIE_THRESHOLD
    periods = max((len(_weekly_energy_by_source(purchases, "completed_at")[source]) for source in SUPPORTED_SOURCES), default=0)
    confidence = _prediction_confidence(len(purchases), periods)
    limits = ["Recommendation is decision support and does not guarantee a sale."]
    if confidence == "insufficient_data":
        limits.insert(0, "Insufficient completed-purchase history for a reliable listing recommendation.")
    recommendation = None if no_preference or confidence == "insufficient_data" else (ranked[0][0] if ranked else None)
    return {"recommended_source": recommendation, "no_strong_preference": no_preference, "scores_by_source": scores, "factors_by_source": factors, "ranking": [source for source, _ in ranked]}, confidence, limits


def _buyer_recommendation(active_listings, purchases) -> Tuple[Dict[str, Any], str, List[str]]:
    if not active_listings:
        return {"recommended_listing": None, "ranked_listings": []}, "insufficient_data", ["No active listings matched the requested filters."]
    historical_demand = _sum_energy_by_source(purchases)
    demand_scores = _normalize_metric(historical_demand, True)
    prices = [float(record["price_per_kwh"]) for record in active_listings if record.get("price_per_kwh") is not None]
    quantities = [float(record["energy_kwh"]) for record in active_listings]
    freshest = max((record.get("created_at") for record in active_listings if isinstance(record.get("created_at"), datetime)), default=None)
    ranked = []
    for record in active_listings:
        price = record.get("price_per_kwh")
        price_score = _single_normalized(float(price), prices, False) if price is not None else 0.0
        quantity_score = _single_normalized(float(record["energy_kwh"]), quantities, True)
        demand_score = demand_scores.get(str(record.get("energy_source")), 0.0)
        recency_score = _recency_score(record.get("created_at"), freshest)
        total_score = 0.40 * price_score + 0.30 * demand_score + 0.20 * quantity_score + 0.10 * recency_score
        ranked.append({"listing_id": record.get("id"), "credit_reference": record.get("credit_reference"), "energy_source": record.get("energy_source"), "location": record.get("location"), "energy_kwh": _round(float(record["energy_kwh"])), "price_per_kwh": _round(float(price), 8) if price is not None else None, "score": _round(total_score, 4), "score_components": {"price_attractiveness": _round(price_score, 4), "historical_demand": _round(demand_score, 4), "quantity_suitability": _round(quantity_score, 4), "listing_recency": _round(recency_score, 4)}})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    confidence = _prediction_confidence(len(purchases), max((len(_weekly_energy_by_source(purchases, "completed_at")[source]) for source in SUPPORTED_SOURCES), default=0))
    limits = ["Recommendation is based on marketplace data and is not financial advice."]
    if confidence == "insufficient_data":
        limits.insert(0, "Insufficient completed-purchase history for a predicted-demand recommendation.")
    return {"recommended_listing": ranked[0] if ranked and confidence != "insufficient_data" else None, "ranked_listings": ranked[:5]}, confidence, limits


# ---------------------------------------------------------------------------
# Time-series, confidence, and generic helpers
# ---------------------------------------------------------------------------


def _weekly_energy_by_source(records, timestamp_field):
    return _weekly_series(records, timestamp_field, "energy_kwh", False)


def _weekly_series(records, timestamp_field, value_field, weighted_price):
    buckets = {source: defaultdict(list) for source in SUPPORTED_SOURCES}
    for record in records:
        source, timestamp = record.get("energy_source"), record.get(timestamp_field)
        if source not in buckets or not isinstance(timestamp, datetime):
            continue
        week = timestamp.date() - timedelta(days=timestamp.weekday())
        buckets[source][week].append(record)
    output = {source: [] for source in SUPPORTED_SOURCES}
    for source in SUPPORTED_SOURCES:
        if not buckets[source]:
            continue
        cursor, end = min(buckets[source]), max(buckets[source])
        while cursor <= end:
            week_records = buckets[source].get(cursor, [])
            if weighted_price:
                value = _weighted_average_price(week_records)
                if value is not None:
                    output[source].append((cursor.isoformat(), value))
            else:
                output[source].append((cursor.isoformat(), sum(float(record.get(value_field) or 0.0) for record in week_records)))
            cursor += timedelta(days=7)
    return output


def _recursive_forecast(values: Sequence[float], periods: int, price: bool) -> Dict[str, Any]:
    key = "predicted_forecast_period_price_per_kwh" if price else "predicted_forecast_period_kwh"
    if len(values) < 2:
        return {key: None, "lower_bound": None, "upper_bound": None, "periods_used": len(values), "reason": "At least two historical periods are required."}
    base_weights = [0.1, 0.2, 0.3, 0.4]
    working = list(values)
    residuals = []
    for index in range(2, len(values)):
        prior = values[max(0, index - 4):index]
        weights = base_weights[-len(prior):]
        estimate = sum(value * weight for value, weight in zip(prior, weights)) / sum(weights)
        residuals.append(values[index] - estimate)
    forecasts = []
    for _ in range(max(1, periods)):
        recent = working[-4:]
        weights = base_weights[-len(recent):]
        forecast = sum(value * weight for value, weight in zip(recent, weights)) / sum(weights)
        forecasts.append(forecast)
        working.append(forecast)
    predicted = statistics.fmean(forecasts) if price else sum(forecasts)
    error = statistics.pstdev(residuals) if len(residuals) >= 2 else statistics.pstdev(values[-4:])
    margin = 1.28 * error * (1.0 if price else math.sqrt(max(1, periods)))
    digits = 8 if price else 2
    return {key: _round(predicted, digits), "lower_bound": _round(max(0.0, predicted - margin), digits), "upper_bound": _round(predicted + margin, digits), "periods_used": len(values), "forecast_periods": max(1, periods)}


def _parse_monthly_price_trend(value: Any) -> Dict[str, List[Tuple[str, float]]]:
    output = {source: [] for source in SUPPORTED_SOURCES}
    if not isinstance(value, Mapping):
        return output
    for outer_key, nested in value.items():
        if not isinstance(nested, Mapping):
            continue
        outer_source = _source(str(outer_key))
        if outer_source:
            for month, stats in nested.items():
                if isinstance(stats, Mapping):
                    price = _number(stats.get("avg_price") or stats.get("avg_price_per_kwh"))
                    if price is not None:
                        output[outer_source].append((str(month), price))
        else:
            for source_key, stats in nested.items():
                source = _source(str(source_key))
                if source and isinstance(stats, Mapping):
                    price = _number(stats.get("avg_price") or stats.get("avg_price_per_kwh"))
                    if price is not None:
                        output[source].append((str(outer_key), price))
    for source in SUPPORTED_SOURCES:
        output[source] = sorted(dict(output[source]).items())
    return output


def _forecast_periods(plan):
    period = plan.get("forecast_period", {})
    if isinstance(period, Mapping) and period.get("from") and period.get("to"):
        try:
            start = date.fromisoformat(str(period["from"])[:10])
            end = date.fromisoformat(str(period["to"])[:10])
            return max(1, math.ceil(((end - start).days + 1) / 7))
        except ValueError:
            pass
    return 4


def _weighted_average_price(records):
    numerator = denominator = 0.0
    for record in records:
        price, energy = _number(record.get("price_per_kwh")), _number(record.get("energy_kwh"))
        if price is None or energy is None or price <= 0 or energy <= 0:
            continue
        numerator += price * energy
        denominator += energy
    return numerator / denominator if denominator else None


def _coefficient_of_variation(values):
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2 or statistics.fmean(clean) == 0:
        return None
    return _round(statistics.pstdev(clean) / statistics.fmean(clean), 4)


def _recent_growth_by_source(records, timestamp_field):
    weekly = _weekly_energy_by_source(records, timestamp_field)
    result = {source: 0.0 for source in SUPPORTED_SOURCES}
    for source in SUPPORTED_SOURCES:
        values = [value for _, value in weekly[source]]
        if len(values) >= 2 and values[-2] != 0:
            result[source] = (values[-1] - values[-2]) / values[-2]
    return result


def _prediction_confidence(record_count, period_count):
    min_records, min_periods = settings.ANALYTICS_MIN_PURCHASE_RECORDS, settings.ANALYTICS_MIN_HISTORY_PERIODS
    if record_count < min_records or period_count < min_periods:
        return "insufficient_data"
    if record_count >= min_records * 3 and period_count >= max(12, min_periods * 2):
        return "high"
    if record_count >= min_records * 2 and period_count >= max(8, min_periods):
        return "medium"
    return "low"


def _minimum_prediction_data_message():
    return f"Reliable prediction requires at least {settings.ANALYTICS_MIN_PURCHASE_RECORDS} completed purchases and {settings.ANALYTICS_MIN_HISTORY_PERIODS} historical periods."


def _descriptive_confidence(records, limitations, aggregates=None):
    if not records and not aggregates:
        return "insufficient_data"
    if limitations:
        return "low"
    return "high" if len(records) >= 30 or bool(aggregates) else "medium"


def _combined_descriptive_confidence(datasets, limitations, aggregates):
    relevant = list(datasets.get("all_listings", [])) + list(datasets.get("purchases", []))
    return _descriptive_confidence(relevant, limitations, any(bool(value) for value in aggregates.values()))


def _time_series_confidence(records, timestamp_field, limitations):
    periods = len({(timestamp.isocalendar().year, timestamp.isocalendar().week) for timestamp in (record.get(timestamp_field) for record in records) if isinstance(timestamp, datetime)})
    if periods < settings.ANALYTICS_MIN_HISTORY_PERIODS:
        return "insufficient_data"
    return "low" if limitations else "high" if periods >= 12 else "medium"


def _tool_limitations(tool_results):
    limitations = []
    for result in tool_results:
        status, tool = str(result.get("execution_status", "")), str(result.get("tool", "unknown"))
        if status == "failed": limitations.append(f"{tool} failed and did not contribute data.")
        elif status == "partial": limitations.append(f"{tool} returned partial data, so calculated totals may be incomplete.")
        elif status == "empty": limitations.append(f"{tool} returned no matching records or aggregate data.")
    return limitations


def _sum_energy_by_source(records):
    totals = {source: 0.0 for source in SUPPORTED_SOURCES}
    for record in records:
        source, energy = record.get("energy_source"), _number(record.get("energy_kwh"))
        if source in totals and energy is not None and energy > 0:
            totals[source] += energy
    return {source: _round(value) or 0.0 for source, value in totals.items()}


def _source(value):
    if not isinstance(value, str): return None
    raw = value.strip(); direct = raw.split(".")[-1].upper().replace("-", "_").replace(" ", "_")
    return direct if direct in SUPPORTED_SOURCES else SOURCE_ALIASES.get(raw.lower().replace("-", "_")) or SOURCE_ALIASES.get(raw.lower())


def _number(value):
    if value is None or isinstance(value, bool): return None
    try: number = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError): return None
    return number if math.isfinite(number) else None


def _datetime(value):
    if isinstance(value, datetime): dt = value
    elif isinstance(value, date): dt = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str) and value.strip():
        try: dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError: return None
    else: return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _normalize_location(value):
    return " ".join(value.strip().lower().split()) if isinstance(value, str) and value.strip() else None


def _deduplicate(records):
    seen, output = set(), []
    for index, raw in enumerate(records):
        record = dict(raw); stable_id = record.get("id") or record.get("purchase_id") or record.get("listing_id")
        fingerprint = str(stable_id) if stable_id is not None else f"missing-{index}-{record}"
        if fingerprint not in seen: seen.add(fingerprint); output.append(record)
    return output


def _filter_records(records, source=None, location=None):
    normalized_location = _normalize_location(location)
    return [record for record in records if (not source or record.get("energy_source") == source) and (not normalized_location or (record.get("location") and normalized_location in str(record.get("location"))))]


def _plan_source(plan):
    for call in plan.get("tool_calls", []) if isinstance(plan.get("tool_calls"), list) else []:
        if isinstance(call, Mapping):
            arguments = call.get("arguments", {})
            if isinstance(arguments, Mapping):
                source = _source(arguments.get("energy_source"))
                if source: return source
    return None


def _plan_location(plan):
    for call in plan.get("tool_calls", []) if isinstance(plan.get("tool_calls"), list) else []:
        if isinstance(call, Mapping):
            arguments = call.get("arguments", {})
            if isinstance(arguments, Mapping) and isinstance(arguments.get("location"), str) and arguments["location"].strip(): return arguments["location"].strip()
    return None


def _normalize_metric(values, higher_is_better):
    numeric = {source: float(values.get(source) or 0.0) for source in SUPPORTED_SOURCES}
    low, high = min(numeric.values()), max(numeric.values())
    if math.isclose(low, high): return {source: 0.5 for source in SUPPORTED_SOURCES}
    return {source: ((value - low) / (high - low)) if higher_is_better else 1.0 - ((value - low) / (high - low)) for source, value in numeric.items()}


def _single_normalized(value, population, higher_is_better):
    if not population or math.isclose(min(population), max(population)): return 0.5
    score = (value - min(population)) / (max(population) - min(population))
    return score if higher_is_better else 1.0 - score


def _recency_score(created_at, freshest):
    if not isinstance(created_at, datetime) or freshest is None: return 0.5
    return max(0.0, 1.0 - min(max(0, (freshest - created_at).days), 30) / 30.0)


def _max_item(values):
    valid = [(key, float(value)) for key, value in values.items() if value is not None]
    return max(valid, key=lambda item: item[1]) if valid else (None, None)


def _positive_max_item(values):
    return _max_item({key: value for key, value in values.items() if value is not None and float(value) > 0})


def _min_item(values):
    valid = [(key, float(value)) for key, value in values.items() if value is not None]
    return min(valid, key=lambda item: item[1]) if valid else (None, None)


def _round(value, digits=2):
    return round(float(value), digits) if value is not None else None


def _generic_dataset_summary(datasets):
    return {"all_listings_count": len(datasets.get("all_listings", [])), "active_listings_count": len(datasets.get("active_listings", [])), "completed_purchases_count": len(datasets.get("purchases", []))}


def _unique_strings(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, Decimal): return float(value)
    if isinstance(value, (datetime, date)): return value.isoformat()
    if isinstance(value, Mapping): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)): return [_json_safe(item) for item in value]
    return str(value)
