"""
analytics_service.py
--------------------

Deterministic analytics, forecasting, and recommendation engine for GreenGrid
Exchange.

This module does not call APIs, OpenSearch, or Amazon Bedrock. It receives the
validated tool plan and normalized marketplace records, performs calculations,
and returns compact structured results that the LLM can explain.

Supported renewable sources:
- SOLAR
- WIND
- HYDRO

Supported intents:
- current_supply
- supply_mix
- historical_supply
- historical_demand
- average_selling_price
- demand_supply_ratio
- market_balance
- supply_stability
- price_volatility
- demand_prediction
- price_prediction
- shortage_prediction
- seller_recommendation
- buyer_recommendation
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

SUPPORTED_SOURCES: Tuple[str, ...] = ("SOLAR", "WIND", "HYDRO")
COMPLETED_PURCHASE_STATUSES = {"completed", "consumed"}
ACTIVE_LISTING_STATUS = "active"

SOURCE_ALIASES = {
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
    # Backward-compatible input normalization only.
    "small_hydro": "HYDRO",
    "small hydro": "HYDRO",
    "small-hydro": "HYDRO",
}

PREDICTION_INTENTS = {
    "demand_prediction",
    "price_prediction",
    "shortage_prediction",
}

RECOMMENDATION_INTENTS = {
    "seller_recommendation",
    "buyer_recommendation",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_plan(
    plan: Mapping[str, Any],
    tool_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Execute deterministic analysis for a planner intent.

    Parameters
    ----------
    plan:
        Validated result from ``bedrock_service.plan_api_calls``.
    tool_results:
        Results returned by ``marketplace_api_service.execute_tool_call``.

    Returns
    -------
    dict
        Compact analytics context suitable for ``QueryApiSummary`` and the
        final answer prompt.
    """
    intent = str(plan.get("intent", "none") or "none").strip().lower()
    datasets = _extract_datasets(tool_results)
    records_analyzed = {
        "all_listings": len(datasets["all_listings"]),
        "active_listings": len(datasets["active_listings"]),
        "purchases": len(datasets["purchases"]),
    }
    limitations = _tool_limitations(tool_results)

    analytics_result: Dict[str, Any] = {}
    prediction_result: Optional[Dict[str, Any]] = None
    recommendation_result: Optional[Dict[str, Any]] = None
    confidence: Optional[str] = None
    calculation_method: Optional[str] = None

    try:
        if intent == "current_supply":
            analytics_result = _current_supply(datasets["active_listings"])
            calculation_method = "sum of active energy_kwh grouped by energy_source"
            confidence = _descriptive_confidence(datasets["active_listings"], limitations)

        elif intent == "supply_mix":
            analytics_result = _supply_mix(datasets["active_listings"])
            calculation_method = "active source kWh divided by total active kWh"
            confidence = _descriptive_confidence(datasets["active_listings"], limitations)

        elif intent == "historical_supply":
            analytics_result = _historical_supply(datasets["all_listings"])
            calculation_method = "sum of listed energy_kwh grouped by energy_source"
            confidence = _descriptive_confidence(datasets["all_listings"], limitations)

        elif intent == "historical_demand":
            analytics_result = _historical_demand(datasets["purchases"])
            calculation_method = "sum of completed purchase energy_kwh grouped by energy_source"
            confidence = _descriptive_confidence(datasets["purchases"], limitations)

        elif intent == "average_selling_price":
            analytics_result = _average_selling_price(datasets["purchases"])
            calculation_method = "volume-weighted realized price by energy_source"
            confidence = _descriptive_confidence(datasets["purchases"], limitations)

        elif intent == "demand_supply_ratio":
            analytics_result = _demand_supply_ratio(
                datasets["all_listings"], datasets["purchases"]
            )
            calculation_method = "completed purchase kWh divided by listed kWh"
            confidence = _combined_descriptive_confidence(datasets, limitations)

        elif intent == "market_balance":
            analytics_result = _market_balance(
                datasets["all_listings"], datasets["purchases"]
            )
            calculation_method = "listed supply kWh minus completed demand kWh"
            confidence = _combined_descriptive_confidence(datasets, limitations)

        elif intent == "supply_stability":
            analytics_result = _supply_stability(datasets["all_listings"])
            calculation_method = "weekly listed-supply coefficient of variation"
            confidence = _time_series_confidence(
                datasets["all_listings"], "created_at", limitations
            )
            limitations.append(
                "Supply stability measures newly listed weekly supply, not historical daily active inventory."
            )

        elif intent == "price_volatility":
            analytics_result = _price_volatility(datasets["purchases"])
            calculation_method = "weekly weighted realized-price coefficient of variation"
            confidence = _time_series_confidence(
                datasets["purchases"], "completed_at", limitations
            )

        elif intent == "demand_prediction":
            analytics_result = _historical_demand(datasets["purchases"])
            prediction_result, confidence, extra_limits = _predict_demand(
                datasets["purchases"]
            )
            limitations.extend(extra_limits)
            calculation_method = "four-period weighted moving average with historical residual bounds"

        elif intent == "price_prediction":
            analytics_result = _average_selling_price(datasets["purchases"])
            prediction_result, confidence, extra_limits = _predict_price(
                datasets["purchases"]
            )
            limitations.extend(extra_limits)
            calculation_method = "four-period weighted moving average of weekly realized prices"

        elif intent == "shortage_prediction":
            analytics_result = {
                "historical_market_balance": _market_balance(
                    datasets["all_listings"], datasets["purchases"]
                ),
                "current_active_supply": _current_supply(
                    datasets["active_listings"]
                ),
            }
            prediction_result, confidence, extra_limits = _predict_shortage(
                listings=datasets["all_listings"],
                active_listings=datasets["active_listings"],
                purchases=datasets["purchases"],
                plan=plan,
            )
            limitations.extend(extra_limits)
            calculation_method = (
                "forecast newly listed supply plus current active inventory "
                "minus forecast completed demand"
            )

        elif intent == "seller_recommendation":
            analytics_result = {
                "current_supply": _current_supply(datasets["active_listings"]),
                "historical_demand": _historical_demand(datasets["purchases"]),
                "demand_supply_ratio": _demand_supply_ratio(
                    datasets["all_listings"], datasets["purchases"]
                ),
                "average_selling_price": _average_selling_price(
                    datasets["purchases"]
                ),
            }
            recommendation_result, confidence, extra_limits = _seller_recommendation(
                datasets["all_listings"],
                datasets["active_listings"],
                datasets["purchases"],
            )
            limitations.extend(extra_limits)
            calculation_method = (
                "weighted opportunity score: demand growth 35%, demand-to-supply "
                "ratio 30%, realized-price strength 20%, low supply saturation 15%"
            )

        elif intent == "buyer_recommendation":
            analytics_result = {
                "current_supply": _current_supply(datasets["active_listings"]),
                "historical_demand": _historical_demand(datasets["purchases"]),
            }
            recommendation_result, confidence, extra_limits = _buyer_recommendation(
                datasets["active_listings"], datasets["purchases"]
            )
            limitations.extend(extra_limits)
            calculation_method = (
                "listing score: price attractiveness 40%, historical demand 30%, "
                "quantity suitability 20%, listing recency 10%"
            )

        else:
            analytics_result = _generic_dataset_summary(datasets)
            calculation_method = "record summary"
            confidence = _combined_descriptive_confidence(datasets, limitations)

    except Exception as exc:  # defensive boundary for the chatbot pipeline
        logger.exception("Analytics processing failed for intent=%s", intent)
        limitations.append(f"Analytics calculation failed: {type(exc).__name__}")
        confidence = "insufficient_data"
        analytics_result = {}
        prediction_result = None
        recommendation_result = None

    limitations = _unique_strings(limitations)

    return {
        "intent": intent,
        "analytics_result": _json_safe(analytics_result),
        "prediction_result": _json_safe(prediction_result) if prediction_result else None,
        "recommendation_result": (
            _json_safe(recommendation_result) if recommendation_result else None
        ),
        "confidence": confidence,
        "calculation_method": calculation_method,
        "limitations": limitations,
        "records_analyzed": records_analyzed,
    }


# ---------------------------------------------------------------------------
# Dataset extraction and validation
# ---------------------------------------------------------------------------


def _extract_datasets(
    tool_results: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    datasets: Dict[str, List[Dict[str, Any]]] = {
        "all_listings": [],
        "active_listings": [],
        "purchases": [],
    }

    for result in tool_results:
        tool = str(result.get("tool", ""))
        records = _records_from_result(result)
        if tool == "get_all_listings":
            datasets["all_listings"].extend(_clean_listings(records))
        elif tool == "get_active_listings":
            datasets["active_listings"].extend(_clean_listings(records, active_only=True))
        elif tool == "get_all_purchases":
            datasets["purchases"].extend(_clean_purchases(records))

    for key, records in datasets.items():
        datasets[key] = _deduplicate(records)

    return datasets


def _records_from_result(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data = result.get("data", {})
    if not isinstance(data, Mapping):
        return []
    records = data.get("records", [])
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _clean_listings(
    records: Sequence[Mapping[str, Any]], active_only: bool = False
) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for raw in records:
        source = _source(raw.get("energy_source") or raw.get("source"))
        energy = _number(raw.get("energy_kwh"))
        price = _number(raw.get("price_per_kwh"))
        status = str(raw.get("status", "")).strip().lower()
        is_available = raw.get("is_available")
        if source is None or energy is None or energy <= 0:
            continue
        if active_only and not (
            status == ACTIVE_LISTING_STATUS
            and (is_available is True or is_available is None)
        ):
            continue
        record = dict(raw)
        record["energy_source"] = source
        record["energy_kwh"] = energy
        record["price_per_kwh"] = price
        record["status"] = status
        record["location"] = _normalize_location(raw.get("location"))
        record["created_at"] = _datetime(raw.get("created_at"))
        record["expires_at"] = _datetime(raw.get("expires_at"))
        cleaned.append(record)
    return cleaned


def _clean_purchases(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for raw in records:
        source = _source(raw.get("energy_source") or raw.get("source"))
        energy = _number(raw.get("energy_kwh"))
        price = _number(raw.get("price_per_kwh"))
        status = str(raw.get("status", "")).strip().lower()
        completed_at = _datetime(raw.get("completed_at"))
        if source is None or energy is None or energy <= 0:
            continue
        if status not in COMPLETED_PURCHASE_STATUSES or completed_at is None:
            continue
        record = dict(raw)
        record["energy_source"] = source
        record["energy_kwh"] = energy
        record["price_per_kwh"] = price
        record["status"] = status
        record["location"] = _normalize_location(raw.get("location"))
        record["completed_at"] = completed_at
        cleaned.append(record)
    return cleaned


# ---------------------------------------------------------------------------
# Descriptive analytics
# ---------------------------------------------------------------------------


def _current_supply(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals = _sum_energy_by_source(records)
    winner, winner_value = _max_item(totals)
    return {
        "supply_by_source_kwh": totals,
        "total_active_supply_kwh": _round(sum(totals.values())),
        "highest_supply_source": winner,
        "highest_supply_kwh": _round(winner_value) if winner_value is not None else None,
        "listing_count": len(records),
    }


def _supply_mix(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals = _sum_energy_by_source(records)
    grand_total = sum(totals.values())
    percentages = {
        source: _round((value / grand_total) * 100, 2) if grand_total else 0.0
        for source, value in totals.items()
    }
    return {
        "active_supply_kwh_by_source": totals,
        "total_active_supply_kwh": _round(grand_total),
        "supply_mix_percentage": percentages,
        "basis": "active energy_kwh, not listing count",
    }


def _historical_supply(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals = _sum_energy_by_source(records)
    winner, value = _max_item(totals)
    return {
        "listed_supply_kwh_by_source": totals,
        "total_listed_supply_kwh": _round(sum(totals.values())),
        "highest_listed_supply_source": winner,
        "highest_listed_supply_kwh": _round(value) if value is not None else None,
        "listing_count": len(records),
    }


def _historical_demand(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals = _sum_energy_by_source(records)
    grand_total = sum(totals.values())
    percentages = {
        source: _round((value / grand_total) * 100, 2) if grand_total else 0.0
        for source, value in totals.items()
    }
    counts = {source: 0 for source in SUPPORTED_SOURCES}
    for record in records:
        source = record.get("energy_source")
        if source in counts:
            counts[source] += 1
    winner, value = _max_item(totals)
    return {
        "demand_kwh_by_source": totals,
        "demand_share_percentage": percentages,
        "purchase_count_by_source": counts,
        "total_completed_demand_kwh": _round(grand_total),
        "highest_demand_source": winner,
        "highest_demand_kwh": _round(value) if value is not None else None,
    }


def _average_selling_price(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prices: Dict[str, Optional[float]] = {}
    volumes: Dict[str, float] = {}
    for source in SUPPORTED_SOURCES:
        source_records = [r for r in records if r.get("energy_source") == source]
        prices[source] = _weighted_average_price(source_records)
        volumes[source] = _round(sum(float(r["energy_kwh"]) for r in source_records)) # type: ignore
    valid = {source: value for source, value in prices.items() if value is not None}
    winner, value = _max_item(valid)
    return {
        "weighted_average_selling_price_by_source": {
            source: _round(value, 8) if value is not None else None
            for source, value in prices.items()
        },
        "sold_volume_kwh_by_source": volumes,
        "highest_average_price_source": winner,
        "highest_average_price_per_kwh": _round(value, 8) if value is not None else None,
        "price_basis": "completed purchase volume-weighted realized price",
    }


def _demand_supply_ratio(
    listings: Sequence[Mapping[str, Any]],
    purchases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    supply = _sum_energy_by_source(listings)
    demand = _sum_energy_by_source(purchases)
    ratios: Dict[str, Optional[float]] = {}
    for source in SUPPORTED_SOURCES:
        ratios[source] = (
            _round(demand[source] / supply[source], 4) if supply[source] > 0 else None
        )
    valid = {source: value for source, value in ratios.items() if value is not None}
    winner, value = _max_item(valid)
    return {
        "listed_supply_kwh_by_source": supply,
        "completed_demand_kwh_by_source": demand,
        "demand_supply_ratio_by_source": ratios,
        "highest_ratio_source": winner,
        "highest_ratio": _round(value, 4) if value is not None else None,
    }


def _market_balance(
    listings: Sequence[Mapping[str, Any]],
    purchases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    supply = _sum_energy_by_source(listings)
    demand = _sum_energy_by_source(purchases)
    balances: Dict[str, float] = {}
    category: Dict[str, str] = {}
    for source in SUPPORTED_SOURCES:
        value = supply[source] - demand[source]
        balances[source] = _round(value) # type: ignore
        category[source] = "surplus" if value > 0 else "shortage" if value < 0 else "balanced"
    return {
        "listed_supply_kwh_by_source": supply,
        "completed_demand_kwh_by_source": demand,
        "market_balance_kwh_by_source": balances,
        "market_condition_by_source": category,
    }


def _supply_stability(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    weekly = _weekly_series(records, "created_at", "energy_kwh", weighted_price=False)
    coefficients: Dict[str, Optional[float]] = {}
    weekly_values: Dict[str, List[Dict[str, Any]]] = {}
    for source in SUPPORTED_SOURCES:
        values = weekly[source]
        weekly_values[source] = [
            {"week": week, "listed_kwh": _round(value)} for week, value in values
        ]
        coefficients[source] = _coefficient_of_variation([value for _, value in values])
    valid = {source: value for source, value in coefficients.items() if value is not None}
    winner, value = _min_item(valid)
    return {
        "weekly_listed_supply": weekly_values,
        "supply_stability_coefficient_by_source": coefficients,
        "most_stable_supply_source": winner,
        "lowest_stability_coefficient": _round(value, 4) if value is not None else None,
    }


def _price_volatility(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    weekly = _weekly_series(records, "completed_at", "price_per_kwh", weighted_price=True)
    coefficients: Dict[str, Optional[float]] = {}
    weekly_values: Dict[str, List[Dict[str, Any]]] = {}
    for source in SUPPORTED_SOURCES:
        values = weekly[source]
        weekly_values[source] = [
            {"week": week, "weighted_price_per_kwh": _round(value, 8)}
            for week, value in values
        ]
        coefficients[source] = _coefficient_of_variation([value for _, value in values])
    valid = {source: value for source, value in coefficients.items() if value is not None}
    winner, value = _max_item(valid)
    return {
        "weekly_weighted_realized_prices": weekly_values,
        "price_volatility_coefficient_by_source": coefficients,
        "highest_price_volatility_source": winner,
        "highest_volatility_coefficient": _round(value, 4) if value is not None else None,
    }


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def _predict_demand(
    purchases: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], str, List[str]]:
    limits: List[str] = []
    weekly = _weekly_energy_by_source(purchases, "completed_at")
    results: Dict[str, Any] = {}
    all_completed = len(purchases)

    for source in SUPPORTED_SOURCES:
        values = [value for _, value in weekly[source]]
        forecast = _weighted_moving_average_forecast(values)
        results[source] = forecast

    eligible = {
        source: result["predicted_next_period_kwh"]
        for source, result in results.items()
        if result.get("predicted_next_period_kwh") is not None
    }
    winner, winner_value = _max_item(eligible)
    periods = max((len(weekly[source]) for source in SUPPORTED_SOURCES), default=0)
    confidence = _prediction_confidence(all_completed, periods)

    if confidence == "insufficient_data":
        limits.append(
            f"Reliable prediction requires at least {settings.ANALYTICS_MIN_PURCHASE_RECORDS} "
            f"completed purchases and {settings.ANALYTICS_MIN_HISTORY_PERIODS} historical periods."
        )
    limits.append("Future marketplace demand is not guaranteed.")

    return {
        "metric": "completed_demand_kwh",
        "model": "four_period_weighted_moving_average",
        "predictions_by_source": results,
        "predicted_highest_demand_source": winner,
        "predicted_highest_demand_kwh": _round(winner_value) if winner_value is not None else None,
        "historical_purchase_records": all_completed,
        "historical_periods": periods,
    }, confidence, limits


def _predict_price(
    purchases: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], str, List[str]]:
    limits: List[str] = []
    weekly = _weekly_series(purchases, "completed_at", "price_per_kwh", weighted_price=True)
    results: Dict[str, Any] = {}

    for source in SUPPORTED_SOURCES:
        values = [value for _, value in weekly[source]]
        forecast = _weighted_moving_average_forecast(values, price=True)
        results[source] = forecast

    eligible = {
        source: result["predicted_next_period_price_per_kwh"]
        for source, result in results.items()
        if result.get("predicted_next_period_price_per_kwh") is not None
    }
    winner, winner_value = _max_item(eligible)
    periods = max((len(weekly[source]) for source in SUPPORTED_SOURCES), default=0)
    confidence = _prediction_confidence(len(purchases), periods)

    if confidence == "insufficient_data":
        limits.append("Insufficient historical completed-price data for a reliable forecast.")
    limits.append("Predicted prices are estimates and are not guaranteed.")

    return {
        "metric": "realized_price_per_kwh",
        "model": "four_period_weighted_moving_average",
        "predictions_by_source": results,
        "predicted_highest_price_source": winner,
        "predicted_highest_price_per_kwh": _round(winner_value, 8) if winner_value is not None else None,
        "historical_purchase_records": len(purchases),
        "historical_periods": periods,
    }, confidence, limits


def _predict_shortage(
    listings: Sequence[Mapping[str, Any]],
    active_listings: Sequence[Mapping[str, Any]],
    purchases: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str, List[str]]:
    limits: List[str] = ["Projected shortage or surplus is an estimate, not a guarantee."]
    location = _plan_location(plan)
    source = _plan_source(plan) or "SOLAR"

    filtered_listings = _filter_records(listings, source=source, location=location)
    filtered_active = _filter_records(active_listings, source=source, location=location)
    filtered_purchases = _filter_records(purchases, source=source, location=location)

    supply_weekly = _weekly_energy_by_source(filtered_listings, "created_at")[source]
    demand_weekly = _weekly_energy_by_source(filtered_purchases, "completed_at")[source]

    supply_forecast = _weighted_moving_average_forecast([v for _, v in supply_weekly])
    demand_forecast = _weighted_moving_average_forecast([v for _, v in demand_weekly])
    predicted_supply = supply_forecast.get("predicted_next_period_kwh")
    predicted_demand = demand_forecast.get("predicted_next_period_kwh")
    active_inventory = sum(float(r["energy_kwh"]) for r in filtered_active)

    projected_gap: Optional[float] = None
    shortage_expected: Optional[bool] = None
    if predicted_supply is not None and predicted_demand is not None:
        projected_gap = predicted_supply + active_inventory - predicted_demand
        shortage_expected = projected_gap < 0 # type: ignore

    periods = min(len(supply_weekly), len(demand_weekly))
    confidence = _prediction_confidence(len(filtered_purchases), periods)
    if confidence == "insufficient_data":
        limits.append("Insufficient matching location/source history for a reliable shortage forecast.")

    return {
        "metric": "projected_market_balance_kwh",
        "energy_source": source,
        "location": location,
        "predicted_new_supply_kwh": _round(predicted_supply) if predicted_supply is not None else None,
        "current_active_supply_kwh": _round(active_inventory),
        "predicted_demand_kwh": _round(predicted_demand) if predicted_demand is not None else None,
        "projected_gap_kwh": _round(projected_gap) if projected_gap is not None else None,
        "shortage_expected": shortage_expected,
        "supply_model": "four_period_weighted_moving_average",
        "demand_model": "four_period_weighted_moving_average",
    }, confidence, limits


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def _seller_recommendation(
    listings: Sequence[Mapping[str, Any]],
    active_listings: Sequence[Mapping[str, Any]],
    purchases: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], str, List[str]]:
    limits = ["Recommendation is decision support and does not guarantee a sale."]
    supply = _sum_energy_by_source(listings)
    active_supply = _sum_energy_by_source(active_listings)
    demand = _sum_energy_by_source(purchases)
    prices = _average_selling_price(purchases)["weighted_average_selling_price_by_source"]
    growth = _recent_growth_by_source(purchases, "completed_at")

    ratios = {
        source: demand[source] / supply[source] if supply[source] > 0 else 0.0
        for source in SUPPORTED_SOURCES
    }
    demand_growth_scores = _normalize_metric(growth, higher_is_better=True)
    ratio_scores = _normalize_metric(ratios, higher_is_better=True)
    price_scores = _normalize_metric(
        {source: prices.get(source) or 0.0 for source in SUPPORTED_SOURCES},
        higher_is_better=True,
    )
    saturation_scores = _normalize_metric(active_supply, higher_is_better=False)

    scores: Dict[str, float] = {}
    factors: Dict[str, Dict[str, float]] = {}
    for source in SUPPORTED_SOURCES:
        score = (
            0.35 * demand_growth_scores[source]
            + 0.30 * ratio_scores[source]
            + 0.20 * price_scores[source]
            + 0.15 * saturation_scores[source]
        )
        scores[source] = _round(score, 4) # type: ignore
        factors[source] = { # type: ignore
            "demand_growth": _round(growth[source], 4),
            "demand_supply_ratio": _round(ratios[source], 4),
            "weighted_realized_price": prices.get(source),
            "active_supply_kwh": _round(active_supply[source]),
        }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    recommendation: Optional[str] = ranked[0][0] if ranked else None
    no_strong_preference = False
    if len(ranked) >= 2 and abs(ranked[0][1] - ranked[1][1]) < settings.ANALYTICS_RECOMMENDATION_TIE_THRESHOLD:
        recommendation = None
        no_strong_preference = True

    periods = len(_weekly_energy_by_source(purchases, "completed_at")[ranked[0][0]]) if ranked else 0
    confidence = _prediction_confidence(len(purchases), periods)
    if confidence == "insufficient_data":
        limits.append("Insufficient completed-purchase history for a reliable listing recommendation.")

    return {
        "recommended_source": recommendation,
        "no_strong_preference": no_strong_preference,
        "scores_by_source": scores,
        "factors_by_source": factors,
        "ranking": [source for source, _ in ranked],
    }, confidence, limits


def _buyer_recommendation(
    active_listings: Sequence[Mapping[str, Any]],
    purchases: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], str, List[str]]:
    limits = ["Recommendation is based on marketplace data and is not financial advice."]
    if not active_listings:
        return {
            "recommended_listing": None,
            "ranked_listings": [],
        }, "insufficient_data", ["No active listings matched the requested filters."]

    historical_demand = _sum_energy_by_source(purchases)
    demand_scores = _normalize_metric(historical_demand, higher_is_better=True)
    prices = [float(r["price_per_kwh"]) for r in active_listings if r.get("price_per_kwh") is not None]
    quantities = [float(r["energy_kwh"]) for r in active_listings]
    freshest = max(
        (r.get("created_at") for r in active_listings if r.get("created_at") is not None), # type: ignore
        default=None,
    )

    ranked: List[Dict[str, Any]] = []
    for record in active_listings:
        price = record.get("price_per_kwh")
        price_score = _single_normalized(float(price), prices, higher_is_better=False) if price is not None else 0.0
        quantity_score = _single_normalized(float(record["energy_kwh"]), quantities, higher_is_better=True)
        demand_score = demand_scores.get(str(record.get("energy_source")), 0.0)
        recency_score = _recency_score(record.get("created_at"), freshest)
        total_score = (
            0.40 * price_score
            + 0.30 * demand_score
            + 0.20 * quantity_score
            + 0.10 * recency_score
        )
        ranked.append({
            "listing_id": record.get("id"),
            "credit_reference": record.get("credit_reference"),
            "energy_source": record.get("energy_source"),
            "location": record.get("location"),
            "energy_kwh": _round(float(record["energy_kwh"])),
            "price_per_kwh": _round(float(price), 8) if price is not None else None,
            "score": _round(total_score, 4),
            "score_components": {
                "price_attractiveness": _round(price_score, 4),
                "historical_demand": _round(demand_score, 4),
                "quantity_suitability": _round(quantity_score, 4),
                "listing_recency": _round(recency_score, 4),
            },
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    confidence = _descriptive_confidence(purchases, [])
    if not purchases:
        limits.append("No completed purchase history was available; historical-demand scoring is neutral.")

    return {
        "recommended_listing": ranked[0] if ranked else None,
        "ranked_listings": ranked[:5],
    }, confidence, limits


# ---------------------------------------------------------------------------
# Time-series and statistics helpers
# ---------------------------------------------------------------------------


def _weekly_energy_by_source(
    records: Sequence[Mapping[str, Any]], timestamp_field: str
) -> Dict[str, List[Tuple[str, float]]]:
    return _weekly_series(records, timestamp_field, "energy_kwh", weighted_price=False)


def _weekly_series(
    records: Sequence[Mapping[str, Any]],
    timestamp_field: str,
    value_field: str,
    weighted_price: bool,
) -> Dict[str, List[Tuple[str, float]]]:
    buckets: Dict[str, Dict[date, List[Mapping[str, Any]]]] = {
        source: defaultdict(list) for source in SUPPORTED_SOURCES
    }
    for record in records:
        source = record.get("energy_source")
        timestamp = record.get(timestamp_field)
        if source not in buckets or not isinstance(timestamp, datetime):
            continue
        week_start = timestamp.date() - timedelta(days=timestamp.weekday())
        buckets[source][week_start].append(record)

    output: Dict[str, List[Tuple[str, float]]] = {source: [] for source in SUPPORTED_SOURCES}
    for source in SUPPORTED_SOURCES:
        if not buckets[source]:
            continue
        all_weeks = sorted(buckets[source])
        cursor = all_weeks[0]
        end = all_weeks[-1]
        while cursor <= end:
            week_records = buckets[source].get(cursor, [])
            if weighted_price:
                value = _weighted_average_price(week_records)
                if value is not None:
                    output[source].append((cursor.isoformat(), value))
            else:
                value = sum(float(r.get(value_field) or 0.0) for r in week_records)
                output[source].append((cursor.isoformat(), value))
            cursor += timedelta(days=7)
    return output


def _weighted_moving_average_forecast(
    values: Sequence[float], price: bool = False
) -> Dict[str, Any]:
    if len(values) < 2:
        key = "predicted_next_period_price_per_kwh" if price else "predicted_next_period_kwh"
        return {
            key: None,
            "lower_bound": None,
            "upper_bound": None,
            "periods_used": len(values),
            "reason": "At least two historical periods are required.",
        }

    recent = list(values[-4:])
    base_weights = [0.1, 0.2, 0.3, 0.4]
    weights = base_weights[-len(recent):]
    weight_total = sum(weights)
    weights = [weight / weight_total for weight in weights]
    forecast = sum(value * weight for value, weight in zip(recent, weights))

    residuals: List[float] = []
    if len(values) >= 3:
        for index in range(2, len(values)):
            prior = values[max(0, index - 4):index]
            prior_weights = base_weights[-len(prior):]
            total = sum(prior_weights)
            estimate = sum(v * (w / total) for v, w in zip(prior, prior_weights))
            residuals.append(values[index] - estimate)
    error = statistics.pstdev(residuals) if len(residuals) >= 2 else (
        statistics.pstdev(recent) if len(recent) >= 2 else 0.0
    )
    lower = max(0.0, forecast - 1.28 * error)
    upper = max(lower, forecast + 1.28 * error)
    key = "predicted_next_period_price_per_kwh" if price else "predicted_next_period_kwh"
    precision = 8 if price else 2
    return {
        key: _round(forecast, precision),
        "lower_bound": _round(lower, precision),
        "upper_bound": _round(upper, precision),
        "periods_used": len(values),
    }


def _weighted_average_price(records: Sequence[Mapping[str, Any]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        price = _number(record.get("price_per_kwh"))
        energy = _number(record.get("energy_kwh"))
        if price is None or energy is None or price <= 0 or energy <= 0:
            continue
        numerator += price * energy
        denominator += energy
    return numerator / denominator if denominator else None


def _coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    mean = statistics.fmean(clean)
    if mean == 0:
        return None
    return _round(statistics.pstdev(clean) / mean, 4)


def _recent_growth_by_source(
    records: Sequence[Mapping[str, Any]], timestamp_field: str
) -> Dict[str, float]:
    weekly = _weekly_energy_by_source(records, timestamp_field)
    result: Dict[str, float] = {source: 0.0 for source in SUPPORTED_SOURCES}
    for source in SUPPORTED_SOURCES:
        values = [value for _, value in weekly[source]]
        if len(values) < 2 or values[-2] == 0:
            result[source] = 0.0
        else:
            result[source] = (values[-1] - values[-2]) / values[-2]
    return result


# ---------------------------------------------------------------------------
# Confidence and limitations
# ---------------------------------------------------------------------------


def _prediction_confidence(record_count: int, period_count: int) -> str:
    min_records = settings.ANALYTICS_MIN_PURCHASE_RECORDS
    min_periods = settings.ANALYTICS_MIN_HISTORY_PERIODS
    if record_count < min_records or period_count < min_periods:
        return "insufficient_data"
    if record_count >= min_records * 3 and period_count >= max(12, min_periods * 2):
        return "high"
    if record_count >= min_records * 2 and period_count >= max(8, min_periods):
        return "medium"
    return "low"


def _descriptive_confidence(records: Sequence[Any], limitations: Sequence[str]) -> str:
    if not records:
        return "insufficient_data"
    if limitations:
        return "low"
    return "high" if len(records) >= 30 else "medium"


def _combined_descriptive_confidence(
    datasets: Mapping[str, Sequence[Any]], limitations: Sequence[str]
) -> str:
    relevant = list(datasets.get("all_listings", [])) + list(datasets.get("purchases", []))
    return _descriptive_confidence(relevant, limitations)


def _time_series_confidence(
    records: Sequence[Mapping[str, Any]], timestamp_field: str, limitations: Sequence[str]
) -> str:
    timestamps = [r.get(timestamp_field) for r in records if isinstance(r.get(timestamp_field), datetime)]
    periods = len({(ts.isocalendar().year, ts.isocalendar().week) for ts in timestamps}) # type: ignore
    if periods < settings.ANALYTICS_MIN_HISTORY_PERIODS:
        return "insufficient_data"
    if limitations:
        return "low"
    return "high" if periods >= 12 else "medium"


def _tool_limitations(tool_results: Sequence[Mapping[str, Any]]) -> List[str]:
    limitations: List[str] = []
    for result in tool_results:
        status = str(result.get("execution_status", ""))
        tool = str(result.get("tool", "unknown"))
        if status == "failed":
            limitations.append(f"{tool} failed and did not contribute data.")
        elif status == "partial":
            limitations.append(f"{tool} returned partial data, so calculated totals may be incomplete.")
        elif status == "empty":
            limitations.append(f"{tool} returned no matching records.")
    return limitations


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _sum_energy_by_source(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    totals = {source: 0.0 for source in SUPPORTED_SOURCES}
    for record in records:
        source = record.get("energy_source")
        energy = _number(record.get("energy_kwh"))
        if source in totals and energy is not None and energy > 0:
            totals[source] += energy
    return {source: _round(value) for source, value in totals.items()} # type: ignore


def _source(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    direct = raw.upper().replace("-", "_").replace(" ", "_")
    if direct in SUPPORTED_SOURCES:
        return direct
    key = raw.lower().replace("-", "_")
    return SOURCE_ALIASES.get(key) or SOURCE_ALIASES.get(raw.lower())


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            try:
                d = date.fromisoformat(value.strip())
                dt = datetime.combine(d, datetime.min.time())
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_location(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.strip().lower().split())


def _deduplicate(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = dict(raw)
        stable_id = record.get("id") or record.get("purchase_id") or record.get("listing_id")
        fingerprint = str(stable_id) if stable_id is not None else f"missing-{index}-{record}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append(record)
    return output


def _filter_records(
    records: Sequence[Mapping[str, Any]],
    source: Optional[str] = None,
    location: Optional[str] = None,
) -> List[Mapping[str, Any]]:
    normalized_location = _normalize_location(location)
    output = []
    for record in records:
        if source and record.get("energy_source") != source:
            continue
        record_location = _normalize_location(record.get("location"))
        if normalized_location and (
            record_location is None or normalized_location not in record_location
        ):
            continue
        output.append(record)
    return output


def _plan_source(plan: Mapping[str, Any]) -> Optional[str]:
    for call in plan.get("tool_calls", []) if isinstance(plan.get("tool_calls"), list) else []:
        if isinstance(call, Mapping):
            arguments = call.get("arguments", {})
            if isinstance(arguments, Mapping):
                source = _source(arguments.get("energy_source"))
                if source:
                    return source
    return None


def _plan_location(plan: Mapping[str, Any]) -> Optional[str]:
    for call in plan.get("tool_calls", []) if isinstance(plan.get("tool_calls"), list) else []:
        if isinstance(call, Mapping):
            arguments = call.get("arguments", {})
            if isinstance(arguments, Mapping):
                value = arguments.get("location")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _normalize_metric(values: Mapping[str, Any], higher_is_better: bool) -> Dict[str, float]:
    numeric = {source: float(values.get(source) or 0.0) for source in SUPPORTED_SOURCES}
    low, high = min(numeric.values()), max(numeric.values())
    if math.isclose(low, high):
        return {source: 0.5 for source in SUPPORTED_SOURCES}
    output = {}
    for source, value in numeric.items():
        score = (value - low) / (high - low)
        output[source] = score if higher_is_better else 1.0 - score
    return output


def _single_normalized(value: float, population: Sequence[float], higher_is_better: bool) -> float:
    if not population:
        return 0.5
    low, high = min(population), max(population)
    if math.isclose(low, high):
        return 0.5
    score = (value - low) / (high - low)
    return score if higher_is_better else 1.0 - score


def _recency_score(created_at: Any, freshest: Optional[datetime]) -> float:
    if not isinstance(created_at, datetime) or freshest is None:
        return 0.5
    age_days = max(0, (freshest - created_at).days)
    return max(0.0, 1.0 - min(age_days, 30) / 30.0)


def _max_item(values: Mapping[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    valid = [(key, float(value)) for key, value in values.items() if value is not None]
    return max(valid, key=lambda item: item[1]) if valid else (None, None)


def _min_item(values: Mapping[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    valid = [(key, float(value)) for key, value in values.items() if value is not None]
    return min(valid, key=lambda item: item[1]) if valid else (None, None)


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _generic_dataset_summary(datasets: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
    return {
        "all_listings_count": len(datasets.get("all_listings", [])),
        "active_listings_count": len(datasets.get("active_listings", [])),
        "completed_purchases_count": len(datasets.get("purchases", [])),
    }


def _unique_strings(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
