"""
rag_service.py
--------------

End-to-end orchestration for GreenGrid Exchange RAG, marketplace analytics,
prediction, and recommendation answers.

Flow:
1. Embed the user question and retrieve relevant RAG chunks.
2. Ask Nova Micro to create a validated read-only marketplace tool plan.
3. Execute approved public GET tools through marketplace_api_service.
4. Calculate deterministic analytics through analytics_service.
5. Build a compact grounded prompt.
6. Generate the final answer with Nova Micro.
7. Return RAG citations and marketplace execution metadata.
"""

from __future__ import annotations

import html
from html.parser import HTMLParser
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from app.config import settings
from app.schemas.query_schema import (
    QueryApiSummary,
    QueryPeriod,  # type: ignore
    QueryRequest,
    QueryResponse,
    QuerySource,
    QueryToolResult,  # type: ignore
)
from app.services import (
    analytics_service,
    bedrock_service,
    marketplace_api_service,
    opensearch_service,
    prompt_service,
    tool_registry,
    ui_constants,
)

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 4


# ---------------------------------------------------------------------------
# Marketplace tool execution
# ---------------------------------------------------------------------------


def _execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one approved public marketplace GET tool."""
    if tool_registry.get_tool_by_name(tool_name) is None:
        raise ValueError(f"Tool is not registered: {tool_name}")

    if tool_name not in {
        "get_all_listings",
        "get_active_listings",
        "get_all_purchases",
    }:
        raise ValueError(f"Tool has no read-only marketplace executor: {tool_name}")

    return marketplace_api_service.execute_tool_call(
        tool_name=tool_name,
        arguments=arguments,
    )


def _execute_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Execute validated planner calls without failing the entire query."""
    raw_calls = plan.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return []

    results: List[Dict[str, Any]] = []

    for call in raw_calls[:MAX_TOOL_CALLS]:
        if not isinstance(call, Mapping):
            continue

        tool_name = str(call.get("tool", "") or "").strip()
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        try:
            result = _execute_tool_call(tool_name, arguments)
            results.append(result)
        except Exception as exc:
            logger.exception(
                "Tool execution failed before API result creation: %s",
                tool_name,
            )
            results.append(
                {
                    "tool": tool_name or "unknown",
                    "data": {
                        "records": [],
                        "sample_records": [],
                        "aggregates": {},
                        "response_metadata": {},
                    },
                    "arguments": arguments,
                    "record_count": 0,
                    "pages_fetched": 0,
                    "endpoint": None,
                    "execution_status": "failed",
                    "error": _safe_error(exc),
                }
            )

    return results


# ---------------------------------------------------------------------------
# API and analytics summary construction
# ---------------------------------------------------------------------------


def _resolve_api_summary(
    question: str,
) -> Tuple[Optional[QueryApiSummary], bool, Dict[str, Any]]:
    """
    Plan, execute, and analyze marketplace tools.

    Returns:
        (summary, api_facts_used, plan)
    """
    plan = bedrock_service.plan_api_calls(question)

    if not plan.get("requires_api_data", False):
        return None, False, plan

    tool_results = _execute_plan(plan)
    if not tool_results:
        return None, False, plan

    analysis = analytics_service.analyze_plan(
        plan=plan,
        tool_results=tool_results,
    )

    successful_or_empty = any(
        result.get("execution_status") in {"success", "partial", "empty"}
        for result in tool_results
    )

    # The new public APIs can return authoritative aggregate sections such as
    # source_breakdown, location_breakdown, supply_by_source, and
    # demand_by_source. Those aggregates count as live facts even if a response
    # contains no raw records.
    api_facts_used = any(
        result.get("execution_status") in {"success", "partial"}
        and (
            int(result.get("record_count", 0) or 0) > 0
            or _has_aggregate_data(result)
        )
        for result in tool_results
    )

    context_type = _context_type(plan)
    compact_tool_results = [
        _to_query_tool_result(result)
        for result in tool_results
    ]

    filters_used = [
        {
            "tool": result.get("tool"),
            "arguments": result.get("arguments", {}),
        }
        for result in tool_results
    ]

    summary = QueryApiSummary(
        context_type=context_type,
        planner_reason=str(plan.get("reason", "") or ""),
        intent=str(analysis.get("intent", plan.get("intent", "none"))),  # type: ignore
        is_prediction=bool(plan.get("is_prediction", False)),  # type: ignore
        is_recommendation=bool(plan.get("is_recommendation", False)),  # type: ignore
        historical_period=_to_query_period(plan.get("historical_period")),  # type: ignore
        forecast_period=_to_query_period(plan.get("forecast_period")),  # type: ignore
        group_by=list(plan.get("group_by", []) or []),  # type: ignore
        metrics=list(plan.get("metrics", []) or []),  # type: ignore
        filters_used=filters_used,  # type: ignore
        records_analyzed=dict(analysis.get("records_analyzed", {}) or {}),  # type: ignore
        analytics_result=dict(analysis.get("analytics_result", {}) or {}),  # type: ignore
        prediction_result=analysis.get("prediction_result"),  # type: ignore
        recommendation_result=analysis.get("recommendation_result"),  # type: ignore
        confidence=analysis.get("confidence"),  # type: ignore
        limitations=list(analysis.get("limitations", []) or []),  # type: ignore
        missing_parameters=list(plan.get("missing_parameters", []) or []),  # type: ignore
        calculation_method=analysis.get("calculation_method"),  # type: ignore
        data_as_of=datetime.now(timezone.utc).date().isoformat(),  # type: ignore
        tool_results=compact_tool_results,  # type: ignore
    )

    # Empty API responses are still useful operational context, but they do not
    # count as API facts. This lets the final answer say that no records matched.
    if successful_or_empty and not api_facts_used:
        summary.limitations = _append_unique(  # type: ignore
            summary.limitations,  # type: ignore
            "No matching marketplace records or aggregate data were available "
            "for the requested scope.",
        )

    return summary, api_facts_used, plan


def _has_aggregate_data(result: Mapping[str, Any]) -> bool:
    """Return True when a tool result contains a non-empty aggregate section."""
    raw_data = result.get("data", {})
    if not isinstance(raw_data, Mapping):
        return False

    aggregates = raw_data.get("aggregates", {})
    if not isinstance(aggregates, Mapping):
        return False

    return any(
        value not in (None, {}, [], "")
        for value in aggregates.values()
    )


def _to_query_tool_result(result: Mapping[str, Any]) -> QueryToolResult:
    """Convert a raw tool result while excluding full record arrays."""
    raw_data = result.get("data", {})
    compact_data: Dict[str, Any] = {
        "sample_records": [],
        "aggregates": {},
        "response_metadata": {},
    }

    if isinstance(raw_data, Mapping):
        samples = raw_data.get("sample_records", [])
        if isinstance(samples, list):
            compact_data["sample_records"] = samples[
                : settings.ANALYTICS_LLM_SAMPLE_RECORDS
            ]

        # Preserve the normalized aggregate sections returned by the new public
        # APIs. These are compact and useful for answer explanation, unlike the
        # full raw records array, which remains excluded.
        aggregates = raw_data.get("aggregates", {})
        if isinstance(aggregates, Mapping):
            compact_data["aggregates"] = dict(aggregates)

        metadata = raw_data.get("response_metadata", {})
        if isinstance(metadata, Mapping):
            compact_data["response_metadata"] = dict(metadata)

    return QueryToolResult(
        tool=str(result.get("tool", "unknown")),
        data=compact_data,
        arguments=dict(result.get("arguments", {}) or {}),
        record_count=int(result.get("record_count", 0) or 0),
        pages_fetched=int(result.get("pages_fetched", 0) or 0),
        endpoint=result.get("endpoint"),
        execution_status=str(result.get("execution_status", "failed")),
        error=result.get("error"),
    )


def _to_query_period(value: Any) -> Optional[QueryPeriod]:
    """Convert planner period dictionaries to the response model."""
    if not isinstance(value, Mapping):
        return None

    from_value = value.get("from")
    to_value = value.get("to")
    if not from_value and not to_value:
        return None

    return QueryPeriod(
        from_date=str(from_value) if from_value else None, # type: ignore
        to_date=str(to_value) if to_value else None, # type: ignore
    )


def _context_type(plan: Mapping[str, Any]) -> str:
    if plan.get("is_prediction", False):
        return "prediction"
    if plan.get("is_recommendation", False):
        return "recommendation"

    tool_count = len(plan.get("tool_calls", []) or [])
    intent = str(plan.get("intent", "none") or "none")
    if tool_count > 1:
        return "mixed"
    if intent not in {"none", "current_supply", "supply_mix"}:
        return "analytics"
    return "marketplace"


# ---------------------------------------------------------------------------
# RAG retrieval and source conversion
# ---------------------------------------------------------------------------


def _retrieve_chunks(question: str, top_k: int) -> List[Dict[str, Any]]:
    """Retrieve RAG chunks; allow API-backed answers if retrieval fails."""
    try:
        embedding = bedrock_service.embed_text(question)
        return opensearch_service.search_similar_chunks(
            embedding,
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning("RAG retrieval unavailable: %s", exc)
        return []


def _build_sources(hits: Sequence[Mapping[str, Any]]) -> List[QuerySource]:
    sources: List[QuerySource] = []

    for hit in hits:
        try:
            sources.append(
                QuerySource(
                    chunk_id=str(hit.get("chunk_id", "") or ""),
                    document_id=str(hit.get("document_id", "") or ""),
                    document_name=str(hit.get("document_name", "") or ""),
                    document_type=str(hit.get("document_type", "") or ""),
                    chunk_index=int(hit.get("chunk_index", 0) or 0),
                    s3_uri=str(hit.get("s3_uri", "") or ""),
                    score=float(hit.get("score", 0.0) or 0.0),
                    snippet=str(hit.get("text", "") or "")[:280].replace(
                        "\n",
                        " ",
                    ),
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning("Skipping malformed RAG source: %s", exc)

    return sources


# ---------------------------------------------------------------------------
# Answer and fallback handling
# ---------------------------------------------------------------------------


def _build_fallback_answer(
    api_summary: Optional[QueryApiSummary],
    hits: Sequence[Mapping[str, Any]],
    generation_error: Exception,
) -> str:
    """Return a useful fallback when final LLM generation is unavailable."""
    logger.warning(
        "Building fallback answer after generation failure: %s",
        generation_error,
    )

    if api_summary is not None:  # type: ignore
        if api_summary.missing_parameters:  # type: ignore
            missing = ", ".join(api_summary.missing_parameters)  # type: ignore
            return f"I need the following information to answer accurately: {missing}."

        if api_summary.confidence == "insufficient_data":  # type: ignore
            limitations = " ".join(api_summary.limitations[:2])  # type: ignore
            return (
                "There is insufficient marketplace data for a reliable answer. "
                f"{limitations}".strip()
            )

        if api_summary.prediction_result:  # type: ignore
            return (
                "The prediction was calculated, but the language-generation service "
                "is temporarily unavailable. Please retry shortly."
            )

        if api_summary.recommendation_result:  # type: ignore
            return (
                "The recommendation analysis was calculated, but the "
                "language-generation service is temporarily unavailable. "
                "Please retry shortly."
            )

        if api_summary.analytics_result:  # type: ignore
            return (
                "Marketplace analytics were calculated, but the language-generation "
                "service is temporarily unavailable. Please retry shortly."
            )

    if hits:
        return (
            "Relevant knowledge-base information was found, but the "
            "answer-generation service is temporarily unavailable. "
            "Please retry shortly."
        )

    return (
        "I could not retrieve enough information to answer this question. "
        "Please refine the question or try again shortly."
    )


def _insufficient_response(
    api_summary: Optional[QueryApiSummary],
    plan: Mapping[str, Any],
) -> str:
    missing = list(plan.get("missing_parameters", []) or [])
    if missing:
        return (
            "I need additional information before I can answer accurately: "
            + ", ".join(missing)
            + "."
        )

    if api_summary and api_summary.limitations:  # type: ignore
        return " ".join(api_summary.limitations[:2])  # type: ignore

    return "No matching marketplace or knowledge-base information was found."


def _live_data_unavailable_response(
    api_summary: Optional[QueryApiSummary],
) -> str:
    """
    Return a grounded response when a live-data question has no usable API data.

    Static RAG examples must never substitute for current or historical
    marketplace facts, predictions, or recommendations.
    """
    if api_summary and api_summary.limitations:  # type: ignore
        detail = " ".join(api_summary.limitations[:2])  # type: ignore
    else:
        detail = "The required public marketplace API returned no usable data."

    return (
        "Live marketplace data is unavailable for this question. "
        "The answer was not generated from static RAG examples. "
        f"{detail}"
    )


def _answer_mode(
    has_hits: bool,
    api_summary: Optional[QueryApiSummary],
    api_facts_used: bool,
) -> str:
    if has_hits and api_summary is not None:
        return "retrieval_plus_api"
    if api_summary is not None:
        return "api_only"
    if has_hits:
        return "retrieval_only"
    return "insufficient_data"


def _effective_period(api_summary: QueryApiSummary) -> Tuple[Optional[str], Optional[str]]:
    """Use model period first, then derive dates from executed tool filters."""
    period = api_summary.historical_period
    start = str(getattr(period, "from_date", "") or "") if period else ""
    end = str(getattr(period, "to_date", "") or "") if period else ""
    if start and end:
        return start[:10], end[:10]
    for item in list(api_summary.filters_used or []):
        if not isinstance(item, Mapping):
            continue
        arguments = item.get("arguments", {})
        if not isinstance(arguments, Mapping):
            continue
        candidate_start = arguments.get("created_from") or arguments.get("completed_from")
        candidate_end = arguments.get("created_to") or arguments.get("completed_to")
        if candidate_start and candidate_end:
            return str(candidate_start)[:10], str(candidate_end)[:10]
    return None, None


def _period_caption_fixed(api_summary: QueryApiSummary) -> str:
    start, end = _effective_period(api_summary)
    if not start or not end:
        return "the requested period"
    current_date = datetime.now(timezone.utc).date()
    today = current_date.isoformat()
    yesterday = (current_date - timedelta(days=1)).isoformat()
    if start == end == today:
        return f"today ({today})"
    if start == end == yesterday:
        return f"yesterday ({yesterday})"
    if start == end:
        return start
    return f"{start} to {end}"


def _forecast_caption(api_summary: QueryApiSummary) -> str:
    period = api_summary.forecast_period
    start = str(getattr(period, "from_date", "") or "") if period else ""
    end = str(getattr(period, "to_date", "") or "") if period else ""
    if not start or not end:
        return "the forecast period"
    current_date = datetime.now(timezone.utc).date()
    next_week_start = current_date + timedelta(days=(7 - current_date.weekday()))
    next_week_end = next_week_start + timedelta(days=6)
    if start[:10] == next_week_start.isoformat() and end[:10] == next_week_end.isoformat():
        return "next week"
    if start == end:
        return start[:10]
    return f"{start[:10]} to {end[:10]}"


def _requested_sources(api_summary: QueryApiSummary) -> List[str]:
    result: List[str] = []
    for item in list(api_summary.filters_used or []):
        if not isinstance(item, Mapping):
            continue
        arguments = item.get("arguments", {})
        if isinstance(arguments, Mapping):
            value = arguments.get("energy_source")
            if value and str(value) not in result:
                result.append(str(value))
    return result


def _insufficient_answer(title: str, message: str, api_summary: Optional[QueryApiSummary]) -> str:
    data_as_of = str(getattr(api_summary, "data_as_of", "") or "")[:10] if api_summary else ""
    note = (
        f'<div style="{ui_constants.NOTE_STYLE}">Data as of: {html.escape(data_as_of)}</div>'
        if len(data_as_of) == 10 else ""
    )
    return (
        f'<section style="{ui_constants.SECTION_STYLE}">'
        f'<div style="{ui_constants.HERO_WARNING_STYLE}">'
        f'<h3 style="{ui_constants.HEADING_STYLE}">{html.escape(title)}</h3>'
        f'<p style="{ui_constants.PARAGRAPH_STYLE}">{html.escape(message)}</p></div>{note}</section>'
    )


def _prediction_has_result(api_summary: QueryApiSummary, keys: Sequence[str]) -> bool:
    result = api_summary.prediction_result
    if not isinstance(result, Mapping):
        return False
    return any(result.get(key) not in (None, "", []) for key in keys)


def _prediction_disclaimer() -> str:
    return "Prediction output is decision support only and should not be taken as financial, trading, or investment advice."


def _render_demand_prediction_or_insufficient(api_summary: QueryApiSummary) -> str:
    prediction = dict(api_summary.prediction_result or {})
    source = (_filter_argument(api_summary, "energy_source") or "").upper()
    location = _filter_argument(api_summary, "location")
    forecast_label = _forecast_caption(api_summary)
    if not _prediction_has_result(api_summary, ("predicted_highest_demand_kwh", "predictions_by_source")):
        scope_source = source.title() if source else "renewable"
        scope = f"{scope_source} credits" + (f" in {location}" if location else "")
        return _insufficient_answer(
            "Demand prediction unavailable",
            f"Due to insufficient historical purchase records and demand forecasting data for {scope}, I cannot predict demand during {forecast_label}.",
            api_summary,
        )

    predictions = prediction.get("predictions_by_source") if isinstance(prediction, Mapping) else {}
    if not isinstance(predictions, Mapping):
        predictions = {}

    model = str(prediction.get("model") or "weighted moving average")
    if source and source in predictions and isinstance(predictions.get(source), Mapping):
        item = dict(predictions[source] or {})
        predicted = _number_or_none(item.get("predicted_forecast_period_kwh"))
        last_observed = _number_or_none(item.get("last_observed_period_kwh"))
        change_pct = _number_or_none(item.get("forecast_change_pct"))
        periods_used = int(item.get("periods_used") or 0)
        forecast_weeks = int(item.get("forecast_periods") or prediction.get("forecast_weeks") or 1)
        trend_text = "is expected to stay roughly flat"
        if change_pct is not None:
            if change_pct > 0:
                trend_text = f"is expected to increase by about {_display_value(abs(change_pct), '%')}"
            elif change_pct < 0:
                trend_text = f"is expected to decrease by about {_display_value(abs(change_pct), '%')}"
        scope = f"{source.title()} demand" + (f" in {location}" if location else "")
        finding = (
            f"{scope} during {forecast_label} is projected at {_display_value(predicted, 'kWh')}. "
            f"Compared with the latest observed period ({_display_value(last_observed, 'kWh')}), demand {trend_text}."
        )
        detail = (
            f"Calculation: recursive 4-point weighted moving average (weights 0.1, 0.2, 0.3, 0.4) over weekly completed-demand history. "
            f"Model used {periods_used} historical period(s) and forecast {forecast_weeks} week(s). "
            "Trend signal compares forecast demand with the most recent observed period. "
            + _prediction_disclaimer()
        )
        table = _responsive_table(
            ["Scope", "Forecast demand", "Latest observed demand", "Expected change"],
            [[scope, _display_value(predicted, "kWh"), _display_value(last_observed, "kWh"), _trend_display(change_pct)]],
        )
        return _render_standard_answer(
            "Demand prediction",
            finding,
            table + f'<div style="{ui_constants.NOTE_STYLE};margin-top:10px;">{html.escape(detail)}</div>',
            api_summary,
            forecast_label,
        )

    leader = str(prediction.get("predicted_highest_demand_source") or "")
    leader_kwh = _number_or_none(prediction.get("predicted_highest_demand_kwh"))
    finding = (
        f"{leader.title()} is projected to have the highest demand during {forecast_label}, at about {_display_value(leader_kwh, 'kWh')}."
        if leader else
        f"Demand was forecast during {forecast_label}, but no clear leading source was identified."
    )
    rows: List[List[str]] = []
    max_periods_used = 0
    forecast_weeks = int(prediction.get("forecast_weeks") or 1)
    for source_name in settings.SUPPORTED_ENERGY_SOURCES:
        item = predictions.get(source_name, {}) if isinstance(predictions.get(source_name), Mapping) else {}
        max_periods_used = max(max_periods_used, int(item.get("periods_used") or 0))
        rows.append([
            source_name.title(),
            _display_value(item.get("predicted_forecast_period_kwh"), "kWh"),
            _display_value(item.get("last_observed_period_kwh"), "kWh"),
            _trend_display(item.get("forecast_change_pct")),
        ])
    detail = (
        f"Calculation: recursive 4-point weighted moving average (weights 0.1, 0.2, 0.3, 0.4) over weekly completed-demand history by source. "
        f"Model used up to {max_periods_used} historical period(s) and forecast {forecast_weeks} week(s). "
        "Trend signal compares each source forecast with its latest observed period. "
        + _prediction_disclaimer()
    )
    return _render_standard_answer(
        "Demand prediction",
        finding,
        _responsive_table(["Source", "Forecast demand", "Latest observed demand", "Expected change"], rows)
        + f'<div style="{ui_constants.NOTE_STYLE};margin-top:10px;">{html.escape(detail)}</div>',
        api_summary,
        forecast_label,
    )


def _render_shortage_prediction_or_insufficient(api_summary: QueryApiSummary) -> str:
    prediction = dict(api_summary.prediction_result or {})
    source = (_filter_argument(api_summary, "energy_source") or "renewable").title()
    location = _filter_argument(api_summary, "location") or "the requested location"
    forecast_label = _forecast_caption(api_summary)
    if not _prediction_has_result(api_summary, ("shortage_expected", "projected_gap_kwh", "projections_by_source", "highest_shortage_risk_source")):
        return _insufficient_answer(
            "Shortage prediction unavailable",
            f"Due to insufficient predictive data regarding new supply and future demand for {source} credits in {location}, I cannot determine if there will be a shortage during {forecast_label}.",
            api_summary,
        )
    projections = prediction.get("projections_by_source") if isinstance(prediction, Mapping) else None
    if isinstance(projections, Mapping) and projections:
        ranked = [
            (str(key), value)
            for key, value in projections.items()
            if isinstance(value, Mapping)
        ]
        shortage_sources = [name for name, value in ranked if bool(value.get("shortage_expected"))]
        top_source = str(prediction.get("highest_shortage_risk_source") or ranked[0][0]).title()
        finding = (
            f"{top_source} currently shows the highest shortage risk during {forecast_label}. A shortage is flagged when forecast demand exceeds forecast new supply plus current active supply."
            if shortage_sources else
            f"No renewable source currently shows a projected shortage during {forecast_label} based on the available marketplace data."
        )
        rows = []
        for name, value in ranked:
            rows.append([
                name.title(),
                _display_value(value.get("predicted_demand_kwh"), "kWh"),
                _display_value(value.get("predicted_new_supply_kwh"), "kWh"),
                _display_value(value.get("current_active_supply_kwh"), "kWh"),
                _display_value(value.get("projected_gap_kwh"), "kWh"),
                "Yes" if value.get("shortage_expected") else "No",
                _trend_display(value.get("demand_trend_pct")),
                _trend_display(value.get("supply_trend_pct")),
            ])
        detail = (
            "Calculation: projected gap = forecast new supply + current active supply - forecast demand. "
            "Trends considered: recent completed-demand trend, recent new-listing trend, and current active inventory. "
            + _prediction_disclaimer()
        )
        return _render_standard_answer(
            "Upcoming shortage risk by renewable source",
            finding,
            _responsive_table(
                ["Source", "Forecast demand", "Forecast new supply", "Current active supply", "Projected gap", "Shortage risk", "Demand trend", "Supply trend"],
                rows,
            ) + f'<div style="{ui_constants.NOTE_STYLE};margin-top:10px;">{html.escape(detail)}</div>',
            api_summary,
        )

    detail = (
        "Calculation: projected gap = forecast new supply + current active supply - forecast demand. "
        "Trends considered: recent completed-demand trend, recent new-listing trend, and current active inventory. "
        + _prediction_disclaimer()
    )
    finding = (
        f"{source} credits are likely to face a shortage in {location} during {forecast_label}."
        if prediction.get("shortage_expected") else
        f"{source} credits are not currently projected to face a shortage in {location} during {forecast_label}."
    )
    rows = [[
        source,
        _display_value(prediction.get("predicted_demand_kwh"), "kWh"),
        _display_value(prediction.get("predicted_new_supply_kwh"), "kWh"),
        _display_value(prediction.get("current_active_supply_kwh"), "kWh"),
        _display_value(prediction.get("projected_gap_kwh"), "kWh"),
        _trend_display(prediction.get("demand_trend_pct")),
        _trend_display(prediction.get("supply_trend_pct")),
    ]]
    return _render_standard_answer(
        "Upcoming shortage prediction",
        finding,
        _responsive_table(
            ["Source", "Forecast demand", "Forecast new supply", "Current active supply", "Projected gap", "Demand trend", "Supply trend"],
            rows,
        ) + f'<div style="{ui_constants.NOTE_STYLE};margin-top:10px;">{html.escape(detail)}</div>',
        api_summary,
    )


def _render_supply_forecast_from_shortage(api_summary: QueryApiSummary) -> str:
    prediction = dict(api_summary.prediction_result or {})
    source = (_filter_argument(api_summary, "energy_source") or "renewable").title()
    location = _filter_argument(api_summary, "location") or "the requested location"
    forecast_label = _forecast_caption(api_summary)
    projections = prediction.get("projections_by_source") if isinstance(prediction, Mapping) else None

    if isinstance(projections, Mapping) and projections:
        rows: List[List[str]] = []
        top_source = None
        top_supply = None
        for name, value in projections.items():
            if not isinstance(value, Mapping):
                continue
            forecast_supply = _number_or_none(value.get("predicted_new_supply_kwh"))
            if top_supply is None or ((forecast_supply or 0.0) > top_supply):
                top_supply = forecast_supply or 0.0
                top_source = str(name)
            rows.append([
                str(name).title(),
                _display_value(value.get("predicted_new_supply_kwh"), "kWh"),
                _display_value(value.get("current_active_supply_kwh"), "kWh"),
                _trend_display(value.get("supply_trend_pct")),
                _trend_display(value.get("demand_trend_pct")),
            ])
        finding = (
            f"{top_source.title()} is projected to have the highest new supply during {forecast_label}, at about {_display_value(top_supply, 'kWh')}."
            if top_source else
            f"Supply forecast for {forecast_label} was generated from available listing and active-inventory data."
        )
        detail = (
            "Calculation: forecast new supply is estimated using a recursive 4-point weighted moving average "
            "(weights 0.1, 0.2, 0.3, 0.4) over recent listing-creation history, then interpreted alongside current active supply. "
            + _prediction_disclaimer()
        )
        return _render_standard_answer(
            "Upcoming supply forecast by renewable source",
            finding,
            _responsive_table(
                ["Source", "Forecast new supply", "Current active supply", "Supply trend", "Demand trend"],
                rows,
            ) + f'<div style="{ui_constants.NOTE_STYLE};margin-top:10px;">{html.escape(detail)}</div>',
            api_summary,
            forecast_label,
        )

    finding = f"{source} supply in {location} is expected to stay stable during {forecast_label} based on available listing and active-inventory history."
    detail = (
        "Calculation: forecast new supply is estimated using a recursive 4-point weighted moving average "
        "(weights 0.1, 0.2, 0.3, 0.4) over recent listing-creation history, interpreted with current active supply. "
        + _prediction_disclaimer()
    )
    return _render_standard_answer(
        "Upcoming supply forecast",
        finding,
        f'<div style="{ui_constants.NOTE_STYLE}">{html.escape(detail)}</div>',
        api_summary,
        forecast_label,
    )


def _render_price_prediction_or_insufficient(api_summary: QueryApiSummary) -> str:
    prediction = dict(api_summary.prediction_result or {})
    source = (_filter_argument(api_summary, "energy_source") or "").upper()
    forecast_label = _forecast_caption(api_summary)
    if not _prediction_has_result(api_summary, ("predicted_highest_price_source", "predictions_by_source")):
        return _insufficient_answer(
            "Price prediction unavailable",
            f"Due to insufficient historical purchase records and pricing trends, I cannot produce a reliable price forecast for {forecast_label}.",
            api_summary,
        )

    predictions = prediction.get("predictions_by_source") if isinstance(prediction, Mapping) else {}
    if not isinstance(predictions, Mapping):
        predictions = {}

    if source and source in predictions and isinstance(predictions.get(source), Mapping):
        item = dict(predictions[source] or {})
        predicted = _number_or_none(item.get("predicted_forecast_period_price_per_kwh"))
        low = _number_or_none(item.get("lower_bound"))
        high = _number_or_none(item.get("upper_bound"))
        change_pct = _number_or_none(item.get("forecast_change_pct"))
        finding = (
            f"{source.title()} price during {forecast_label} is projected around {_display_value(predicted, 'per kWh')} "
            f"with an estimated range of {_display_value(low, 'per kWh')} to {_display_value(high, 'per kWh')}."
        )
        detail = (
            "Calculation: weighted moving-average forecast from realized completed-purchase prices. "
            "Range is reported as lower and upper uncertainty bounds from the forecast error band. "
            + _prediction_disclaimer()
        )
        table = _responsive_table(
            ["Source", "Forecast price", "Price range", "Expected change"],
            [[source.title(), _display_value(predicted, "per kWh"), f"{_display_value(low)} to {_display_value(high)}", _trend_display(change_pct)]],
        )
        return _render_standard_answer(
            "Price prediction",
            finding,
            table + f'<div style="{ui_constants.NOTE_STYLE};margin-top:10px;">{html.escape(detail)}</div>',
            api_summary,
            forecast_label,
        )

    leader = str(prediction.get("predicted_highest_price_source") or "")
    leader_price = _number_or_none(prediction.get("predicted_highest_price_per_kwh"))
    finding = (
        f"{leader.title()} is projected to have the highest price during {forecast_label}, around {_display_value(leader_price, 'per kWh')}."
        if leader else
        f"A price forecast was generated for {forecast_label}, but no single source clearly leads."
    )
    rows: List[List[str]] = []
    for source_name in settings.SUPPORTED_ENERGY_SOURCES:
        item = predictions.get(source_name, {}) if isinstance(predictions.get(source_name), Mapping) else {}
        low = _number_or_none(item.get("lower_bound"))
        high = _number_or_none(item.get("upper_bound"))
        rows.append([
            source_name.title(),
            _display_value(item.get("predicted_forecast_period_price_per_kwh"), "per kWh"),
            f"{_display_value(low)} to {_display_value(high)}",
            _trend_display(item.get("forecast_change_pct")),
        ])
    detail = (
        "Calculation: weighted moving-average forecast from realized completed-purchase prices by source. "
        "Ranges are lower and upper uncertainty bounds from forecast error. "
        + _prediction_disclaimer()
    )
    return _render_standard_answer(
        "Price prediction",
        finding,
        _responsive_table(["Source", "Forecast price", "Price range", "Expected change"], rows)
        + f'<div style="{ui_constants.NOTE_STYLE};margin-top:10px;">{html.escape(detail)}</div>',
        api_summary,
        forecast_label,
    )


def _render_ratio_html(api_summary: QueryApiSummary) -> str:
    analytics = dict(api_summary.analytics_result or {})
    ratios = analytics.get("demand_supply_ratio_by_source", {}) or {}
    supply = analytics.get("listed_supply_kwh_by_source", {}) or {}
    demand = analytics.get("completed_demand_kwh_by_source", {}) or {}
    ordered = sorted(
        settings.SUPPORTED_ENERGY_SOURCES,
        key=lambda source: _number_or_none(ratios.get(source)) or -1.0,
        reverse=True,
    )
    leader = next((source for source in ordered if _number_or_none(ratios.get(source)) is not None), None)
    finding = (
        f"{leader.title()} has the highest demand-to-supply ratio at "
        f"{_display_value((_number_or_none(ratios.get(leader)) or 0.0) * 100, '%')}."
        if leader else "The available data is insufficient to calculate demand-to-supply ratios."
    )
    rows = []
    for source in ordered:
        ratio = _number_or_none(ratios.get(source))
        rows.append([
            source.title(),
            _display_value(demand.get(source), "kWh"),
            _display_value(supply.get(source), "kWh"),
            _display_value(ratio * 100, "%") if ratio is not None else "-",
        ])
    return _render_standard_answer(
        "Demand-to-supply ratio by renewable source",
        finding,
        _responsive_table(["Source", "Completed demand", "Listed supply", "Demand-to-supply ratio"], rows),
        api_summary,
    )


def _render_buyer_recommendation_or_insufficient(api_summary: QueryApiSummary) -> str:
    recommendation = api_summary.recommendation_result
    listing = recommendation.get("recommended_listing") if isinstance(recommendation, Mapping) else None
    if not listing:
        return _insufficient_answer(
            "Recommendation unavailable",
            "Due to insufficient price, availability, and demand data, I cannot reliably recommend the best renewable credit.",
            api_summary,
        )
    return _render_complete_api_fallback(api_summary)


def _render_price_guidance_or_insufficient(api_summary: QueryApiSummary) -> str:
    analytics = dict(api_summary.analytics_result or {})
    average = analytics.get("average_selling_price", {}) if isinstance(analytics, Mapping) else {}
    prices = average.get("weighted_average_selling_price_by_source", {}) if isinstance(average, Mapping) else {}
    available = [(source, _number_or_none(value)) for source, value in prices.items() if _number_or_none(value) is not None]
    if not available or str(api_summary.confidence) == "insufficient_data":
        return _insufficient_answer(
            "Pricing guidance unavailable",
            "Due to insufficient completed sales and comparable pricing history, I cannot recommend a reliable price for your credit.",
            api_summary,
        )
    rows = [[source.title(), _display_value(value, "per kWh")] for source, value in available]
    return _render_standard_answer(
        "Recent realized prices",
        "Use recent realized prices only as a reference; the available data does not guarantee a sale at any specific price.",
        _responsive_table(["Source", "Realized price"], rows),
        api_summary,
    )


def _render_historical_shortage_or_insufficient(api_summary: QueryApiSummary) -> str:
    analytics = dict(api_summary.analytics_result or {})
    source = (_filter_argument(api_summary, "energy_source") or "SOLAR").upper()
    location = _filter_argument(api_summary, "location") or "the requested location"
    total_supply = analytics.get("total_supply_kwh_by_source", {}) or {}
    demand = analytics.get("realized_demand_kwh_by_source", {}) or {}
    supply_value = _number_or_none(total_supply.get(source))
    demand_value = _number_or_none(demand.get(source))
    period = _period_caption_fixed(api_summary)
    if supply_value is None or demand_value is None:
        return _insufficient_answer(
            "Historical shortage analysis unavailable",
            f"Due to insufficient data regarding {period} total completed demand and available supply for {source.title()} credits in {location}, I cannot determine if there was a shortage.",
            api_summary,
        )
    shortage = demand_value > supply_value
    finding = (
        f"Yes, there was a shortage of {source.title()} credits in {location} during {period}. "
        f"Demand of {_display_value(demand_value, 'kWh')} exceeded supply of {_display_value(supply_value, 'kWh')}."
        if shortage else
        f"No shortage was identified for {source.title()} credits in {location} during {period}. "
        f"Supply was {_display_value(supply_value, 'kWh')} and demand was {_display_value(demand_value, 'kWh')}."
    )
    return _render_standard_answer("Historical shortage analysis", finding, "", api_summary, period)


def _render_recent_listings_html(api_summary: QueryApiSummary) -> str:
    records: List[Mapping[str, Any]] = []
    for result in list(api_summary.tool_results or []):
        if getattr(result, "tool", "") != "get_all_listings":
            continue
        data = getattr(result, "data", None)
        sample = getattr(data, "sample_records", None) if data is not None else None
        if isinstance(sample, list):
            records.extend(item for item in sample if isinstance(item, Mapping))
    if not records:
        analytics = dict(api_summary.analytics_result or {})
        total = _display_value(analytics.get("total_listed_supply_kwh"), "kWh")
        leader = analytics.get("highest_listed_supply_source")
        return _insufficient_answer(
            "Recent listings unavailable",
            f"Due to insufficient data on individual listing events, I cannot show the specific credits that were recently listed. Aggregate listed supply is {total}"
            + (f", led by {str(leader).title()}." if leader else "."),
            api_summary,
        )
    rows = [[
        str(item.get("id") or "-"),
        str(item.get("energy_source") or "-").title(),
        _display_value(item.get("energy_kwh"), "kWh"),
        _display_value(item.get("price_per_kwh"), ""),
        str(item.get("location") or "-"),
        str(item.get("created_at") or "-")[:19],
    ] for item in records]
    return _render_standard_answer(
        "Recently listed credits",
        f"{len(rows)} recent listing events are available in the returned marketplace data.",
        _responsive_table(["Listing", "Source", "Supply", "Price per kWh", "Location", "Listed at"], rows),
        api_summary,
    )


# ---------------------------------------------------------------------------
# Public orchestration entry point
# ---------------------------------------------------------------------------


def _render_marketplace_summary_html(
    api_summary: QueryApiSummary,
) -> str:
    """Render complete responsive marketplace summary HTML deterministically."""
    analytics = dict(api_summary.analytics_result or {})
    period = analytics.get("period", {})
    period_from = str(period.get("from") or "") if isinstance(period, Mapping) else ""
    period_to = str(period.get("to") or "") if isinstance(period, Mapping) else ""
    is_today = bool(analytics.get("is_current_day_summary", False))

    listing_activity = analytics.get("period_listing_activity", {}) or {}
    purchase_activity = analytics.get("period_purchase_activity", {}) or {}
    source_stats = analytics.get("source_statistics", {}) or {}
    balance = analytics.get("period_market_balance", {}) or {}
    current_inventory = analytics.get("current_unsold_inventory_from_period", {}) or {}

    period_label = _summary_period_label(period_from, period_to, is_today)
    is_historical_summary = _is_historical_marketplace_summary(
        period_from,
        period_to,
        is_today,
    )
    if period_label in {"Current", "Marketplace"}:
        period_label = _period_caption_fixed(api_summary)
    listed_total = _number_or_none(
        listing_activity.get("newly_listed_supply_kwh_in_period")
        or listing_activity.get("listed_supply_kwh")
    )
    demand_total = _number_or_none(
        purchase_activity.get("completed_demand_kwh_in_period")
        or purchase_activity.get("completed_demand_kwh")
    )
    available_total = _number_or_none(current_inventory.get("total_active_supply_kwh"))
    listing_count = _int_or_zero(
        listing_activity.get("new_listing_count_in_period")
        or listing_activity.get("listing_count")
    )
    purchase_count = _int_or_zero(
        purchase_activity.get("completed_purchase_count_in_period")
        or purchase_activity.get("completed_purchase_count")
    )

    demand_by_source = purchase_activity.get("completed_demand_by_source_kwh_in_period", {}) or purchase_activity.get("completed_demand_by_source_kwh", {}) or {}
    listed_by_source = listing_activity.get("newly_listed_supply_by_source_kwh_in_period", {}) or listing_activity.get("listed_supply_by_source_kwh", {}) or {}
    balance_by_source = balance.get("market_balance_kwh_by_source", {}) or {}

    demand_leader = _positive_leader(demand_by_source)
    supply_leader = _positive_leader(listed_by_source)
    summary = _marketplace_interpretation(
        period_label=period_label,
        listed_total=listed_total,
        demand_total=demand_total,
        available_total=available_total,
        supply_leader=supply_leader,
        demand_leader=demand_leader,
    )

    if is_historical_summary:
        cards = [
            (f"Listed supply during {period_label}", listed_total, "kWh"),
            (f"Completed demand during {period_label}", demand_total, "kWh"),
            (f"Listings during {period_label}", listing_count, ""),
            (f"Completed purchases during {period_label}", purchase_count, ""),
        ]
    else:
        cards = [
            ("Currently available now", available_total, "kWh"),
            ("Active listings now", _int_or_zero(current_inventory.get("listing_count")), ""),
            (f"Listed supply during {period_label}", listed_total, "kWh"),
            (f"Completed demand during {period_label}", demand_total, "kWh"),
        ]
    card_html = "".join(
        _metric_card(label, value, unit)
        for label, value, unit in cards
        if value is not None
    )

    if is_historical_summary:
        candidates = [
            (f"Listed supply during {period_label}", "listed", "kWh"),
            (f"Completed demand during {period_label}", "demand", "kWh"),
            (f"Market balance during {period_label}", "balance", "kWh"),
            ("Average realized price", "realized", ""),
        ]
    else:
        candidates = [
            ("Currently available now", "available", "kWh"),
            (f"Listed supply during {period_label}", "listed", "kWh"),
            (f"Completed demand during {period_label}", "demand", "kWh"),
            (f"Market balance during {period_label}", "balance", "kWh"),
            ("Average current asking price", "asking", ""),
            ("Average realized price", "realized", ""),
        ]
    rows = []
    for source in settings.SUPPORTED_ENERGY_SOURCES:
        stats = source_stats.get(source, {}) if isinstance(source_stats, Mapping) else {}
        rows.append({
            "source": source.title(),
            "available": _number_or_none(stats.get("currently_available_supply_kwh_from_period")),
            "listed": _number_or_none(listed_by_source.get(source)),
            "demand": _number_or_none(demand_by_source.get(source)),
            "balance": _number_or_none(balance_by_source.get(source)),
            "asking": _number_or_none(stats.get("average_current_asking_price_per_kwh")),
            "realized": _number_or_none(stats.get("average_realized_price_per_kwh_in_period")),
        })

    visible_columns = [
        column
        for column in candidates
        if any(row[column[1]] is not None for row in rows)
    ]

    source_table = _responsive_table(
        headers=["Source"] + [column[0] for column in visible_columns],
        rows=[
            [row["source"]] + [_display_value(row[column[1]], column[2]) for column in visible_columns]
            for row in rows
        ],
    ) if visible_columns else ""

    activity_table = _responsive_table(
        headers=["Activity", "Count", "kWh"],
        rows=[
            [f"Listings during {period_label}", str(listing_count), _display_value(listed_total, "kWh")],
            [f"Completed purchases during {period_label}", str(purchase_count), _display_value(demand_total, "kWh")],
        ],
    )

    data_as_of = str(api_summary.data_as_of or "")[:10]
    scope_note = (
        f"Period: {html.escape(period_label)}"
        + (f" | Data as of: {html.escape(data_as_of)}" if len(data_as_of) == 10 else "")
    )

    return (
        '<section class="ai-response" style="box-sizing:border-box;width:100%;max-width:100%;'
        'display:flex;flex-direction:column;gap:14px;color:#172033;font-family:Arial,Helvetica,sans-serif;'
        'font-size:14px;line-height:1.5;overflow-wrap:anywhere;">'
        '<div class="ai-hero ai-info" style="box-sizing:border-box;padding:16px;border-radius:12px;'
        'background:#eff6ff;border-left:4px solid #2563eb;">'
        f'<h3 style="box-sizing:border-box;margin:0 0 6px;font-size:18px;">&#128200; {html.escape(period_label)} marketplace summary</h3>'
        f'<p style="box-sizing:border-box;margin:0;">{html.escape(summary)}</p></div>'
        f'<div style="box-sizing:border-box;width:100%;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;">{card_html}</div>'
        + (
            '<section style="box-sizing:border-box;width:100%;">'
            '<h4 style="box-sizing:border-box;margin:0 0 8px;font-size:15px;">Supply by renewable source</h4>'
            f'{source_table}</section>'
            if source_table else ""
        )
        + '<section style="box-sizing:border-box;width:100%;">'
        '<h4 style="box-sizing:border-box;margin:0 0 8px;font-size:15px;">Period activity</h4>'
        f'{activity_table}</section>'
        '<div class="ai-note" style="box-sizing:border-box;padding:11px 13px;border-radius:8px;'
        f'background:#f8fafc;color:#475569;font-size:13px;">{scope_note}</div>'
        '</section>'
    )


def _summary_period_label(period_from: str, period_to: str, is_today: bool) -> str:
    if is_today:
        return "Today"
    if period_from and period_from == period_to:
        return period_from
    if period_from and period_to:
        return f"{period_from} to {period_to}"
    return "Current"


def _is_historical_marketplace_summary(
    period_from: str,
    period_to: str,
    is_today: bool,
) -> bool:
    if is_today:
        return False
    if not period_from or not period_to:
        return False
    return True


def _marketplace_interpretation(
    period_label: str,
    listed_total: Optional[float],
    demand_total: Optional[float],
    available_total: Optional[float],
    supply_leader: Optional[str],
    demand_leader: Optional[str],
) -> str:
    del period_label
    parts: List[str] = []
    if supply_leader and demand_leader:
        if supply_leader == demand_leader:
            parts.append(f"{supply_leader.title()} led both listed supply and completed demand during the period.")
        else:
            parts.append(f"{supply_leader.title()} led listed supply, while {demand_leader.title()} led completed demand during the period.")
    elif supply_leader:
        parts.append(f"{supply_leader.title()} contributed the most listed supply during the period.")
    elif demand_leader:
        parts.append(f"{demand_leader.title()} recorded the most completed demand during the period.")
    else:
        parts.append("No source recorded material listed supply or completed demand during the period.")

    if listed_total is not None and demand_total is not None:
        if listed_total > demand_total:
            parts.append("Listed supply exceeded completed demand overall.")
        elif listed_total < demand_total:
            parts.append("Completed demand exceeded listed supply overall.")
        else:
            parts.append("Listed supply and completed demand were balanced overall.")

    # Current inventory language only belongs in today summaries.
    return " ".join(parts)


def _metric_card(label: str, value: Any, unit: str) -> str:
    return (
        '<div class="ai-card" style="box-sizing:border-box;padding:12px;border:1px solid #dbe3ef;'
        'border-radius:10px;background:#ffffff;">'
        f'<div style="box-sizing:border-box;color:#64748b;font-size:12px;">{html.escape(label)}</div>'
        f'<div style="box-sizing:border-box;font-size:17px;font-weight:700;">{html.escape(_display_value(value, unit))}</div>'
        '</div>'
    )


def _responsive_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not headers or not rows:
        return ""

    th = "".join(
        '<th style="box-sizing:border-box;padding:10px 12px;border:1px solid #cbd5e1;'
        'background:#f1f5f9;text-align:left;vertical-align:middle;font-weight:700;white-space:nowrap;">'
        f'{html.escape(str(value))}</th>'
        for value in headers
    )
    body = "".join(
        '<tr style="box-sizing:border-box;">'
        + "".join(
            '<td style="box-sizing:border-box;padding:10px 12px;border:1px solid #cbd5e1;'
            'text-align:left;vertical-align:middle;white-space:normal;">'
            f'{html.escape(str(value))}</td>'
            for value in row
        )
        + '</tr>'
        for row in rows
    )
    return (
        '<div class="ai-table-wrap" style="box-sizing:border-box;width:100%;max-width:100%;'
        'overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid #cbd5e1;border-radius:10px;">'
        '<table class="ai-table" style="box-sizing:border-box;width:100%;min-width:640px;'
        'border-collapse:collapse;border-spacing:0;background:#ffffff;">'
        f'<thead style="box-sizing:border-box;"><tr style="box-sizing:border-box;">{th}</tr></thead>'
        f'<tbody style="box-sizing:border-box;">{body}</tbody></table></div>'
    )


def _display_value(value: Any, unit: str = "") -> str:
    number = _number_or_none(value)
    if number is None:
        return "-"
    rendered = str(int(number)) if float(number).is_integer() else f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}".strip()


def _trend_display(value: Any) -> str:
    number = _number_or_none(value)
    if number is None:
        return "-"
    magnitude = abs(number)
    rendered = str(int(magnitude)) if float(magnitude).is_integer() else f"{magnitude:.2f}".rstrip("0").rstrip(".")
    if number > 0:
        return f"Up {rendered}%"
    if number < 0:
        return f"Down {rendered}%"
    return "Flat"


def _number_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _positive_leader(values: Any) -> Optional[str]:
    if not isinstance(values, Mapping):
        return None
    valid = [
        (str(key), number)
        for key, value in values.items()
        if (number := _number_or_none(value)) is not None and number > 0
    ]
    return max(valid, key=lambda item: item[1])[0] if valid else None


_VOID_TAGS = {
    "br",
    "hr",
    "img",
    "input",
    "meta",
    "link",
    "source",
    "area",
    "base",
    "col",
    "embed",
    "param",
    "track",
    "wbr",
}


class _FragmentBalanceParser(HTMLParser):
    """Track whether an HTML fragment ends with all non-void tags closed."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[str] = []
        self.invalid = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        del attrs
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        del tag
        del attrs
        return

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if not self.stack or self.stack[-1] != tag:
            self.invalid = True
            return
        self.stack.pop()


def _is_complete_html_fragment(value: str) -> bool:
    """Reject truncated or structurally unbalanced LLM HTML."""
    if not isinstance(value, str) or not value.strip():
        return False
    parser = _FragmentBalanceParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return False
    return not parser.invalid and not parser.stack and value.rstrip().endswith(">")


def _render_seller_recommendation_html(api_summary: QueryApiSummary) -> str:
    """Render seller recommendations deterministically and completely."""
    recommendation = dict(api_summary.recommendation_result or {})
    confidence = str(api_summary.confidence or "insufficient_data")
    source = recommendation.get("recommended_source")
    no_preference = bool(recommendation.get("no_strong_preference", False))
    scores = recommendation.get("scores_by_source", {}) or {}
    factors = recommendation.get("factors_by_source", {}) or {}

    if no_preference or not source:
        finding = (
            "The available marketplace evidence does not show a strong preference "
            "for one renewable source."
        )
    elif confidence == "insufficient_data":
        finding = (
            f"{str(source).title()} appears to be the best listing option based on limited available history."
        )
    else:
        finding = (
            f"{str(source).title()} has the strongest current listing opportunity "
            "based on the supplied marketplace factors."
        )

    rows = []
    for energy_source in settings.SUPPORTED_ENERGY_SOURCES:
        source_factors = factors.get(energy_source, {}) if isinstance(factors, Mapping) else {}
        rows.append(
            [
                energy_source.title(),
                _display_value(scores.get(energy_source), ""),
                _display_value(source_factors.get("active_supply_kwh"), "kWh"),
                _display_value(source_factors.get("demand_supply_ratio"), ""),
                _display_value(source_factors.get("weighted_realized_price"), ""),
            ]
        )

    table = _responsive_table(
        headers=[
            "Source",
            "Opportunity score",
            "Active supply",
            "Demand-to-supply ratio",
            "Realized price",
        ],
        rows=rows,
    )
    data_as_of = str(api_summary.data_as_of or "")[:10]
    limitations = list(api_summary.limitations or [])
    limitation_html = ""
    if limitations:
        limitation_html = (
            f'<div style="{ui_constants.LIMITATION_STYLE}">'
            + html.escape(" ".join(limitations))
            + "</div>"
        )

    return (
        f'<section style="{ui_constants.SECTION_STYLE}">'
        f'<div style="{ui_constants.HERO_SUCCESS_STYLE}">'
        f'<h3 style="{ui_constants.HEADING_STYLE}">&#10024; Recommendation</h3>'
        f'<p style="{ui_constants.PARAGRAPH_STYLE}">{html.escape(finding)}</p></div>'
        f'<section style="{ui_constants.SUBSECTION_STYLE}">'
        f'<h4 style="{ui_constants.SUBHEADING_STYLE}">Supporting metrics</h4>'
        f"{table}</section>"
        f'<div style="{ui_constants.NOTE_STYLE}">Confidence: {html.escape(confidence.replace("_", " ").title())}'
        + (f' | Data as of: {html.escape(data_as_of)}' if len(data_as_of) == 10 else "")
        + "</div>"
        + limitation_html
        + "</section>"
    )


def _human_label(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _scalar_display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _render_complete_api_fallback(api_summary: QueryApiSummary) -> str:
    """Return complete responsive HTML when any LLM fragment is truncated."""
    intent = str(api_summary.intent or "analytics")
    analytics = dict(api_summary.analytics_result or {})
    prediction = api_summary.prediction_result
    recommendation = api_summary.recommendation_result

    if intent == "seller_recommendation":
        return _render_seller_recommendation_html(api_summary)

    payload = recommendation or prediction or analytics
    rows: List[List[str]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                rows.append([_human_label(str(key)), _scalar_display(value)])

    table = _responsive_table(["Metric", "Value"], rows) if rows else ""
    data_as_of = str(api_summary.data_as_of or "")[:10]
    return (
        f'<section style="{ui_constants.SECTION_STYLE}">'
        f'<div style="{ui_constants.HERO_INFO_STYLE}">'
        f'<h3 style="box-sizing:border-box;margin:0;font-size:18px;">{html.escape(_human_label(intent))}</h3></div>'
        + (f'<section style="{ui_constants.SUBSECTION_STYLE}">{table}</section>' if table else "")
        + f'<div style="{ui_constants.NOTE_STYLE}">Data as of: {html.escape(data_as_of) if len(data_as_of) == 10 else "-"}</div>'
        "</section>"
    )


def _filter_argument(api_summary: QueryApiSummary, key: str) -> Optional[str]:
    for item in list(api_summary.filters_used or []):
        if not isinstance(item, Mapping):
            continue
        arguments = item.get("arguments", {})
        if isinstance(arguments, Mapping) and arguments.get(key) not in (None, ""):
            return str(arguments[key])
    return None


def _render_current_supply_html(api_summary: QueryApiSummary) -> str:
    analytics = dict(api_summary.analytics_result or {})
    requested_source = _filter_argument(api_summary, "energy_source")
    requested_location = _filter_argument(api_summary, "location")
    matching = analytics.get("matching_listings", []) or []
    if requested_source:
        matching = [
            item
            for item in matching
            if isinstance(item, Mapping)
            and str(item.get("energy_source", "")).upper() == requested_source.upper()
        ]
    if requested_location:
        matching = [
            item
            for item in matching
            if isinstance(item, Mapping)
            and requested_location.lower() in str(item.get("location", "")).lower()
        ]

    if requested_source or requested_location:
        source_label = requested_source.title() if requested_source else "Renewable"
        location_label = requested_location or "the requested location"
        if matching:
            total = sum(
                _number_or_none(item.get("energy_kwh")) or 0.0
                for item in matching
                if isinstance(item, Mapping)
            )
            finding = (
                f"{len(matching)} {source_label} listing{'s are' if len(matching) != 1 else ' is'} currently "
                f"available in {location_label}, totaling {_display_value(total, 'kWh')}."
            )
        else:
            finding = f"No currently available {source_label} listings matched {location_label}."

        rows: List[List[str]] = []
        for item in matching:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                [
                    str(item.get("credit_reference") or item.get("listing_id") or "-"),
                    str(item.get("energy_source") or "-").title(),
                    _display_value(item.get("energy_kwh"), "kWh"),
                    _display_value(item.get("price_per_kwh"), ""),
                    str(item.get("location") or "-"),
                ]
            )
        table = _responsive_table(
            ["Credit / Listing", "Source", "Available supply", "Price per kWh", "Location"],
            rows,
        ) if rows else ""
        title = "Available marketplace credits"
    else:
        leader = analytics.get("highest_supply_source")
        leader_value = analytics.get("highest_supply_kwh")
        total = analytics.get("total_active_supply_kwh")
        if leader:
            finding = (
                f"{str(leader).title()} is dominating current marketplace inventory with "
                f"{_display_value(leader_value, 'kWh')} available, out of "
                f"{_display_value(total, 'kWh')} total active supply."
            )
        else:
            finding = "No active marketplace supply is currently available."

        supply = analytics.get("supply_by_source_kwh", {}) or {}
        rows = sorted(
            [[str(source).title(), _display_value(value, "kWh")] for source, value in supply.items()],
            key=lambda row: _number_or_none(row[1].split()[0]) or 0.0,
            reverse=True,
        )
        table = _responsive_table(["Source", "Available supply"], rows)
        title = "Current marketplace supply"

    return _render_standard_answer(title, finding, table, api_summary)


def _render_demand_and_supply_html(api_summary: QueryApiSummary) -> str:
    analytics = dict(api_summary.analytics_result or {})
    requested_source = _filter_argument(api_summary, "energy_source")
    period = _period_caption_fixed(api_summary)

    remaining = analytics.get("remaining_supply_kwh_by_source", {}) or {}
    sold = analytics.get("sold_supply_kwh_by_source", {}) or {}
    total = analytics.get("total_supply_kwh_by_source", {}) or {}
    demand = analytics.get("realized_demand_kwh_by_source", {}) or {}

    sources = [requested_source] if requested_source else list(settings.SUPPORTED_ENERGY_SOURCES)
    rows: List[List[str]] = []
    for source in sources:
        if not source:
            continue
        rows.append(
            [
                str(source).title(),
                _display_value(remaining.get(source), "kWh"),
                _display_value(sold.get(source), "kWh"),
                _display_value(total.get(source), "kWh"),
                _display_value(demand.get(source), "kWh"),
            ]
        )

    if requested_source:
        source_label = requested_source.title()
        finding = (
            f"During {period}, {source_label} total supply was {_display_value(total.get(requested_source), 'kWh')}, "
            f"comprising {_display_value(remaining.get(requested_source), 'kWh')} remaining and "
            f"{_display_value(sold.get(requested_source), 'kWh')} sold. Realized demand was "
            f"{_display_value(demand.get(requested_source), 'kWh')}."
        )
    else:
        leader = analytics.get("highest_total_supply_source")
        finding = f"During {period}, {str(leader).title() if leader else 'no source'} had the highest total supply."

    table = _responsive_table(
        ["Source", "Remaining supply", "Sold supply", "Total supply", "Realized demand"],
        rows,
    )
    return _render_standard_answer("Demand and supply", finding, table, api_summary, period)


def _render_historical_demand_html(api_summary: QueryApiSummary) -> str:
    analytics = dict(api_summary.analytics_result or {})
    demand = analytics.get("demand_kwh_by_source", {}) or {}
    shares = analytics.get("demand_share_percentage", {}) or {}
    ranking = list(analytics.get("demand_ranking_desc", []) or [])
    requested = _requested_sources(api_summary)
    ordered = [source for source in ranking if not requested or source in requested]
    if not ordered:
        ordered = requested or list(settings.SUPPORTED_ENERGY_SOURCES)
    leader = ordered[0] if ordered else None
    period = _period_caption_fixed(api_summary)
    finding = (
        f"{str(leader).title()} had the most completed demand during {period}, at "
        f"{_display_value(demand.get(leader), 'kWh')}."
        if leader else f"No completed demand was recorded during {period}."
    )
    rows = [
        [str(source).title(), _display_value(demand.get(source), "kWh"), _display_value(shares.get(source), "%")]
        for source in ordered
    ]
    table = _responsive_table(["Source", "Completed demand", "Demand share"], rows)
    return _render_standard_answer("Demand comparison", finding, table, api_summary, period)


def _render_supply_mix_html(api_summary: QueryApiSummary) -> str:
    analytics = dict(api_summary.analytics_result or {})
    supply = analytics.get("active_supply_kwh_by_source", {}) or {}
    shares = analytics.get("supply_mix_percentage", {}) or {}
    ordered = sorted(
        settings.SUPPORTED_ENERGY_SOURCES,
        key=lambda source: _number_or_none(shares.get(source)) or 0.0,
        reverse=True,
    )
    leader = ordered[0] if ordered else None
    finding = (
        f"{str(leader).title()} has the largest share of active marketplace supply at "
        f"{_display_value(shares.get(leader), '%')}."
        if leader else "No active marketplace supply is available."
    )
    rows = [
        [source.title(), _display_value(supply.get(source), "kWh"), _display_value(shares.get(source), "%")]
        for source in ordered
    ]
    return _render_standard_answer(
        "Active marketplace supply mix",
        finding,
        _responsive_table(["Source", "Active supply", "Market share"], rows),
        api_summary,
    )


def _render_standard_answer(
    title: str,
    finding: str,
    table: str,
    api_summary: QueryApiSummary,
    period: Optional[str] = None,
) -> str:
    data_as_of = str(api_summary.data_as_of or "")[:10]
    note_parts: List[str] = []
    if period:
        note_parts.append(f"Period: {period}")
    if len(data_as_of) == 10:
        note_parts.append(f"Data as of: {data_as_of}")
    note = " | ".join(note_parts)

    return (
        f'<section style="{ui_constants.SECTION_STYLE}">'
        f'<div style="{ui_constants.HERO_INFO_STYLE}">'
        f'<h3 style="{ui_constants.HEADING_STYLE}">{html.escape(title)}</h3>'
        f'<p style="{ui_constants.PARAGRAPH_STYLE}">{html.escape(finding)}</p></div>'
        + (f'<section style="{ui_constants.SUBSECTION_STYLE}">{table}</section>' if table else "")
        + (
            f'<div style="{ui_constants.NOTE_STYLE}">{html.escape(note)}</div>'
            if note
            else ""
        )
        + "</section>"
    )


def _clear_embedded_source_metadata(
    api_summary: Optional[QueryApiSummary],
) -> Optional[QueryApiSummary]:
    """Avoid duplicate legacy source rendering after HTML embedding."""
    if api_summary is not None:
        api_summary.tool_results = []  # type: ignore
    return api_summary


def answer_question(request: QueryRequest) -> QueryResponse:
    """Run the complete GreenGrid grounded-answer pipeline."""
    top_k = request.top_k or settings.OPENSEARCH_TOP_K

    # Plan and API execution do not depend on successful RAG retrieval.
    api_summary, api_facts_used, plan = _resolve_api_summary(request.question)

    # New live-data safeguard: if the planner marks the question as requiring
    # live marketplace data and the APIs return neither records nor aggregates,
    # stop before RAG retrieval and answer generation. This prevents document
    # examples from being presented as live marketplace facts.
    if plan.get("requires_live_data", False) and not api_facts_used:
        return QueryResponse(
            answer=_live_data_unavailable_response(api_summary),
            source_count=0,
            sources=[],
            answer_mode="insufficient_data",
            api_facts_used=False,
            api_summary=api_summary,
        )

    if str(plan.get("intent", "none")) in {"marketplace_summary", "demand_and_supply"}:
        hits: List[Dict[str, Any]] = []
    else:
        hits = _retrieve_chunks(request.question, top_k)
    sources = _build_sources(hits)

    if not hits and api_summary is None:
        return QueryResponse(
            answer=_insufficient_response(None, plan),
            source_count=0,
            sources=[],
            answer_mode="insufficient_data",
            api_facts_used=False,
            api_summary=None,
        )

    prompt = prompt_service.build_rag_prompt(
        question=request.question,
        retrieved_chunks=hits,
        api_context=(
            api_summary.model_dump(by_alias=True)
            if api_summary is not None
            else None
        ),
    )

    intent = str(plan.get("intent", "none"))
    question_text = request.question.lower()
    supply_focus_question = (
        any(term in question_text for term in ("supply", "availability", "listings", "listed"))
        and "shortage" not in question_text
        and "short in" not in question_text
        and "shortfall" not in question_text
    )
    if api_summary is not None and "shortage" in question_text and ("last month" in question_text or "previous month" in question_text):
        answer = _render_historical_shortage_or_insufficient(api_summary)
    elif api_summary is not None and "recently listed" in question_text:
        answer = _render_recent_listings_html(api_summary)
    elif api_summary is not None and ("what price should i set" in question_text or "price should i set" in question_text):
        answer = _render_price_guidance_or_insufficient(api_summary)
    elif intent == "marketplace_summary" and api_summary is not None:
        answer = _render_marketplace_summary_html(api_summary)
    elif intent == "current_supply" and api_summary is not None:
        answer = _render_current_supply_html(api_summary)
    elif intent == "demand_and_supply" and api_summary is not None:
        answer = _render_demand_and_supply_html(api_summary)
    elif intent == "historical_demand" and api_summary is not None:
        answer = _render_historical_demand_html(api_summary)
    elif intent == "supply_mix" and api_summary is not None:
        answer = _render_supply_mix_html(api_summary)
    elif intent == "demand_prediction" and api_summary is not None:
        answer = _render_demand_prediction_or_insufficient(api_summary)
    elif intent == "shortage_prediction" and api_summary is not None:
        answer = _render_supply_forecast_from_shortage(api_summary) if supply_focus_question else _render_shortage_prediction_or_insufficient(api_summary)
    elif intent == "price_prediction" and api_summary is not None:
        answer = _render_price_prediction_or_insufficient(api_summary)
    elif intent == "demand_supply_ratio" and api_summary is not None:
        answer = _render_ratio_html(api_summary)
    elif intent == "buyer_recommendation" and api_summary is not None:
        answer = _render_buyer_recommendation_or_insufficient(api_summary)
    elif intent == "seller_recommendation" and api_summary is not None:
        answer = _render_seller_recommendation_html(api_summary)
    else:
        try:
            answer = bedrock_service.generate_answer(prompt)
            if not answer.strip():
                raise RuntimeError("Bedrock returned an empty answer.")
            if not _is_complete_html_fragment(answer):
                if api_summary is not None:
                    logger.warning("Replacing incomplete LLM HTML for intent=%s", intent)
                    answer = _render_complete_api_fallback(api_summary)
                else:
                    raise RuntimeError("Bedrock returned incomplete HTML.")
        except (RuntimeError, ValueError) as exc:
            logger.warning("Final answer generation unavailable: %s", exc)
            answer = _build_fallback_answer(api_summary, hits, exc)

    # Keep source metadata outside response.answer. The frontend may ignore or
    # render QueryResponse.sources and api_summary.tool_results separately.
    # The generated HTML answer must not contain a Sources Used section.
    mode = _answer_mode(bool(hits), api_summary, api_facts_used)

    logger.info(
        "answer_question: top_k=%d sources=%d mode=%s "
        "api_facts_used=%s intent=%s",
        top_k,
        len(sources),
        mode,
        api_facts_used,
        plan.get("intent", "none"),
    )

    return QueryResponse(
        answer=answer,
        source_count=len(sources),
        sources=sources,
        answer_mode=mode,
        api_facts_used=api_facts_used,
        api_summary=api_summary,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _append_unique(values: Sequence[str], new_value: str) -> List[str]:
    output = list(values)
    if new_value not in output:
        output.append(new_value)
    return output


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return message[:500]
