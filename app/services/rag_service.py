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
import logging
from datetime import datetime, timezone
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

    cards = [
        ("Currently available from period", available_total, "kWh"),
        ("Active listings from period", _int_or_zero(current_inventory.get("listing_count")), ""),
        ("Listed supply in period", listed_total, "kWh"),
        ("Completed demand in period", demand_total, "kWh"),
    ]
    card_html = "".join(
        _metric_card(label, value, unit)
        for label, value, unit in cards
        if value is not None
    )

    candidates = [
        ("Currently available from period", "available", "kWh"),
        ("Listed supply in period", "listed", "kWh"),
        ("Completed demand in period", "demand", "kWh"),
        ("Market balance in period", "balance", "kWh"),
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
            ["Listings in period", str(listing_count), _display_value(listed_total, "kWh")],
            ["Completed purchases in period", str(purchase_count), _display_value(demand_total, "kWh")],
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
        return "Today's"
    if period_from and period_from == period_to:
        return period_from
    if period_from and period_to:
        return f"{period_from} to {period_to}"
    return "Marketplace"


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

    if available_total is not None and available_total > 0:
        parts.append("Some inventory originating in the period remains available now.")
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


def _append_collapsible_sources(
    answer: str,
    sources: Sequence[QuerySource],
    api_summary: Optional[QueryApiSummary],
) -> str:
    """Append one collapsible source section after the complete answer."""
    document_names: List[str] = []
    for source in sources:
        name = str(getattr(source, "document_name", "") or "").strip()
        if name and name not in document_names:
            document_names.append(name)

    tool_names: List[str] = []
    if api_summary is not None:
        for result in list(getattr(api_summary, "tool_results", []) or []):
            name = str(getattr(result, "tool", "") or "").strip()
            if name and name not in tool_names:
                tool_names.append(name)

    if not document_names and not tool_names:
        return answer

    sections: List[str] = []
    if document_names:
        document_items = "".join(
            (
                '<li style="box-sizing:border-box;margin:0;padding:0;">'
                f'&#128196; {html.escape(name)}'
                "</li>"
            )
            for name in document_names
        )
        sections.append(
            '<div style="box-sizing:border-box;display:flex;flex-direction:column;gap:6px;">'
            '<strong style="box-sizing:border-box;">Documents</strong>'
            '<ul style="box-sizing:border-box;margin:0;padding-left:22px;display:flex;flex-direction:column;gap:4px;">'
            f"{document_items}</ul></div>"
        )

    if tool_names:
        tool_items = "".join(
            (
                '<li style="box-sizing:border-box;margin:0;padding:0;">'
                f'&#127760; {html.escape(name)}'
                "</li>"
            )
            for name in tool_names
        )
        sections.append(
            '<div style="box-sizing:border-box;display:flex;flex-direction:column;gap:6px;">'
            '<strong style="box-sizing:border-box;">Marketplace Data</strong>'
            '<ul style="box-sizing:border-box;margin:0;padding-left:22px;display:flex;flex-direction:column;gap:4px;">'
            f"{tool_items}</ul></div>"
        )

    disclosure = (
        '<details style="box-sizing:border-box;width:100%;max-width:100%;margin-top:14px;'
        'padding:10px 12px;border:1px solid #dbe3ef;border-radius:10px;'
        'background:#f8fafc;color:#334155;">'
        '<summary style="box-sizing:border-box;cursor:pointer;font-weight:700;'
        'list-style-position:inside;">&#128204; Sources Used</summary>'
        '<div style="box-sizing:border-box;display:flex;flex-direction:column;gap:12px;'
        'margin-top:10px;">'
        + "".join(sections)
        + "</div></details>"
    )
    return f"{answer.rstrip()}\n{disclosure}"


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

    if str(plan.get("intent", "none")) == "marketplace_summary" and api_summary is not None:
        # Deterministic rendering avoids LLM truncation for large inline-styled tables.
        answer = _render_marketplace_summary_html(api_summary)
    else:
        try:
            answer = bedrock_service.generate_answer(prompt)
            if not answer.strip():
                raise RuntimeError("Bedrock returned an empty answer.")
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
