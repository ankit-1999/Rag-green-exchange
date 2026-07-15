"""
prompt_service.py
-----------------

Builds the final grounded answer prompt for GreenGrid Exchange.

The final LLM answer is returned as a polished HTML fragment so the frontend can
render it directly inside the chat message. The response must never contain a
complete HTML document, JavaScript, CSS, event handlers, forms, or unsafe URLs.

The prompt combines:
- Retrieved RAG chunks for rules and explanations
- Compact marketplace API metadata and normalized API aggregates
- Deterministic analytics, prediction, and recommendation results
- The user's question

The LLM explains calculated results. The LLM must not recalculate, modify, or
invent marketplace values.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.config import settings


SUPPORTED_ENERGY_SOURCES = tuple(
    getattr(
        settings,
        "SUPPORTED_ENERGY_SOURCES",
        (
            "SOLAR",
            "WIND",
            "HYDRO",
            "BIOMASS",
            "GEOTHERMAL",
            "TIDAL",
            "OTHER",
        ),
    )
)

SOURCE_DISPLAY_NAMES = {
    "SOLAR": "Solar",
    "WIND": "Wind",
    "HYDRO": "Hydro",
    "BIOMASS": "Biomass",
    "GEOTHERMAL": "Geothermal",
    "TIDAL": "Tidal",
    "OTHER": "Other",
}

SOURCE_SYMBOLS = {
    "SOLAR": "&#9728;&#65039;",
    "WIND": "&#127788;&#65039;",
    "HYDRO": "&#128167;",
    "BIOMASS": "&#127807;",
    "GEOTHERMAL": "&#127755;",
    "TIDAL": "&#127754;",
    "OTHER": "&#9889;",
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

ANALYTICS_INTENTS = {
    "historical_supply",
    "historical_demand",
    "demand_and_supply",
    "average_selling_price",
    "demand_supply_ratio",
    "market_balance",
    "supply_stability",
    "price_volatility",
    "supply_by_location",
    "marketplace_summary",
}

ALLOWED_HTML_TAGS = (
    "section",
    "div",
    "p",
    "h3",
    "h4",
    "ul",
    "ol",
    "li",
    "strong",
    "span",
    "small",
    "br",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
)

ALLOWED_HTML_CLASSES = (
    "ai-response",
    "ai-hero",
    "ai-title",
    "ai-subtitle",
    "ai-section",
    "ai-section-title",
    "ai-highlight",
    "ai-success",
    "ai-warning",
    "ai-danger",
    "ai-info",
    "ai-neutral",
    "ai-grid",
    "ai-card",
    "ai-card-title",
    "ai-card-value",
    "ai-card-label",
    "ai-list",
    "ai-metric-list",
    "ai-metric-row",
    "ai-metric-name",
    "ai-metric-value",
    "ai-badge",
    "ai-badge-high",
    "ai-badge-medium",
    "ai-badge-low",
    "ai-badge-insufficient",
    "ai-table",
    "ai-note",
    "ai-muted",
    "ai-source-solar",
    "ai-source-wind",
    "ai-source-hydro",
    "ai-source-biomass",
    "ai-source-geothermal",
    "ai-source-tidal",
    "ai-source-other",
)


# ---------------------------------------------------------------------------
# Serialization and normalization
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> str:
    """Serialize values not handled by the standard JSON encoder."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _json_text(value: Any) -> str:
    """Serialize prompt context deterministically."""
    return json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )


def _normalize_question(question: str) -> str:
    normalized = (question or "").strip()
    if not normalized:
        raise ValueError("Question cannot be empty.")
    return normalized


def _needs_elaborate_response(question: str) -> bool:
    """Detect questions that explicitly request explanation or methodology."""
    normalized = question.lower()
    keywords = (
        "explain",
        "elaborate",
        "detail",
        "detailed",
        "why",
        "how",
        "method",
        "calculation",
        "compare",
        "analysis",
        "factors",
    )
    return any(keyword in normalized for keyword in keywords)


def _supported_sources_text() -> str:
    return ", ".join(
        SOURCE_DISPLAY_NAMES.get(source, source.title())
        for source in SUPPORTED_ENERGY_SOURCES
    )


def _supported_sources_lines() -> str:
    return "\n".join(
        f"- {source} ({SOURCE_DISPLAY_NAMES.get(source, source.title())})"
        for source in SUPPORTED_ENERGY_SOURCES
    )


def _source_class_lines() -> str:
    return "\n".join(
        f"- ai-source-{source.lower().replace('_', '-')}"
        for source in SUPPORTED_ENERGY_SOURCES
    )


def _source_symbol_lines() -> str:
    return "\n".join(
        f"- {SOURCE_DISPLAY_NAMES.get(source, source.title())}: "
        f"{SOURCE_SYMBOLS.get(source, '&#9889;')}"
        for source in SUPPORTED_ENERGY_SOURCES
    )


# ---------------------------------------------------------------------------
# RAG source construction
# ---------------------------------------------------------------------------


def _build_context_blocks(
    retrieved_chunks: Sequence[Mapping[str, Any]],
) -> str:
    """Build RAG context blocks with stable source identifiers."""
    blocks: List[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = str(chunk.get("text", "") or "").strip()
        if not text:
            continue

        blocks.append(
            "\n".join(
                [
                    f"[SOURCE {index}]",
                    f"document_name: {chunk.get('document_name', 'unknown')}",
                    f"document_type: {chunk.get('document_type', 'unknown')}",
                    f"chunk_id: {chunk.get('chunk_id', 'unknown')}",
                    f"chunk_index: {chunk.get('chunk_index', 'unknown')}",
                    "content:",
                    text,
                ]
            )
        )

    return "\n\n".join(blocks) if blocks else "None"


# ---------------------------------------------------------------------------
# API context compaction
# ---------------------------------------------------------------------------


def _compact_api_context(
    api_context: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Keep calculated results, normalized API aggregates, and execution metadata
    while removing complete raw API record arrays from the final LLM prompt.
    """
    if not api_context:
        return None

    compact: Dict[str, Any] = {
        "context_type": api_context.get("context_type"),
        "planner_reason": api_context.get("planner_reason"),
        "intent": api_context.get("intent", "none"),
        "is_prediction": bool(api_context.get("is_prediction", False)),
        "is_recommendation": bool(api_context.get("is_recommendation", False)),
        "historical_period": api_context.get("historical_period"),
        "forecast_period": api_context.get("forecast_period"),
        "group_by": api_context.get("group_by", []),
        "metrics": api_context.get("metrics", []),
        "filters_used": api_context.get("filters_used", []),
        "records_analyzed": api_context.get("records_analyzed", {}),
        "analytics_result": api_context.get("analytics_result", {}),
        "prediction_result": api_context.get("prediction_result"),
        "recommendation_result": api_context.get("recommendation_result"),
        "confidence": api_context.get("confidence"),
        "calculation_method": api_context.get("calculation_method"),
        "limitations": api_context.get("limitations", []),
        "missing_parameters": api_context.get("missing_parameters", []),
        "data_as_of": api_context.get("data_as_of"),
        "tool_results": [],
    }

    raw_tool_results = api_context.get("tool_results", [])
    if isinstance(raw_tool_results, list):
        for result in raw_tool_results:
            if not isinstance(result, Mapping):
                continue

            data = result.get("data", {})
            sample_records: List[Any] = []
            aggregates: Dict[str, Any] = {}
            response_metadata: Dict[str, Any] = {}

            if isinstance(data, Mapping):
                possible_samples = data.get("sample_records", [])
                if isinstance(possible_samples, list):
                    sample_records = possible_samples[
                        : settings.ANALYTICS_LLM_SAMPLE_RECORDS
                    ]

                possible_aggregates = data.get("aggregates", {})
                if isinstance(possible_aggregates, Mapping):
                    aggregates = dict(possible_aggregates)

                possible_metadata = data.get("response_metadata", {})
                if isinstance(possible_metadata, Mapping):
                    response_metadata = dict(possible_metadata)

            compact["tool_results"].append(
                {
                    "tool": result.get("tool"),
                    "endpoint": result.get("endpoint"),
                    "arguments": result.get("arguments", {}),
                    "record_count": result.get("record_count", 0),
                    "pages_fetched": result.get("pages_fetched", 0),
                    "execution_status": result.get("execution_status"),
                    "error": result.get("error"),
                    "sample_records": sample_records,
                    "aggregates": aggregates,
                    "response_metadata": response_metadata,
                }
            )

    return compact


# ---------------------------------------------------------------------------
# HTML formatting guidance
# ---------------------------------------------------------------------------


def _html_design_system() -> str:
    """Return the fixed visual vocabulary allowed in model-generated HTML."""
    return f"""
HTML DESIGN SYSTEM

Return one safe HTML fragment. Use only these tags:
{', '.join(ALLOWED_HTML_TAGS)}.

Allowed class names:
{chr(10).join(f'- {class_name}' for class_name in ALLOWED_HTML_CLASSES)}

Source-specific class names:
{_source_class_lines()}

Never use style attributes. Never invent class names outside this list.
Never use scripts, links, images, forms, inputs, buttons, iframes, SVG, canvas,
video, audio, object, embed, meta, base, or event-handler attributes.
Never include id, href, src, onclick, onerror, onload, data-*, aria-*, role,
title, target, contenteditable, or any other attributes.
The only permitted attribute is class with one or more allowed class names.

Use Unicode symbols sparingly:
- Prediction: &#128302;
- Recommendation: &#10024;
- Analytics: &#128200;
- Success: &#9989;
- Warning: &#9888;&#65039;
- Information: &#8505;&#65039;
- Location: &#128205;
- Confidence: &#127919;
{_source_symbol_lines()}

TABLE RULES:
- Whenever two or more sources, locations, listings, periods, or options are
  compared, use a real HTML table.
- Use exactly this nesting:
  <table class="ai-table"><thead><tr><th>...</th></tr></thead>
  <tbody><tr><td>...</td></tr></tbody></table>
- Put every heading in its own th element.
- Put every value in its own td element.
- Never flatten a table into plain text.
- When a general source comparison is requested, include all supported sources:
  {_supported_sources_text()}.
- If a supported source has no value, show 0 for quantities or percentages and
  Not available for unavailable prices or predictions.

VISIBLE-CONTENT RULES:
- Never emit instructional placeholder text such as "Lead with...", "Use this
  structure", "Use metric rows", "Only supplied factors", or "...".
- Every visible sentence must be a final answer grounded in supplied context.
- Never include Markdown fences such as ```html or ```.
- Do not display internal enum keys or raw JSON keys. Convert insufficient_data
  to the human-readable label "Insufficient data".

Prefer short cards, clear labels, compact bullet lists, and small tables.
Do not create decorative content that is not supported by the data.
""".strip()


def _response_template(
    question: str,
    api_context: Optional[Mapping[str, Any]],
) -> str:
    """Return intent-specific HTML structure requirements."""
    if not api_context:
        if _needs_elaborate_response(question):
            return """
Create one ai-response section containing:
1. An ai-hero ai-info block with a concise Direct answer title and final answer.
2. An ai-section with an Explanation heading and grounded explanation.
3. An ai-note ai-neutral block only when a rule or limitation is relevant.
Do not copy these instructions into the visible response.
""".strip()
        return """
Create one compact ai-response section with an ai-hero ai-info block containing
an Answer title and a direct final answer. Do not copy this instruction.
""".strip()

    intent = str(api_context.get("intent", "none") or "none")
    is_prediction = (
        bool(api_context.get("is_prediction", False))
        or intent in PREDICTION_INTENTS
    )
    is_recommendation = (
        bool(api_context.get("is_recommendation", False))
        or intent in RECOMMENDATION_INTENTS
    )
    confidence = str(api_context.get("confidence", "") or "").lower()

    if is_prediction:
        state_class = (
            "ai-warning"
            if confidence == "insufficient_data"
            else "ai-info"
        )
        return f"""
Create one ai-response section containing:
1. An ai-hero {state_class} block titled &#128302; Prediction with the actual
   prediction, or a clear insufficient-data statement.
2. An ai-grid with cards for predicted leader, expected value or range, and
   confidence when those values are supplied.
3. An ai-section titled &#128200; Key figures containing a proper ai-table when
   multiple sources are compared.
4. An ai-section titled Data period and method with exact supplied dates and the
   supplied calculation method.
5. An ai-section titled Factors considered using only supplied factors.
6. An ai-note ai-warning block containing supplied limitations and stating that
   forecasts are not guaranteed.
Use the confidence classes high, medium, low, or insufficient as applicable.
Do not copy these instructions or placeholders into the response.
""".strip()

    if is_recommendation:
        return """
Create one ai-response section containing:
1. An ai-hero ai-success block titled &#10024; Recommendation with the actual
   recommendation. If no_strong_preference is true, explicitly state that no
   strong preference exists. If confidence is insufficient_data, do not force a
   recommendation.
2. An ai-section titled Why with concise grounded reasons.
3. An ai-section titled &#128200; Supporting metrics containing a proper ai-table
   when multiple sources or listings are compared.
4. An ai-section titled &#127919; Confidence with the correct ai-badge class.
5. An ai-note ai-warning block containing supplied limitations and stating that
   the result is decision support, not a guarantee or financial advice.
Do not copy these instructions or placeholders into the response.
""".strip()

    if intent == "demand_and_supply":
        return """
Create one ai-response section containing:
1. An ai-hero ai-info block titled &#128200; Demand and supply with the direct
   answer for the requested source, location, and period.
2. An ai-section titled Key metrics with a proper ai-table. For a single-source
   question, include only that requested source. Use these columns: Source,
   Remaining supply, Sold supply, Total supply, and Realized demand.
3. State that Total supply equals Remaining supply plus Sold supply.
4. An ai-section titled Period and method with exact supplied dates and the
   deterministic calculation method.
5. An ai-note only when supplied limitations exist.
Do not include unrelated sources when an energy_source filter is present. Do not
show a value from another source. Do not copy these instructions into the answer.
""".strip()

    if intent == "marketplace_summary":
        return """
Create one ai-response section containing:
1. An ai-hero ai-info block titled &#128200; Marketplace summary with a concise
   statement of current active supply, active listing count, today's completed
   demand, and the leading source.
2. An ai-grid with cards for Active supply, Active listings, New supply today,
   and Completed demand today.
3. An ai-section titled Supply by renewable source containing one proper
   ai-table with exactly these columns: Source, Active listings, Available
   supply, Market share, Average asking price, New supply today, Completed
   demand today, and Average realized price. Include all supported sources.
4. An ai-section titled Today's activity containing a proper ai-table for new
   listings, newly listed kWh, completed purchases, completed demand kWh, and
   realized price information supplied in API_CONTEXT.
5. An ai-section titled Location highlights containing only supplied top-supply
   and top-demand locations.
6. An ai-note ai-neutral block with the supplied data date or as-of timestamp.
Do not create a per-source table where Total Active Supply and Highest Supply
are repeated in every row. Do not copy these instructions into the response.
""".strip()

    if intent in ANALYTICS_INTENTS:
        return """
Create one ai-response section containing:
1. An ai-hero ai-info block titled &#128200; Analysis with the main finding.
2. An ai-section titled Key metrics containing a proper ai-table whenever
   multiple sources, locations, or periods are compared.
3. An ai-section titled Period and method using exact supplied dates and method.
4. An ai-note ai-neutral block only when supplied limitations exist.
Do not copy these instructions or placeholders into the response.
""".strip()

    return """
Create one ai-response section containing:
1. An ai-hero ai-info block titled &#8505;&#65039; Marketplace insight with the
   direct live marketplace finding.
2. An ai-section titled Key metrics. Use a proper ai-table for multi-source or
   multi-option comparisons.
3. An ai-note ai-neutral block stating the supplied data scope or as-of time.
Do not copy these instructions or placeholders into the response.
""".strip()


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the final deterministic HTML-answer prompt for Nova Micro."""
    normalized_question = _normalize_question(question)
    rag_context = _build_context_blocks(retrieved_chunks)
    compact_api_context = _compact_api_context(api_context)
    api_text = _json_text(compact_api_context) if compact_api_context else "None"
    response_template = _response_template(
        normalized_question,
        compact_api_context,
    )

    return f"""
You are the grounded analytics, prediction, and recommendation assistant for
GreenGrid Exchange.

SUPPORTED ENERGY SOURCES:
{_supported_sources_lines()}

GROUNDING PRIORITY:
1. analytics_result, prediction_result, and recommendation_result contain
   deterministic calculations and are authoritative.
2. Live API facts, normalized aggregates, and record counts are authoritative
   for current and historical marketplace values.
3. RAG_CONTEXT is authoritative for definitions, marketplace rules,
   methodology, and limitations.
4. If a static document example conflicts with live API-derived data, use the
   live API-derived value.
5. Static RAG examples must never substitute for unavailable live marketplace
   data.

FACT DEFINITIONS:
- Current available supply means active available listing energy_kwh.
- Remaining supply means unsold credit kWh returned by get_all_listings for the scope.
- Sold supply means credit kWh returned by completed get_all_purchases.
- Total supply means remaining listing kWh plus sold completed-purchase kWh.
- Realized demand means sold energy_kwh from completed purchases.
- Historical listed supply alone is not total supply because sold credits no longer appear in get_all_listings.
- Average selling price means volume-weighted realized price unless the supplied
  calculation explicitly states another validated method.
- Demand-to-supply ratio means completed demand kWh divided by listed supply kWh
  for the same source, location, and period.
- Supported renewable source values are: {', '.join(SUPPORTED_ENERGY_SOURCES)}.
- Normalize old Small Hydro wording to Hydro. Never display SMALL_HYDRO as a
  separate source.

STRICT CONTENT RULES:
- Use only API_CONTEXT and RAG_CONTEXT.
- Do not invent, estimate, alter, round differently, or recalculate supplied
  numbers.
- Do not infer missing sources, locations, prices, quantities, dates, or status.
- For general source comparisons, consider every supported source. If the API
  result contains no record for a supported source, show zero or Not available
  rather than silently omitting that source.
- For explicitly scoped questions, compare only the requested sources,
  locations, or listings.
- Do not expose credentials, tokens, passwords, private keys, private wallet
  data, personal information, seller identifiers, or buyer identifiers.
- Do not claim that the assistant executed a purchase, listing, cancellation,
  blockchain transaction, or state-changing action.
- Do not call a forecast an observed fact.
- Do not describe a recommendation as guaranteed or as financial advice.
- If confidence is insufficient_data, clearly state that reliable prediction or
  recommendation is not possible and do not force a winner.
- If missing_parameters is non-empty, explain what is missing without inventing it.
- If an API failed or returned partial data, show that limitation prominently.
- If no matching records or aggregate data exist, state that no matching live
  marketplace data was found.
- State exact historical and forecast periods when supplied.
- State the calculation method when supplied.
- Use human-readable source labels: {_supported_sources_text()}.
- Do not put document names or API tool names in the answer unless explicitly
  requested; the client renders sources and API usage separately.

{_html_design_system()}

REQUIRED RESPONSE TEMPLATE:
{response_template}

USER QUESTION:
{normalized_question}

API_CONTEXT:
{api_text}

RAG_CONTEXT:
{rag_context}

Return only the safe HTML fragment. Do not return Markdown, JSON, code fences,
a full HTML document, a style block, or explanatory text outside the fragment.
""".strip()
