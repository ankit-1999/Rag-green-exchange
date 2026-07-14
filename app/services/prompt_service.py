"""
prompt_service.py
-----------------

Builds the final grounded answer prompt for GreenGrid Exchange.

The final LLM answer is returned as a polished HTML fragment so the frontend can
render it directly inside the chat message. The response must never contain a
complete HTML document, JavaScript, CSS, event handlers, forms, or unsafe URLs.

The prompt combines:
- Retrieved RAG chunks for rules and explanations
- Compact marketplace API metadata
- Deterministic analytics, prediction, and recommendation results
- The user's question

The LLM explains calculated results. The LLM must not recalculate, modify, or
invent marketplace values.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence


SUPPORTED_ENERGY_SOURCES = ("SOLAR", "WIND", "HYDRO")

PREDICTION_INTENTS = {
    "demand_prediction",
    "price_prediction",
    "shortage_prediction",
}

RECOMMENDATION_INTENTS = {
    "seller_recommendation",
    "buyer_recommendation",
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


# ---------------------------------------------------------------------------
# RAG source construction
# ---------------------------------------------------------------------------


def _build_context_blocks(retrieved_chunks: Sequence[Mapping[str, Any]]) -> str:
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
    Keep calculated results and execution metadata while removing complete raw
    API record arrays from the final LLM prompt.
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
            response_metadata: Dict[str, Any] = {}

            if isinstance(data, Mapping):
                possible_samples = data.get("sample_records", [])
                if isinstance(possible_samples, list):
                    sample_records = possible_samples[:5]

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
                    "response_metadata": response_metadata,
                }
            )

    return compact


# ---------------------------------------------------------------------------
# HTML formatting guidance
# ---------------------------------------------------------------------------


def _html_design_system() -> str:
    """Return the fixed visual vocabulary allowed in model-generated HTML."""
    return """
HTML DESIGN SYSTEM

Return one safe HTML fragment. Use only these tags:
section, div, p, h3, h4, ul, ol, li, strong, span, small, br, table, thead,
tbody, tr, th, td.

Allowed class names:
- ai-response
- ai-hero
- ai-title
- ai-subtitle
- ai-section
- ai-section-title
- ai-highlight
- ai-success
- ai-warning
- ai-danger
- ai-info
- ai-neutral
- ai-grid
- ai-card
- ai-card-title
- ai-card-value
- ai-card-label
- ai-list
- ai-metric-list
- ai-metric-row
- ai-metric-name
- ai-metric-value
- ai-badge
- ai-badge-high
- ai-badge-medium
- ai-badge-low
- ai-badge-insufficient
- ai-table
- ai-note
- ai-muted
- ai-source-solar
- ai-source-wind
- ai-source-hydro

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
- Solar: &#9728;&#65039;
- Wind: &#127788;&#65039;
- Hydro: &#128167;
- Location: &#128205;
- Confidence: &#127919;

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
Use this HTML structure:
<section class="ai-response">
  <div class="ai-hero ai-info">
    <h3 class="ai-title">Direct answer</h3>
    <p class="ai-subtitle">...</p>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">Explanation</h4>
    <p>...</p>
  </div>
  <div class="ai-note ai-neutral">Applicable rule or limitation.</div>
</section>
""".strip()
        return """
Use one compact HTML response:
<section class="ai-response">
  <div class="ai-hero ai-info">
    <h3 class="ai-title">Answer</h3>
    <p class="ai-subtitle">Direct answer.</p>
  </div>
</section>
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
        state_class = "ai-warning" if confidence == "insufficient_data" else "ai-info"
        return f"""
Use this structure:
<section class="ai-response">
  <div class="ai-hero {state_class}">
    <h3 class="ai-title">&#128302; Prediction</h3>
    <p class="ai-subtitle">Lead with the predicted result, or clearly state that the data is insufficient.</p>
  </div>
  <div class="ai-grid">
    <div class="ai-card">
      <div class="ai-card-label">Predicted leader</div>
      <div class="ai-card-value">...</div>
    </div>
    <div class="ai-card">
      <div class="ai-card-label">Expected value or range</div>
      <div class="ai-card-value">...</div>
    </div>
    <div class="ai-card">
      <div class="ai-card-label">Confidence</div>
      <div class="ai-card-value"><span class="ai-badge ...">...</span></div>
    </div>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">&#128200; Key figures</h4>
    <div class="ai-metric-list">Use compact metric rows or a small table.</div>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">Data period and method</h4>
    <p>State exact historical and forecast dates plus the calculation method.</p>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">Factors considered</h4>
    <ul class="ai-list"><li>Only supplied factors.</li></ul>
  </div>
  <div class="ai-note ai-warning"><strong>Limitations:</strong> Include supplied limitations and state that forecasts are not guaranteed.</div>
</section>

Confidence badge mapping:
- high: ai-badge ai-badge-high
- medium: ai-badge ai-badge-medium
- low: ai-badge ai-badge-low
- insufficient_data: ai-badge ai-badge-insufficient
""".strip()

    if is_recommendation:
        return """
Use this structure:
<section class="ai-response">
  <div class="ai-hero ai-success">
    <h3 class="ai-title">&#10024; Recommendation</h3>
    <p class="ai-subtitle">Lead with the recommendation. If no_strong_preference is true, say so clearly.</p>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">Why</h4>
    <ul class="ai-list"><li>Use only supplied reasons.</li></ul>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">&#128200; Supporting metrics</h4>
    <div class="ai-metric-list">Use metric rows or a compact comparison table.</div>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">&#127919; Confidence</h4>
    <p><span class="ai-badge ...">...</span></p>
  </div>
  <div class="ai-note ai-warning"><strong>Limitations:</strong> State that the result is decision support, not a guarantee or financial advice.</div>
</section>
""".strip()

    if intent in {
        "historical_supply",
        "historical_demand",
        "average_selling_price",
        "demand_supply_ratio",
        "market_balance",
        "supply_stability",
        "price_volatility",
    }:
        return """
Use this structure:
<section class="ai-response">
  <div class="ai-hero ai-info">
    <h3 class="ai-title">&#128200; Analysis</h3>
    <p class="ai-subtitle">Lead with the main finding.</p>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">Key metrics</h4>
    <div class="ai-metric-list">Use metric rows or a compact table.</div>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">Period and method</h4>
    <p>State exact dates and calculation method.</p>
  </div>
  <div class="ai-note ai-neutral"><strong>Limitations:</strong> Include only supplied limitations.</div>
</section>
""".strip()

    return """
Use this structure:
<section class="ai-response">
  <div class="ai-hero ai-info">
    <h3 class="ai-title">&#8505;&#65039; Marketplace insight</h3>
    <p class="ai-subtitle">Lead with the direct finding.</p>
  </div>
  <div class="ai-section">
    <h4 class="ai-section-title">Key metrics</h4>
    <div class="ai-metric-list">Use compact metric rows.</div>
  </div>
  <div class="ai-note ai-neutral">State the data scope.</div>
</section>
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
- SOLAR
- WIND
- HYDRO

GROUNDING PRIORITY:
1. analytics_result, prediction_result, and recommendation_result contain
   deterministic calculations and are authoritative.
2. Live API facts and record counts are authoritative for current marketplace
   values.
3. RAG_CONTEXT is authoritative for definitions, marketplace rules,
   methodology, and limitations.
4. If a static document example conflicts with live API-derived data, use the
   live API-derived value.

FACT DEFINITIONS:
- Current available supply means active available listing energy_kwh.
- Historical listed supply means listing energy_kwh created in the stated period.
- Realized demand means energy_kwh from completed purchases.
- Average selling price means volume-weighted realized price unless the supplied
  calculation explicitly says otherwise.
- Demand-to-supply ratio means completed demand kWh divided by listed supply kWh
  for the same scope and period.
- HYDRO is the supported source name.

STRICT CONTENT RULES:
- Use only API_CONTEXT and RAG_CONTEXT.
- Do not invent, estimate, alter, round differently, or recalculate supplied
  numbers.
- Do not infer missing sources, locations, prices, quantities, dates, or status.
- Do not expose credentials, tokens, passwords, private keys, private wallet
  data, or personal information.
- Do not claim that the assistant executed a purchase, listing, cancellation,
  blockchain transaction, or state-changing action.
- Do not call a forecast an observed fact.
- Do not describe a recommendation as guaranteed or as financial advice.
- If confidence is insufficient_data, clearly state that reliable prediction or
  recommendation is not possible and do not force a winner.
- If missing_parameters is non-empty, explain what is missing without inventing it.
- If an API failed or returned partial data, show that limitation prominently.
- If no matching records exist, state that no matching records were found.
- State exact historical and forecast periods when supplied.
- State the calculation method when supplied.
- Use human-readable source labels: Solar, Wind, and Hydro.
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
