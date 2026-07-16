"""Amazon Bedrock integration and deterministic marketplace API planner."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app.services import tool_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported marketplace values and intent definitions
# ---------------------------------------------------------------------------

SOURCES = set(
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

SOURCE_ALIASES = {
    "solar": "SOLAR",
    "solar energy": "SOLAR",
    "solar power": "SOLAR",
    "wind": "WIND",
    "wind energy": "WIND",
    "wind power": "WIND",
    "hydro": "HYDRO",
    "hydropower": "HYDRO",
    "hydro energy": "HYDRO",
    "hydro power": "HYDRO",
    "biomass": "BIOMASS",
    "bio mass": "BIOMASS",
    "biomass energy": "BIOMASS",
    "bioenergy": "BIOMASS",
    "geothermal": "GEOTHERMAL",
    "geothermal energy": "GEOTHERMAL",
    "geothermal power": "GEOTHERMAL",
    "tidal": "TIDAL",
    "tidal energy": "TIDAL",
    "tidal power": "TIDAL",
    "other": "OTHER",
    "other source": "OTHER",
    "other renewable": "OTHER",
    # Backward-compatible input normalization only.
    "small hydro": "HYDRO",
    "small-hydro": "HYDRO",
    "small_hydro": "HYDRO",
}

LISTING_STATUSES = {
    "ACTIVE",
    "SOLD",
    "EXPIRED",
    "CANCELLED",
}

PURCHASE_STATUSES = {
    "ACTIVE",
    "PENDING",
    "COMPLETED",
    "CONSUMED",
    "CANCELLED",
    "FAILED",
}

SORT_FIELDS = {
    "price_per_kwh",
    "energy_kwh",
    "created_at",
}

SORT_ORDERS = {"asc", "desc"}

MARKETPLACE_TOOLS = {
    "get_all_listings",
    "get_active_listings",
    "get_all_purchases",
}

INTENTS = {
    "none",
    "current_supply",
    "supply_mix",
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
    "demand_prediction",
    "price_prediction",
    "shortage_prediction",
    "seller_recommendation",
    "buyer_recommendation",
}

LIVE_DATA_INTENTS = INTENTS - {"none"}

PREDICTION_INTENTS = {
    "demand_prediction",
    "price_prediction",
    "shortage_prediction",
}

RECOMMENDATION_INTENTS = {
    "seller_recommendation",
    "buyer_recommendation",
}

GROUPS = {
    "energy_source",
    "location",
    "day",
    "week",
    "month",
}

METRICS = {
    "energy_kwh",
    "price_per_kwh",
    "total_price",
    "listing_count",
    "purchase_count",
    "demand_supply_ratio",
    "market_balance",
    "price_volatility",
    "supply_stability",
}


# Python enforces these combinations even if Nova proposes only one tool.
REQUIRED_TOOLS_BY_INTENT = {
    "current_supply": (
        "get_active_listings",
    ),
    "supply_mix": (
        "get_active_listings",
    ),
    "historical_supply": (
        "get_all_listings",
    ),
    "historical_demand": (
        "get_all_purchases",
    ),
    "demand_and_supply": (
        "get_all_listings",
        "get_all_purchases",
    ),
    "average_selling_price": (
        "get_all_purchases",
    ),
    "demand_supply_ratio": (
        "get_all_listings",
        "get_all_purchases",
    ),
    "market_balance": (
        "get_all_listings",
        "get_all_purchases",
    ),
    "supply_stability": (
        "get_all_listings",
    ),
    "price_volatility": (
        "get_all_purchases",
    ),
    "supply_by_location": (
        "get_active_listings",
    ),
    "marketplace_summary": (
        "get_active_listings",
        "get_all_listings",
        "get_all_purchases",
    ),
    "demand_prediction": (
        "get_all_listings",
        "get_all_purchases",
    ),
    "price_prediction": (
        "get_all_purchases",
        "get_active_listings",
    ),
    "shortage_prediction": (
        "get_all_listings",
        "get_all_purchases",
        "get_active_listings",
    ),
    "seller_recommendation": (
        "get_all_listings",
        "get_all_purchases",
        "get_active_listings",
    ),
    "buyer_recommendation": (
        "get_all_purchases",
        "get_active_listings",
    ),
}


# ---------------------------------------------------------------------------
# Bedrock client, embeddings, and answer generation
# ---------------------------------------------------------------------------


def _get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.AWS_REGION,
    )


def embed_text(text: str) -> List[float]:
    value = (text or "").strip()
    if not value:
        raise ValueError("Cannot embed empty text.")

    body = json.dumps(
        {
            "inputText": value,
            "dimensions": settings.BEDROCK_EMBEDDING_DIMENSION,
            "normalize": True,
        }
    )

    try:
        response = _get_bedrock_client().invoke_model(
            modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        embedding = json.loads(
            response["body"].read()
        ).get("embedding")

        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(
                "Bedrock returned no embedding vector."
            )

        return [float(item) for item in embedding]

    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(
            f"Bedrock embedding failed: {exc}"
        ) from exc


def generate_answer(prompt: str) -> str:
    value = (prompt or "").strip()
    if not value:
        raise ValueError("Prompt cannot be empty.")

    try:
        response = _get_bedrock_client().converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": value}],
                }
            ],
            inferenceConfig={
                "maxTokens": settings.BEDROCK_LLM_MAX_TOKENS,
                "temperature": settings.BEDROCK_LLM_TEMPERATURE,
            },
        )

        blocks = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )

        answer = "\n".join(
            block["text"].strip()
            for block in blocks
            if isinstance(block, Mapping)
            and isinstance(block.get("text"), str)
        ).strip()

        return _clean_html_answer(answer)

    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(
            f"Bedrock generation failed: {exc}"
        ) from exc


def _clean_html_answer(answer: str) -> str:
    """Remove accidental Markdown fences around the final HTML fragment."""
    value = (answer or "").strip()
    value = re.sub(
        r"^```(?:html)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\s*```$",
        "",
        value,
    )
    return value.strip()


# ---------------------------------------------------------------------------
# Date context
# ---------------------------------------------------------------------------


def _date_context(
    today: Optional[date] = None,
) -> Dict[str, str]:
    current = today or datetime.now(timezone.utc).date()
    week_start = current - timedelta(days=current.weekday())
    month_start = current.replace(day=1)
    last_month_end = month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)

    if month_start.month == 12:
        next_month_start = date(
            month_start.year + 1,
            1,
            1,
        )
    else:
        next_month_start = date(
            month_start.year,
            month_start.month + 1,
            1,
        )

    if next_month_start.month == 12:
        following = date(
            next_month_start.year + 1,
            1,
            1,
        )
    else:
        following = date(
            next_month_start.year,
            next_month_start.month + 1,
            1,
        )

    return {
        "today": current.isoformat(),
        "this_week_start": week_start.isoformat(),
        "this_week_end": (
            week_start + timedelta(days=6)
        ).isoformat(),
        "last_week_start": (
            week_start - timedelta(days=7)
        ).isoformat(),
        "last_week_end": (
            week_start - timedelta(days=1)
        ).isoformat(),
        "this_month_start": month_start.isoformat(),
        "this_month_to_date_end": current.isoformat(),
        "last_month_start": last_month_start.isoformat(),
        "last_month_end": last_month_end.isoformat(),
        "next_month_start": next_month_start.isoformat(),
        "next_month_end": (
            following - timedelta(days=1)
        ).isoformat(),
        "rolling_28_days_start": (
            current - timedelta(days=27)
        ).isoformat(),
        "rolling_180_days_start": (
            current
            - timedelta(
                days=(
                    settings.ANALYTICS_DEFAULT_HISTORY_DAYS
                    - 1
                )
            )
        ).isoformat(),
    }


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------


def _planner_prompt(
    question: str,
    catalog: str,
    dates: Mapping[str, str],
) -> str:
    """
    Build the marketplace-aware read-only API planner prompt.

    The planner determines intent and extracts filters. Python subsequently
    enforces the mandatory tool combination and API parameters for that intent.
    """
    supported_sources = "\n".join(
        f"- {source}"
        for source in settings.SUPPORTED_ENERGY_SOURCES
    )

    return f"""
You are the read-only marketplace API planner for GreenGrid Exchange.

Your only responsibilities are:
1. Identify the user's intent.
2. Decide whether live marketplace data is mandatory.
3. Select the required read-only GET tools.
4. Extract source, location, date, status, price, quantity, grouping, and sorting
   filters.
5. Return one valid JSON object.

You do not answer the user's question and do not calculate marketplace results.

SUPPORTED RENEWABLE SOURCES:
{supported_sources}

SOURCE NORMALIZATION:
- solar, solar energy, solar power -> SOLAR
- wind, wind energy, wind power -> WIND
- hydro, hydropower, hydro energy, hydro power -> HYDRO
- biomass, bio mass, biomass energy, bioenergy -> BIOMASS
- geothermal, geothermal energy, geothermal power -> GEOTHERMAL
- tidal, tidal energy, tidal power -> TIDAL
- other, other renewable, other source -> OTHER
- legacy Small Hydro wording -> HYDRO

Never output SMALL_HYDRO as a separate source.

AVAILABLE DATA SOURCES:

1. get_active_listings
Use for current available credits, current active supply, source market share,
location supply, active price or quantity filtering, buyer candidates, current
inventory, and shortage or seller-recommendation context.

2. get_all_listings
Use for historical listed supply, listing creation trends, listing statuses,
asking-price history, supply stability, demand-to-supply denominator,
market-balance analysis, prediction inputs, and seller-recommendation history.

3. get_all_purchases
Use for completed marketplace demand, realized selling prices, historical demand,
location demand, price volatility, demand prediction, price prediction, shortage
prediction, and recommendation inputs. Set group_by_month=true for monthly price
prediction data.

LIVE DATA ENFORCEMENT:
Every question about current or historical supply, availability, percentages,
source comparison, location comparison, demand, price, ratio, balance, stability,
volatility, prediction, shortage, buying, selling, or recommendations requires
live marketplace API data.

For those questions:
- requires_api_data must be true.
- requires_live_data must be true.
- tool_calls must not be empty.
- RAG documents and sample document values must not replace live marketplace data.
- If an API later fails or returns no usable records or aggregates, the final
  pipeline must report that live data was unavailable.

Conceptual questions may use RAG without marketplace APIs, for example:
- What is an electricity credit?
- How does GreenGrid Exchange work?
- What does demand-to-supply ratio mean?
- How is price volatility calculated?
- What are the supported renewable sources?
- What is the difference between active and sold listings?

MANDATORY INTENT-TO-TOOL RULES:

current_supply:
- Must use get_active_listings.
- Group by energy_source.

supply_mix:
- Must use get_active_listings.
- Do not apply one energy_source filter because all supported sources are compared.

historical_supply:
- Must use get_all_listings.

historical_demand:
- Must use get_all_purchases with status=COMPLETED.

demand_and_supply:
- Use when the question asks for demand and supply together for a source, location, or period.
- Must use get_all_listings and get_all_purchases with identical source, location, and period filters.
- Purchases must use status=COMPLETED.
- Remaining supply comes from get_all_listings.
- Sold supply and realized demand come from get_all_purchases.
- Total supply equals remaining listing kWh plus completed purchase kWh.
- Never use get_all_purchases alone for a demand-and-supply question.

average_selling_price:
- Must use get_all_purchases with status=COMPLETED.

supply_by_location:
- Must use get_active_listings.
- Apply energy_source when the user specifies one.
- Group by location.

marketplace_summary:
- Use for: Summarize today's marketplace, Summarize marketplace activity, and Show platform statistics.
- Must use get_active_listings, including for yesterday and last-week summaries.
- get_active_listings represents credits still available for sale now.
- For historical periods, analytics must retain only active listings whose created_at falls inside the requested period to measure remaining availability from that period.
- Must use get_all_listings with requested created_from and created_to for period listing supply.
- Must use get_all_purchases with status=COMPLETED and requested completed_from and completed_to for period realized demand.
- Do not apply an energy_source filter; summarize all supported sources.
- Group by energy_source and location.
- Include current unsold inventory from period, period listing supply, period completed demand, realized prices, market balance, and location highlights.

demand_supply_ratio:
- Must use get_all_listings and get_all_purchases.
- Both tools must use the same source, location, and historical period.
- Purchases must use status=COMPLETED.

market_balance:
- Must use get_all_listings and get_all_purchases.
- Both tools must use the same source, location, and historical period.
- Purchases must use status=COMPLETED.

supply_stability:
- Must use get_all_listings.
- Prefer 90 to 180 days of history.

price_volatility:
- Must use get_all_purchases with status=COMPLETED.
- Prefer 90 to 180 days of history.

demand_prediction:
- Must use get_all_listings and get_all_purchases.
- Purchases must use status=COMPLETED.
- Use at least the previous 180 days unless a longer valid period is supplied.
- For a general highest-demand question, do not apply a source filter; compare
  SOLAR, WIND, HYDRO, BIOMASS, GEOTHERMAL, TIDAL, and OTHER.

price_prediction:
- Must use get_all_purchases and get_active_listings.
- Purchases must use status=COMPLETED and group_by_month=true.
- Use at least the previous 180 days of purchase history.
- For a general source prediction, do not apply one source filter.

shortage_prediction:
- Must use get_all_listings, get_all_purchases, and get_active_listings.
- Apply the same source and location to all three tools when supplied.
- Purchases must use status=COMPLETED.
- Use at least the previous 180 days of history.

seller_recommendation:
- Must use get_all_listings, get_all_purchases, and get_active_listings.
- Purchases must use status=COMPLETED.
- Use the previous 28 days for a this-week listing recommendation.
- If the user names multiple sources, compare only those sources in analytics but
  do not incorrectly apply one source filter to all API calls.
- If no sources are named, compare every supported source.

buyer_recommendation:
- Must use get_active_listings and get_all_purchases.
- Active listings are candidates; completed purchases provide historical demand.
- Consider every supported source unless the user explicitly scopes the request.

DATE RULES:
- today -> DATE_CONTEXT.today
- this week -> DATE_CONTEXT.this_week_start through DATE_CONTEXT.this_week_end
- last week -> DATE_CONTEXT.last_week_start through DATE_CONTEXT.last_week_end
- this month -> DATE_CONTEXT.this_month_start through DATE_CONTEXT.this_month_to_date_end
- last month -> DATE_CONTEXT.last_month_start through DATE_CONTEXT.last_month_end
- next month -> DATE_CONTEXT.next_month_start through DATE_CONTEXT.next_month_end
- Predictions without an explicit history use rolling_180_days_start through today.
- Seller recommendations for this week use rolling_28_days_start through today.
- Next-month dates are the forecast period, never historical API filters.
- If the user says selected period without supplying one, add date_range to
  missing_parameters and do not invent a date range.

FILTER RULES:
- Use created_from and created_to only with get_all_listings.
- Use completed_from and completed_to only with get_all_purchases.
- Use status=COMPLETED for realized demand and realized price.
- Use group_by_month=true only when monthly purchase price trends are required.
- Use skip=0 and limit=200.
- Add location only when the user specifies a location.
- Add energy_source only when one specific source is requested.
- Do not apply one source filter when comparing multiple or all supported sources.
- Do not generate separate API calls per source when one unfiltered call can
  return all supported source records.
- Use only filters defined in the selected tool payload_schema.
- Never invent unsupported filters.

ROUTING EXAMPLES:

Question: Summarize today's marketplace.
Result: marketplace_summary, get_active_listings plus period-filtered all listings and completed purchases, no source filter, group by source and location. Analytics filters active listings by created_at to report credits from the requested period that remain available now.

Question: Summarize marketplace activity.
Result: marketplace_summary with current unsold inventory from period and period listing and purchase activity.

Question: Show platform statistics.
Result: marketplace_summary with current unsold inventory from period, listing activity in period, completed demand in period, prices, and location highlights.

Question: Which renewable source currently has the highest available supply?
Result: current_supply, get_active_listings, group_by energy_source, no source filter.

Question: What percentage of active marketplace supply comes from each source?
Result: supply_mix, get_active_listings, compare all supported sources.

Question: Compare demand for Biomass, Geothermal, and Tidal last month.
Result: historical_demand, get_all_purchases, status COMPLETED, last-month dates.
Do not apply one energy_source filter because multiple sources are compared.

Question: What was the demand and supply for Solar credits during this period?
Result: demand_and_supply, get_all_listings plus get_all_purchases, identical SOLAR and date filters, purchases status COMPLETED. Total supply is remaining listing kWh plus sold purchase kWh.

Question: Which source had the highest average selling price last month?
Result: average_selling_price, get_all_purchases, status COMPLETED, no source filter.

Question: Predict which renewable source will have highest demand next month.
Result: demand_prediction, all listings plus purchases, 180-day history, next-month
forecast, compare all seven supported sources, group by source and week.

Question: Is Noida likely to face a Biomass-credit shortage next month?
Result: shortage_prediction, all three tools, energy_source BIOMASS and location
Noida on every tool, 180-day history, next-month forecast.

Question: Which location has the greatest Tidal-credit supply?
Result: supply_by_location, get_active_listings, energy_source TIDAL, group by location.

Question: Should I list Solar or Wind credits this week?
Result: seller_recommendation, all three tools, previous 28 days, compare the two
requested sources without applying one source as the sole API filter.

SAFETY RULES:
- Only select tools present in TOOL_CATALOG.
- Never select POST, PATCH, PUT, or DELETE operations.
- Never select document-ingestion, authenticated-user, or blockchain tools.
- Never answer the user's question in the planner.
- Never calculate predictions in the planner.
- Never invent marketplace facts.
- Return only valid JSON without Markdown or code fences.

ALLOWED INTENTS:
- none
- current_supply
- supply_mix
- historical_supply
- historical_demand
- demand_and_supply
- average_selling_price
- demand_supply_ratio
- market_balance
- supply_stability
- price_volatility
- supply_by_location
- marketplace_summary
- demand_prediction
- price_prediction
- shortage_prediction
- seller_recommendation
- buyer_recommendation

DATE_CONTEXT:
{json.dumps(dict(dates), ensure_ascii=True, indent=2)}

TOOL_CATALOG:
{catalog}

OUTPUT JSON SCHEMA:
{{
  "requires_api_data": true,
  "requires_live_data": true,
  "reason": "Short explanation of why live API data is required.",
  "intent": "one allowed intent",
  "is_prediction": false,
  "is_recommendation": false,
  "historical_period": {{
    "from": "YYYY-MM-DD or null",
    "to": "YYYY-MM-DD or null"
  }},
  "forecast_period": {{
    "from": "YYYY-MM-DD or null",
    "to": "YYYY-MM-DD or null"
  }},
  "group_by": ["energy_source"],
  "metrics": ["energy_kwh"],
  "missing_parameters": [],
  "tool_calls": [
    {{
      "tool": "tool_name_from_catalog",
      "arguments": {{}}
    }}
  ]
}}

For conceptual questions that do not require live data:
{{
  "requires_api_data": false,
  "requires_live_data": false,
  "reason": "The question can be answered using project documentation.",
  "intent": "none",
  "is_prediction": false,
  "is_recommendation": false,
  "historical_period": {{"from": null, "to": null}},
  "forecast_period": {{"from": null, "to": null}},
  "group_by": [],
  "metrics": [],
  "missing_parameters": [],
  "tool_calls": []
}}

USER QUESTION:
{question}
""".strip()


# ---------------------------------------------------------------------------
# Public planner entry point
# ---------------------------------------------------------------------------


def plan_api_calls(question: str) -> Dict[str, Any]:
    value = (question or "").strip()
    if not value:
        return _empty_plan("empty_question")

    dates = _date_context()

    # Use the compact catalog for the planner to reduce repeated input tokens.
    # The full catalog remains available for validation and documentation.
    if hasattr(tool_registry, "build_compact_planner_tools_text"):
        catalog = tool_registry.build_compact_planner_tools_text()
    else:
        catalog = tool_registry.build_planner_tools_text()

    prompt = _planner_prompt(
        value,
        catalog,
        dates,
    )

    try:
        response = _get_bedrock_client().converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": settings.BEDROCK_PLANNER_MAX_TOKENS,
                "temperature": settings.BEDROCK_PLANNER_TEMPERATURE,
            },
        )

        blocks = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )

        planner_text = "\n".join(
            block["text"].strip()
            for block in blocks
            if isinstance(block, Mapping)
            and isinstance(block.get("text"), str)
        )

        plan = _normalize_plan(
            _parse_json(planner_text)
        )
        plan = _enforce_required_plan_components(
            plan,
            dates,
        )

        logger.info(
            "API plan intent=%s calls=%s",
            plan["intent"],
            plan["tool_calls"],
        )
        return plan

    except (ClientError, BotoCoreError) as exc:
        logger.exception("Planner invocation failed")
        return _empty_plan(
            f"planner_bedrock_error:{type(exc).__name__}"
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        logger.warning("Invalid planner output: %s", exc)
        return _empty_plan("planner_invalid_output")


# ---------------------------------------------------------------------------
# Planner response parsing and normalization
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse a planner JSON object, tolerating accidental Markdown fences."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty planner output")

    cleaned = text.strip()

    match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        cleaned = match.group(1)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(cleaned[start : end + 1])

    if not isinstance(result, dict):
        raise ValueError("Planner output is not an object")

    return result


def _normalize_plan(
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    intent = str(
        raw.get("intent", "none") or "none"
    ).strip().lower()

    if intent not in INTENTS:
        intent = "none"

    calls = _normalize_calls(
        raw.get("tool_calls")
    )

    requires_api_data = (
        bool(calls)
        or bool(raw.get("requires_api_data", False))
    )

    requires_live_data = bool(
        raw.get("requires_live_data", False)
    )

    if intent in LIVE_DATA_INTENTS:
        requires_api_data = True
        requires_live_data = True

    return {
        "requires_api_data": requires_api_data,
        "requires_live_data": requires_live_data,
        "reason": str(
            raw.get("reason", "") or ""
        ).strip(),
        "intent": intent,
        "is_prediction": (
            bool(raw.get("is_prediction", False))
            or intent in PREDICTION_INTENTS
        ),
        "is_recommendation": (
            bool(raw.get("is_recommendation", False))
            or intent in RECOMMENDATION_INTENTS
        ),
        "historical_period": _period(
            raw.get("historical_period")
        ),
        "forecast_period": _period(
            raw.get("forecast_period")
        ),
        "group_by": _string_list(
            raw.get("group_by"),
            GROUPS,
        ),
        "metrics": _string_list(
            raw.get("metrics"),
            METRICS,
        ),
        "missing_parameters": _missing(
            raw.get("missing_parameters")
        ),
        "tool_calls": calls,
    }


# ---------------------------------------------------------------------------
# Deterministic plan enforcement
# ---------------------------------------------------------------------------


def _enforce_required_plan_components(
    plan: Mapping[str, Any],
    dates: Mapping[str, str],
) -> Dict[str, Any]:
    enforced = dict(plan)
    intent = str(enforced.get("intent", "none"))
    calls = [
        dict(call)
        for call in enforced.get("tool_calls", [])
        if isinstance(call, Mapping)
    ]

    if intent not in REQUIRED_TOOLS_BY_INTENT:
        enforced["requires_api_data"] = (
            bool(calls)
            or bool(enforced.get("requires_api_data"))
        )
        enforced["requires_live_data"] = (
            bool(enforced.get("requires_live_data"))
            or intent in LIVE_DATA_INTENTS
        )
        enforced["tool_calls"] = calls
        return enforced

    history_from, history_to = _history_range(
        intent,
        enforced.get("historical_period"),
        dates,
    )

    source = _first_arg(calls, "energy_source")
    location = _first_arg(calls, "location")
    indexed = {
        str(call.get("tool")): call
        for call in calls
    }
    output: List[Dict[str, Any]] = []

    # General comparison intents must not inherit one accidental source filter.
    general_comparison_intents = {
        "current_supply",
        "supply_mix",
        "marketplace_summary",
        "demand_prediction",
        "price_prediction",
        "seller_recommendation",
        "buyer_recommendation",
    }

    for tool in REQUIRED_TOOLS_BY_INTENT[intent]:
        call = indexed.get(
            tool,
            {
                "tool": tool,
                "arguments": {},
            },
        )
        args = dict(call.get("arguments", {}) or {})

        if source and intent not in general_comparison_intents:
            args.setdefault(
                "energy_source",
                _source(source),
            )
        elif intent in general_comparison_intents:
            # Remove a single planner-selected source from general all-source
            # comparisons. Explicit multi-source scoping is handled after API
            # retrieval by analytics_service.
            args.pop("energy_source", None)

        if location:
            args.setdefault("location", location)

        if tool == "get_all_listings":
            args["created_from"] = history_from
            args["created_to"] = history_to

        elif tool == "get_all_purchases":
            args["status"] = "COMPLETED"
            args["completed_from"] = history_from
            args["completed_to"] = history_to
            args["group_by_month"] = (
                intent == "price_prediction"
            )

        args["skip"] = 0
        args["limit"] = min(
            settings.MARKETPLACE_API_PAGE_SIZE,
            200,
        )

        output.append(
            {
                "tool": tool,
                "arguments": _normalize_args(
                    tool,
                    args,
                ),
            }
        )

    enforced["tool_calls"] = output
    enforced["requires_api_data"] = True
    enforced["requires_live_data"] = True
    enforced["historical_period"] = {
        "from": history_from,
        "to": history_to,
    }

    if intent in PREDICTION_INTENTS:
        enforced["forecast_period"] = {
            "from": dates["next_month_start"],
            "to": dates["next_month_end"],
        }

    groups = {
        "current_supply": ["energy_source"],
        "supply_mix": ["energy_source"],
        "historical_supply": ["energy_source"],
        "historical_demand": ["energy_source"],
        "demand_and_supply": ["energy_source"],
        "average_selling_price": ["energy_source"],
        "demand_supply_ratio": ["energy_source"],
        "market_balance": ["energy_source"],
        "supply_stability": ["energy_source", "week"],
        "price_volatility": ["energy_source", "week"],
        "supply_by_location": ["location"],
        "marketplace_summary": ["energy_source", "location"],
        "demand_prediction": ["energy_source", "week"],
        "price_prediction": ["energy_source", "month"],
        "shortage_prediction": ["week"],
        "seller_recommendation": ["energy_source", "week"],
        "buyer_recommendation": ["energy_source"],
    }

    metrics = {
        "current_supply": ["energy_kwh"],
        "supply_mix": ["energy_kwh"],
        "historical_supply": ["energy_kwh"],
        "historical_demand": ["energy_kwh", "purchase_count"],
        "demand_and_supply": ["energy_kwh", "listing_count", "purchase_count"],
        "average_selling_price": ["price_per_kwh", "energy_kwh"],
        "demand_supply_ratio": ["energy_kwh", "demand_supply_ratio"],
        "market_balance": ["energy_kwh", "market_balance"],
        "supply_stability": ["energy_kwh", "supply_stability"],
        "price_volatility": ["price_per_kwh", "price_volatility"],
        "supply_by_location": ["energy_kwh", "listing_count"],
        "marketplace_summary": ["energy_kwh", "price_per_kwh", "listing_count", "purchase_count", "market_balance"],
        "demand_prediction": ["energy_kwh", "demand_supply_ratio"],
        "price_prediction": ["price_per_kwh", "energy_kwh"],
        "shortage_prediction": ["energy_kwh", "market_balance"],
        "seller_recommendation": [
            "energy_kwh",
            "price_per_kwh",
            "demand_supply_ratio",
        ],
        "buyer_recommendation": [
            "energy_kwh",
            "price_per_kwh",
        ],
    }

    enforced["group_by"] = _merge(
        enforced.get("group_by"),
        groups[intent],
    )
    enforced["metrics"] = _merge(
        enforced.get("metrics"),
        metrics[intent],
    )

    return enforced


def _history_range(
    intent: str,
    period: Any,
    dates: Mapping[str, str],
) -> tuple[str, str]:
    supplied = _period(period)

    if intent in PREDICTION_INTENTS:
        if supplied["from"] and supplied["to"]:
            try:
                start = date.fromisoformat(
                    supplied["from"][:10]
                )
                end = date.fromisoformat(
                    supplied["to"][:10]
                )
                if (
                    (end - start).days
                    >= settings.ANALYTICS_DEFAULT_HISTORY_DAYS - 1
                ):
                    return supplied["from"], supplied["to"]
            except ValueError:
                pass

        return (
            dates["rolling_180_days_start"],
            dates["today"],
        )

    if intent in {
        "supply_stability",
        "price_volatility",
    }:
        return (
            dates["rolling_180_days_start"],
            dates["today"],
        )

    if intent == "seller_recommendation":
        return (
            dates["rolling_28_days_start"],
            dates["today"],
        )

    if supplied["from"] and supplied["to"]:
        return supplied["from"], supplied["to"]

    return (
        dates["rolling_28_days_start"],
        dates["today"],
    )


# ---------------------------------------------------------------------------
# Tool-call and argument normalization
# ---------------------------------------------------------------------------


def _normalize_calls(
    value: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    allowed = (
        set(tool_registry.get_allowed_tool_names())
        & MARKETPLACE_TOOLS
    )
    output: List[Dict[str, Any]] = []
    seen = set()

    for call in value:
        if not isinstance(call, Mapping):
            continue

        tool = str(call.get("tool", ""))
        if tool not in allowed:
            continue

        normalized = {
            "tool": tool,
            "arguments": _normalize_args(
                tool,
                call.get("arguments"),
            ),
        }
        key = json.dumps(normalized, sort_keys=True)

        if key not in seen:
            seen.add(key)
            output.append(normalized)

    return output[:4]


def _normalize_args(
    tool: str,
    value: Any,
) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    definition = tool_registry.get_tool_by_name(tool) or {}
    schema = definition.get("payload_schema", {})
    allowed = (
        set(schema.keys())
        if isinstance(schema, Mapping)
        else set()
    )
    out: Dict[str, Any] = {}

    for key, item in raw.items():
        if key not in allowed or item in (None, ""):
            continue

        if key == "energy_source":
            source = _source(item)
            if source:
                out[key] = source

        elif key == "location" and isinstance(item, str):
            location = item.strip()
            if location:
                out[key] = location[:200]

        elif key == "status":
            status = str(item).strip().upper()
            valid = (
                PURCHASE_STATUSES
                if tool == "get_all_purchases"
                else LISTING_STATUSES
            )
            if status in valid:
                out[key] = status

        elif key in {
            "created_from",
            "created_to",
            "completed_from",
            "completed_to",
        }:
            parsed = _iso(item)
            if parsed:
                out[key] = parsed

        elif key == "group_by_month":
            parsed_boolean = _boolean(item)
            if parsed_boolean is not None:
                out[key] = parsed_boolean

        elif key in {
            "min_price_per_kwh",
            "max_price_per_kwh",
        }:
            try:
                number = float(item)
                if number > 0:
                    out[key] = number
            except (TypeError, ValueError):
                pass

        elif key == "min_energy_kwh":
            try:
                number = float(item)
                if number > 0:
                    out[key] = (
                        int(number)
                        if number.is_integer()
                        else number
                    )
            except (TypeError, ValueError):
                pass

        elif key == "limit":
            try:
                number = int(item)
                if number > 0:
                    out[key] = min(
                        number,
                        settings.MARKETPLACE_API_PAGE_SIZE,
                        200,
                    )
            except (TypeError, ValueError):
                pass

        elif key == "skip":
            try:
                number = int(item)
                if number >= 0:
                    out[key] = number
            except (TypeError, ValueError):
                pass

        elif (
            key == "sort_by"
            and str(item).lower() in SORT_FIELDS
        ):
            out[key] = str(item).lower()

        elif (
            key == "sort_order"
            and str(item).lower() in SORT_ORDERS
        ):
            out[key] = str(item).lower()

    out.setdefault("skip", 0)
    out.setdefault(
        "limit",
        min(settings.MARKETPLACE_API_PAGE_SIZE, 200),
    )

    if tool == "get_all_purchases":
        out.setdefault("status", "COMPLETED")
        out.setdefault("group_by_month", False)

    if tool == "get_active_listings":
        out.setdefault("sort_by", "created_at")
        out.setdefault("sort_order", "desc")

    return out


def _source(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    enum_suffix = raw.split(".")[-1]
    direct = (
        enum_suffix.upper()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if direct in SOURCES:
        return direct

    return SOURCE_ALIASES.get(raw.lower())


def _iso(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None

    candidate = value.strip()

    try:
        date.fromisoformat(candidate)
        return candidate
    except ValueError:
        pass

    try:
        datetime.fromisoformat(
            candidate.replace("Z", "+00:00")
        )
        return candidate
    except ValueError:
        return None


def _boolean(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in {0, 1}:
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    return None


def _period(
    value: Any,
) -> Dict[str, Optional[str]]:
    if not isinstance(value, Mapping):
        return {
            "from": None,
            "to": None,
        }

    return {
        "from": _iso(value.get("from")),
        "to": _iso(value.get("to")),
    }


def _first_arg(
    calls: Sequence[Mapping[str, Any]],
    key: str,
) -> Any:
    for call in calls:
        args = call.get("arguments", {})
        if (
            isinstance(args, Mapping)
            and args.get(key) not in (None, "")
        ):
            return args[key]

    return None


def _string_list(
    value: Any,
    allowed: Set[str],
) -> List[str]:
    if not isinstance(value, list):
        return []

    return list(
        dict.fromkeys(
            str(item).strip().lower()
            for item in value
            if str(item).strip().lower() in allowed
        )
    )


def _missing(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    return list(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    )


def _merge(
    existing: Any,
    required: Sequence[str],
) -> List[str]:
    values = (
        list(existing)
        if isinstance(existing, list)
        else []
    )

    for item in required:
        if item not in values:
            values.append(item)

    return values


def _empty_plan(reason: str) -> Dict[str, Any]:
    return {
        "requires_api_data": False,
        "requires_live_data": False,
        "reason": reason,
        "intent": "none",
        "is_prediction": False,
        "is_recommendation": False,
        "historical_period": {
            "from": None,
            "to": None,
        },
        "forecast_period": {
            "from": None,
            "to": None,
        },
        "group_by": [],
        "metrics": [],
        "missing_parameters": [],
        "tool_calls": [],
    }
