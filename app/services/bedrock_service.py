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

SOURCES = {"SOLAR", "WIND", "HYDRO"}
SOURCE_ALIASES = {
    "solar": "SOLAR", "solar energy": "SOLAR", "solar power": "SOLAR",
    "wind": "WIND", "wind energy": "WIND", "wind power": "WIND",
    "hydro": "HYDRO", "hydropower": "HYDRO", "hydro energy": "HYDRO",
    "small hydro": "HYDRO", "small-hydro": "HYDRO", "small_hydro": "HYDRO",
}
LISTING_STATUSES = {"active", "sold", "expired", "cancelled"}
PURCHASE_STATUSES = {"active", "pending", "completed", "consumed", "cancelled", "failed"}
SORT_FIELDS = {"price_per_kwh", "energy_kwh", "created_at", "expires_at"}
SORT_ORDERS = {"asc", "desc"}
MARKETPLACE_TOOLS = {"get_all_listings", "get_active_listings", "get_all_purchases"}
INTENTS = {
    "none", "current_supply", "supply_mix", "historical_supply",
    "historical_demand", "average_selling_price", "demand_supply_ratio",
    "market_balance", "supply_stability", "price_volatility",
    "demand_prediction", "price_prediction", "shortage_prediction",
    "seller_recommendation", "buyer_recommendation",
}
PREDICTION_INTENTS = {"demand_prediction", "price_prediction", "shortage_prediction"}
RECOMMENDATION_INTENTS = {"seller_recommendation", "buyer_recommendation"}
GROUPS = {"energy_source", "location", "day", "week", "month"}
METRICS = {
    "energy_kwh", "price_per_kwh", "total_price", "listing_count",
    "purchase_count", "demand_supply_ratio", "market_balance",
    "price_volatility", "supply_stability",
}

# Python enforces these combinations even if Nova proposes only one tool.
REQUIRED_TOOLS_BY_INTENT = {
    "demand_supply_ratio": ("get_all_listings", "get_all_purchases"),
    "market_balance": ("get_all_listings", "get_all_purchases"),
    "demand_prediction": ("get_all_listings", "get_all_purchases"),
    "price_prediction": ("get_all_purchases", "get_active_listings"),
    "shortage_prediction": (
        "get_all_listings", "get_all_purchases", "get_active_listings"
    ),
    "seller_recommendation": (
        "get_all_listings", "get_all_purchases", "get_active_listings"
    ),
    "buyer_recommendation": ("get_all_purchases", "get_active_listings"),
}


def _get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)


def embed_text(text: str) -> List[float]:
    value = (text or "").strip()
    if not value:
        raise ValueError("Cannot embed empty text.")
    body = json.dumps({
        "inputText": value,
        "dimensions": settings.BEDROCK_EMBEDDING_DIMENSION,
        "normalize": True,
    })
    try:
        response = _get_bedrock_client().invoke_model(
            modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json", accept="application/json", body=body,
        )
        embedding = json.loads(response["body"].read()).get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Bedrock returned no embedding vector.")
        return [float(item) for item in embedding]
    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(f"Bedrock embedding failed: {exc}") from exc


def generate_answer(prompt: str) -> str:
    value = (prompt or "").strip()
    if not value:
        raise ValueError("Prompt cannot be empty.")
    try:
        response = _get_bedrock_client().converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": value}]}],
            inferenceConfig={
                "maxTokens": settings.BEDROCK_LLM_MAX_TOKENS,
                "temperature": settings.BEDROCK_LLM_TEMPERATURE,
            },
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "\n".join(
            block["text"].strip() for block in blocks
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ).strip()
    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(f"Bedrock generation failed: {exc}") from exc


def _date_context(today: Optional[date] = None) -> Dict[str, str]:
    current = today or datetime.now(timezone.utc).date()
    week_start = current - timedelta(days=current.weekday())
    month_start = current.replace(day=1)
    last_month_end = month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    if month_start.month == 12:
        next_month_start = date(month_start.year + 1, 1, 1)
    else:
        next_month_start = date(month_start.year, month_start.month + 1, 1)
    if next_month_start.month == 12:
        following = date(next_month_start.year + 1, 1, 1)
    else:
        following = date(next_month_start.year, next_month_start.month + 1, 1)
    return {
        "today": current.isoformat(),
        "this_week_start": week_start.isoformat(),
        "this_week_end": (week_start + timedelta(days=6)).isoformat(),
        "last_week_start": (week_start - timedelta(days=7)).isoformat(),
        "last_week_end": (week_start - timedelta(days=1)).isoformat(),
        "this_month_start": month_start.isoformat(),
        "this_month_to_date_end": current.isoformat(),
        "last_month_start": last_month_start.isoformat(),
        "last_month_end": last_month_end.isoformat(),
        "next_month_start": next_month_start.isoformat(),
        "next_month_end": (following - timedelta(days=1)).isoformat(),
        "rolling_28_days_start": (current - timedelta(days=27)).isoformat(),
        "rolling_180_days_start": (
            current - timedelta(days=settings.ANALYTICS_DEFAULT_HISTORY_DAYS - 1)
        ).isoformat(),
    }


def _planner_prompt(
    question: str,
    catalog: str,
    dates: Mapping[str, str],
) -> str:
    """
    Build the marketplace-aware read-only API planner prompt.

    Important:
    The planner determines the analytical intent and extracts filters.
    Python subsequently enforces the mandatory tool combination for the intent.
    """

    return f"""
You are the read-only marketplace API planner for GreenGrid Exchange.

Your only responsibility is to:

1. Identify the user's intent.
2. Decide whether live marketplace data is mandatory.
3. Select the required read-only GET tools.
4. Extract source, location, date, status, price, quantity, and sorting filters.
5. Return one valid JSON object.

You do not answer the user's question.

SUPPORTED RENEWABLE SOURCES:

- SOLAR
- WIND
- HYDRO

Normalize all source language as follows:

- solar, solar energy, solar power -> SOLAR
- wind, wind energy, wind power -> WIND
- hydro, hydropower, hydro energy -> HYDRO

Never output SMALL_HYDRO.

AVAILABLE DATA SOURCES:

1. get_active_listings

Use for:

- Current available credits
- Current active supply
- Active marketplace supply percentage
- Active listings by location
- Lowest or highest-priced active listings
- Current marketplace inventory
- Current buyer recommendation candidates
- Active inventory used in shortage prediction

2. get_all_listings

Use for:

- Historical listed supply
- Listings created during a historical period
- Historical supply trends
- Listing-status analysis
- Asking-price history
- Supply stability
- Demand-to-supply ratio
- Market-balance analysis
- Prediction inputs

3. get_all_purchases

Use for:

- Realized marketplace demand
- Completed purchase volume
- Historical demand trends
- Realized selling prices
- Average selling price
- Price volatility
- Demand prediction
- Price prediction
- Shortage prediction
- Recommendation inputs

LIVE DATA ENFORCEMENT RULES:

The following question types always require live marketplace API data:

- Current supply
- Current availability
- Marketplace percentages
- Source comparison
- Location comparison
- Historical demand
- Historical supply
- Average marketplace price
- Average selling price
- Price comparison
- Demand-to-supply ratio
- Shortage or surplus analysis
- Supply stability
- Price volatility
- Demand prediction
- Supply prediction
- Price prediction
- Shortage prediction
- Buyer recommendation
- Seller recommendation
- Questions asking what the user should list or buy
- Questions asking which source, location, credit, or listing is best

For those questions:

- requires_api_data must be true.
- requires_live_data must be true.
- Never return an empty tool_calls list.
- Never allow RAG documents to substitute for marketplace API data.
- Static examples in documents are not live marketplace facts.
- Sample credits in documents must not be used for calculations.
- If live APIs later fail or return no usable records, the final answer must
  state that the result could not be calculated from live marketplace data.

Conceptual questions may use RAG without marketplace APIs.

Examples of conceptual questions:

- What is an electricity credit?
- How does GreenGrid Exchange work?
- What does demand-to-supply ratio mean?
- How is price volatility calculated?
- What are the supported renewable sources?
- What is the difference between active and sold listings?

MANDATORY INTENT-TO-TOOL RULES:

current_supply:

- Must use get_active_listings.

supply_mix:

- Must use get_active_listings.
- Do not apply an energy_source filter because all sources must be compared.

historical_supply:

- Must use get_all_listings.

historical_demand:

- Must use get_all_purchases.
- Use status=completed.

average_selling_price:

- Must use get_all_purchases.
- Use status=completed.

demand_supply_ratio:

- Must use get_all_listings.
- Must use get_all_purchases.
- Both tools must use the same period and location/source scope.
- Purchases must use status=completed.

market_balance:

- Must use get_all_listings.
- Must use get_all_purchases.
- Both tools must use the same period and location/source scope.
- Purchases must use status=completed.

supply_stability:

- Must use get_all_listings.
- Prefer at least 90 to 180 days of historical data.

price_volatility:

- Must use get_all_purchases.
- Use status=completed.
- Prefer at least 90 to 180 days of historical data.

supply_by_location:

- Must use get_active_listings.
- Apply energy_source only when specified.
- Group results by location.

demand_prediction:

- Must use get_all_listings.
- Must use get_all_purchases.
- Purchases must use status=completed.
- Use at least the previous 180 days as history unless the user explicitly
  supplies a longer valid period.
- Do not apply an energy_source filter when predicting the highest-demand source
  across Solar, Wind, and Hydro.

price_prediction:

- Must use get_all_purchases.
- Must use get_active_listings.
- Purchases must use status=completed.
- Use at least the previous 180 days as historical input.

shortage_prediction:

- Must use get_all_listings.
- Must use get_all_purchases.
- Must use get_active_listings.
- Apply the same source and location to all three tools.
- Purchases must use status=completed.
- Use at least the previous 180 days as historical input.

seller_recommendation:

- Must use get_all_listings.
- Must use get_all_purchases.
- Must use get_active_listings.
- Purchases must use status=completed.
- Use the previous 28 days as supporting history for a question about
  listing credits this week.
- Do not apply a single source filter when comparing Solar and Wind.

buyer_recommendation:

- Must use get_active_listings.
- Must use get_all_purchases.
- Use active listings as the recommendation candidates.
- Use completed purchases for historical demand.

DATE RULES:

Use the supplied DATE_CONTEXT.

- "today" -> DATE_CONTEXT.today
- "this week" -> DATE_CONTEXT.this_week_start through
  DATE_CONTEXT.this_week_end
- "last week" -> DATE_CONTEXT.last_week_start through
  DATE_CONTEXT.last_week_end
- "this month" -> DATE_CONTEXT.this_month_start through
  DATE_CONTEXT.this_month_to_date_end
- "last month" -> DATE_CONTEXT.last_month_start through
  DATE_CONTEXT.last_month_end
- "next month" -> DATE_CONTEXT.next_month_start through
  DATE_CONTEXT.next_month_end
- Predictions without an explicit historical period must use
  DATE_CONTEXT.rolling_180_days_start through DATE_CONTEXT.today.
- Seller recommendations for "this week" must use
  DATE_CONTEXT.rolling_28_days_start through DATE_CONTEXT.today.

The next-month date range is the forecast period.

Do not use the next-month dates as historical API filters.

If the user says "during the selected period" but does not supply a date range
or relative period:

- Add "date_range" to missing_parameters.
- Do not invent a date range.

FILTER RULES:

- Use created_from and created_to only with listing tools.
- Use completed_from and completed_to only with purchase tools.
- Use status=completed for realized demand and realized-price analysis.
- Use skip=0.
- Use limit=100.
- Add location only if the user specifies a location.
- Add energy_source only if the user asks about one specific source.
- Do not apply one source filter when comparing Solar, Wind, and Hydro.
- Do not make separate API calls per source when one unfiltered call can return
  all source records.
- Use only filters defined in the selected tool's payload_schema.
- Never invent unsupported filters.

QUESTION ROUTING EXAMPLES:

Question:
Which renewable source currently has the highest available supply?

Required result:

- intent=current_supply
- requires_live_data=true
- tool=get_active_listings
- group_by=energy_source
- metric=energy_kwh

Question:
What percentage of active marketplace supply comes from Solar, Wind, and Hydro?

Required result:

- intent=supply_mix
- requires_live_data=true
- tool=get_active_listings
- no energy_source filter
- group_by=energy_source
- metrics=energy_kwh

Question:
Compare demand for Solar, Wind, and Hydro during last month.

Required result:

- intent=historical_demand
- requires_live_data=true
- tool=get_all_purchases
- status=completed
- completed_from=last_month_start
- completed_to=last_month_end
- no energy_source filter

Question:
Which renewable source had the highest average selling price last month?

Required result:

- intent=average_selling_price
- requires_live_data=true
- tool=get_all_purchases
- status=completed
- completed_from=last_month_start
- completed_to=last_month_end

Question:
Predict which renewable source will have the highest demand next month.

Required result:

- intent=demand_prediction
- requires_live_data=true
- tools=get_all_listings and get_all_purchases
- historical filters=previous 180 days through today
- forecast period=next month
- no energy_source filter
- group_by=energy_source and week

Question:
Is Noida likely to face a Solar-credit shortage next month?

Required result:

- intent=shortage_prediction
- requires_live_data=true
- tools=get_all_listings, get_all_purchases, get_active_listings
- energy_source=SOLAR on every tool
- location=Noida on every tool
- historical filters=previous 180 days through today
- forecast period=next month

Question:
Which location has the greatest Wind-credit supply?

Required result:

- intent=supply_by_location
- requires_live_data=true
- tool=get_active_listings
- energy_source=WIND
- group_by=location

Question:
Should I list Solar or Wind credits this week?

Required result:

- intent=seller_recommendation
- requires_live_data=true
- tools=get_all_listings, get_all_purchases, get_active_listings
- historical filters=previous 28 days through today
- no single energy_source filter
- group_by=energy_source and week

SAFETY RULES:

- Only select tools present in TOOL_CATALOG.
- Never select POST, PATCH, PUT, or DELETE operations.
- Never select document-ingestion tools.
- Never select authenticated-user tools.
- Never select blockchain tools.
- Never answer the user's question.
- Never calculate predictions in the planner.
- Never invent marketplace facts.
- Return only valid JSON.
- Do not return Markdown or code fences.

ALLOWED INTENTS:

- none
- current_supply
- supply_mix
- historical_supply
- historical_demand
- average_selling_price
- demand_supply_ratio
- market_balance
- supply_stability
- price_volatility
- supply_by_location
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
  "group_by": [
    "energy_source"
  ],
  "metrics": [
    "energy_kwh"
  ],
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
  "historical_period": {{
    "from": null,
    "to": null
  }},
  "forecast_period": {{
    "from": null,
    "to": null
  }},
  "group_by": [],
  "metrics": [],
  "missing_parameters": [],
  "tool_calls": []
}}

USER QUESTION:

{question}
""".strip()


def plan_api_calls(question: str) -> Dict[str, Any]:
    value = (question or "").strip()
    if not value:
        return _empty_plan("empty_question")
    dates = _date_context()
    prompt = _planner_prompt(value, tool_registry.build_planner_tools_text(), dates)
    try:
        response = _get_bedrock_client().converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": settings.BEDROCK_PLANNER_MAX_TOKENS,
                "temperature": settings.BEDROCK_PLANNER_TEMPERATURE,
            },
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        planner_text = "\n".join(
            block["text"].strip() for block in blocks
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        )
        plan = _normalize_plan(_parse_json(planner_text))
        plan = _enforce_required_plan_components(plan, dates)
        logger.info("API plan intent=%s calls=%s", plan["intent"], plan["tool_calls"])
        return plan
    except (ClientError, BotoCoreError) as exc:
        logger.exception("Planner invocation failed")
        return _empty_plan(f"planner_bedrock_error:{type(exc).__name__}")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Invalid planner output: %s", exc)
        return _empty_plan("planner_invalid_output")


def _parse_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty planner output")
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL | re.I)
    if match:
        cleaned = match.group(1)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Planner output is not an object")
    return result


def _normalize_plan(raw: Mapping[str, Any]) -> Dict[str, Any]:
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

    if intent != "none" and intent in {
        "current_supply",
        "supply_mix",
        "historical_supply",
        "historical_demand",
        "average_selling_price",
        "demand_supply_ratio",
        "market_balance",
        "supply_stability",
        "price_volatility",
        "supply_by_location",
        "demand_prediction",
        "price_prediction",
        "shortage_prediction",
        "seller_recommendation",
        "buyer_recommendation",
    }:
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


def _enforce_required_plan_components(plan: Mapping[str, Any], dates: Mapping[str, str]) -> Dict[str, Any]:
    enforced = dict(plan)
    intent = str(enforced.get("intent", "none"))
    calls = [dict(call) for call in enforced.get("tool_calls", []) if isinstance(call, Mapping)]
    if intent not in REQUIRED_TOOLS_BY_INTENT:
        enforced["requires_api_data"] = bool(calls) or bool(enforced.get("requires_api_data"))
        enforced["tool_calls"] = calls
        return enforced

    history_from, history_to = _history_range(intent, enforced.get("historical_period"), dates)
    source = _first_arg(calls, "energy_source")
    location = _first_arg(calls, "location")
    indexed = {str(call.get("tool")): call for call in calls}
    output: List[Dict[str, Any]] = []

    for tool in REQUIRED_TOOLS_BY_INTENT[intent]:
        call = indexed.get(tool, {"tool": tool, "arguments": {}})
        args = dict(call.get("arguments", {}) or {})
        if source and intent not in {"demand_prediction", "price_prediction", "seller_recommendation"}:
            args.setdefault("energy_source", source)
        if location:
            args.setdefault("location", location)
        if tool == "get_all_listings":
            args["created_from"] = history_from
            args["created_to"] = history_to
        elif tool == "get_all_purchases":
            args["status"] = "completed"
            args["completed_from"] = history_from
            args["completed_to"] = history_to
        args.setdefault("skip", 0)
        args.setdefault("limit", settings.MARKETPLACE_API_PAGE_SIZE)
        output.append({"tool": tool, "arguments": _normalize_args(tool, args)})

    enforced["tool_calls"] = output
    enforced["requires_api_data"] = True
    enforced["historical_period"] = {"from": history_from, "to": history_to}
    if intent in PREDICTION_INTENTS:
        forecast = enforced.get("forecast_period", {})
        if not forecast.get("from") or not forecast.get("to"):
            enforced["forecast_period"] = {
                "from": dates["next_month_start"], "to": dates["next_month_end"]
            }

    groups = {
        "demand_supply_ratio": ["energy_source"], "market_balance": ["energy_source"],
        "demand_prediction": ["energy_source", "week"],
        "price_prediction": ["energy_source", "week"],
        "shortage_prediction": ["week"],
        "seller_recommendation": ["energy_source", "week"],
        "buyer_recommendation": ["energy_source"],
    }
    metrics = {
        "demand_supply_ratio": ["energy_kwh", "demand_supply_ratio"],
        "market_balance": ["energy_kwh", "market_balance"],
        "demand_prediction": ["energy_kwh", "demand_supply_ratio"],
        "price_prediction": ["price_per_kwh", "energy_kwh"],
        "shortage_prediction": ["energy_kwh", "market_balance"],
        "seller_recommendation": ["energy_kwh", "price_per_kwh", "demand_supply_ratio"],
        "buyer_recommendation": ["energy_kwh", "price_per_kwh"],
    }
    enforced["group_by"] = _merge(enforced.get("group_by"), groups[intent])
    enforced["metrics"] = _merge(enforced.get("metrics"), metrics[intent])
    return enforced


def _history_range(intent: str, period: Any, dates: Mapping[str, str]) -> tuple[str, str]:
    supplied = _period(period)
    if intent in PREDICTION_INTENTS:
        if supplied["from"] and supplied["to"]:
            try:
                start = date.fromisoformat(supplied["from"][:10])
                end = date.fromisoformat(supplied["to"][:10])
                if (end - start).days >= settings.ANALYTICS_DEFAULT_HISTORY_DAYS - 1:
                    return supplied["from"], supplied["to"]
            except ValueError:
                pass
        return dates["rolling_180_days_start"], dates["today"]
    if intent == "seller_recommendation":
        return dates["rolling_28_days_start"], dates["today"]
    if supplied["from"] and supplied["to"]:
        return supplied["from"], supplied["to"]
    return dates["rolling_28_days_start"], dates["today"]


def _normalize_calls(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = set(tool_registry.get_allowed_tool_names()) & MARKETPLACE_TOOLS
    output, seen = [], set()
    for call in value:
        if not isinstance(call, Mapping):
            continue
        tool = str(call.get("tool", ""))
        if tool not in allowed:
            continue
        normalized = {"tool": tool, "arguments": _normalize_args(tool, call.get("arguments"))}
        key = json.dumps(normalized, sort_keys=True)
        if key not in seen:
            seen.add(key)
            output.append(normalized)
    return output[:4]


def _normalize_args(tool: str, value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    definition = tool_registry.get_tool_by_name(tool) or {}
    schema = definition.get("payload_schema", {})
    allowed = set(schema.keys()) if isinstance(schema, Mapping) else set()
    out: Dict[str, Any] = {}
    for key, item in raw.items():
        if key not in allowed or item in (None, ""):
            continue
        if key == "energy_source":
            source = _source(item)
            if source: out[key] = source
        elif key == "location" and isinstance(item, str): out[key] = item.strip()[:200]
        elif key == "status":
            status = str(item).strip().lower()
            valid = PURCHASE_STATUSES if tool == "get_all_purchases" else LISTING_STATUSES
            if status in valid: out[key] = status
        elif key in {"created_from", "created_to", "completed_from", "completed_to"}:
            parsed = _iso(item)
            if parsed: out[key] = parsed
        elif key in {"min_price_per_kwh", "max_price_per_kwh"}:
            try:
                number = float(item)
                if number > 0: out[key] = number
            except (TypeError, ValueError): pass
        elif key in {"min_energy_kwh", "limit"}:
            try:
                number = int(item)
                if number > 0: out[key] = min(number, settings.MARKETPLACE_API_PAGE_SIZE) if key == "limit" else number
            except (TypeError, ValueError): pass
        elif key == "skip":
            try:
                number = int(item)
                if number >= 0: out[key] = number
            except (TypeError, ValueError): pass
        elif key == "sort_by" and str(item).lower() in SORT_FIELDS: out[key] = str(item).lower()
        elif key == "sort_order" and str(item).lower() in SORT_ORDERS: out[key] = str(item).lower()
    out.setdefault("skip", 0)
    out.setdefault("limit", settings.MARKETPLACE_API_PAGE_SIZE)
    if tool == "get_all_purchases": out.setdefault("status", "completed")
    return out


def _source(value: Any) -> Optional[str]:
    if not isinstance(value, str): return None
    raw = value.strip()
    direct = raw.upper().replace("-", "_").replace(" ", "_")
    return direct if direct in SOURCES else SOURCE_ALIASES.get(raw.lower())


def _iso(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip(): return None
    candidate = value.strip()
    try: date.fromisoformat(candidate); return candidate
    except ValueError: pass
    try: datetime.fromisoformat(candidate.replace("Z", "+00:00")); return candidate
    except ValueError: return None


def _period(value: Any) -> Dict[str, Optional[str]]:
    if not isinstance(value, Mapping): return {"from": None, "to": None}
    return {"from": _iso(value.get("from")), "to": _iso(value.get("to"))}


def _first_arg(calls: Sequence[Mapping[str, Any]], key: str) -> Any:
    for call in calls:
        args = call.get("arguments", {})
        if isinstance(args, Mapping) and args.get(key) not in (None, ""):
            return args[key]
    return None


def _string_list(value: Any, allowed: Set[str]) -> List[str]:
    if not isinstance(value, list): return []
    return list(dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip().lower() in allowed))


def _missing(value: Any) -> List[str]:
    if not isinstance(value, list): return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _merge(existing: Any, required: Sequence[str]) -> List[str]:
    values = list(existing) if isinstance(existing, list) else []
    for item in required:
        if item not in values: values.append(item)
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