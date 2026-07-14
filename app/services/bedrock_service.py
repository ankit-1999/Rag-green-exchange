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


def _planner_prompt(question: str, catalog: str, dates: Mapping[str, str]) -> str:
    return f"""You are the read-only API planner for GreenGrid Exchange.
Supported sources: SOLAR, WIND, HYDRO.
Select only tools in TOOL_CATALOG. Never select write, authenticated-user, or blockchain operations.

Routing:
- get_active_listings: current availability, active supply, supply mix, candidates.
- get_all_listings: historical listed supply, stability, ratios, forecast input.
- get_all_purchases: completed demand, realized prices, volatility, forecast input.

Intent rules:
- demand_supply_ratio and market_balance require listings + purchases.
- demand_prediction requires listings + purchases.
- price_prediction requires purchases + active listings.
- shortage_prediction requires listings + purchases + active listings.
- seller_recommendation requires listings + purchases + active listings.
- buyer_recommendation requires purchases + active listings.
- Predictions without an explicit history use the previous 180 days.
- Next month is the forecast period, not the history filter.
- Use status=completed for realized demand.
- Do not filter one source when comparing all sources.
- Normalize all hydro wording to HYDRO.
- Return only JSON.

DATE_CONTEXT:
{json.dumps(dict(dates), indent=2)}

TOOL_CATALOG:
{catalog}

Schema:
{{"requires_api_data":true,"reason":"short reason","intent":"allowed intent","is_prediction":false,"is_recommendation":false,"historical_period":{{"from":null,"to":null}},"forecast_period":{{"from":null,"to":null}},"group_by":[],"metrics":[],"missing_parameters":[],"tool_calls":[{{"tool":"tool name","arguments":{{}}}}]}}

Question:
{question}"""


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
    intent = str(raw.get("intent", "none") or "none").strip().lower()
    if intent not in INTENTS:
        intent = "none"
    calls = _normalize_calls(raw.get("tool_calls"))
    return {
        "requires_api_data": bool(calls) or bool(raw.get("requires_api_data", False)),
        "reason": str(raw.get("reason", "") or "").strip(),
        "intent": intent,
        "is_prediction": bool(raw.get("is_prediction", False)) or intent in PREDICTION_INTENTS,
        "is_recommendation": bool(raw.get("is_recommendation", False)) or intent in RECOMMENDATION_INTENTS,
        "historical_period": _period(raw.get("historical_period")),
        "forecast_period": _period(raw.get("forecast_period")),
        "group_by": _string_list(raw.get("group_by"), GROUPS),
        "metrics": _string_list(raw.get("metrics"), METRICS),
        "missing_parameters": _missing(raw.get("missing_parameters")),
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
        "requires_api_data": False, "reason": reason, "intent": "none",
        "is_prediction": False, "is_recommendation": False,
        "historical_period": {"from": None, "to": None},
        "forecast_period": {"from": None, "to": None},
        "group_by": [], "metrics": [], "missing_parameters": [], "tool_calls": [],
    }
