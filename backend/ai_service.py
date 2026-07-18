"""
ai_service.py — Gemma 4 integration for myShop AI

Priority order for inference:
  1. Custom GEMMA_API_URL endpoint  (Vertex AI, self-hosted, AI Studio proxy)
  2. Google AI Studio REST API      (gemma-4-31b-it, gemma-4-26b-a4b-it)
  3. HuggingFace Serverless         (google/gemma-2-9b-it fallback)
  4. Deterministic Bengali fallback (only when every provider fails)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────


def _strip_env(value: str) -> str:
    return (value or "").strip().strip('"').strip("'")


GEMMA_API_URL = _strip_env(os.getenv("GEMMA_API_URL", ""))
GEMMA_API_KEY = _strip_env(os.getenv("GEMMA_API_KEY", ""))
GEMMA_MODEL = _strip_env(os.getenv("GEMMA_MODEL", "gemma-4-31b-it")) or "gemma-4-31b-it"

GOOGLE_AI_STUDIO_KEY = _strip_env(os.getenv("GOOGLE_AI_STUDIO_KEY", GEMMA_API_KEY))
HF_API_KEY = _strip_env(os.getenv("HUGGINGFACE_API_KEY", ""))

GOOGLE_AI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/models/google"

# Gemma 4 models return separate thought/answer parts; MINIMAL avoids 500s and timeouts.
GEMMA4_MODELS = ("gemma-4-31b-it", "gemma-4-26b-a4b-it")

DEFAULT_TIMEOUT_SECONDS = 90
QUERY_TIMEOUT_SECONDS = 90
REPORT_TIMEOUT_SECONDS = 120
# Gemma 4 returns 500 for long report prompts below ~3072 output tokens.
INSIGHTS_MAX_OUTPUT_TOKENS = 4096
QUERY_MAX_OUTPUT_TOKENS = 2048
MAX_OUTPUT_TOKENS = 8192
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# ─── Logging helpers ─────────────────────────────────────────────────────────


def _truncate(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated {len(text) - limit} chars]"


def _log_stage(stage: str, purpose: str, **fields: Any) -> None:
    payload = " | ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("[gemma:%s:%s] %s", purpose, stage, payload)


# ─── Response parsing ────────────────────────────────────────────────────────


def _extract_text_from_parts(parts: list[dict[str, Any]], *, include_thoughts: bool = False) -> str:
    """Extract visible answer text from Google AI Studio candidate parts."""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        is_thought = part.get("thought") is True
        if is_thought and not include_thoughts:
            continue
        chunks.append(text)
    return "\n".join(chunks).strip()


def _parse_google_ai_response(
    data: dict[str, Any],
    *,
    purpose: str,
    model: str,
    include_thoughts: bool = False,
) -> tuple[Optional[str], dict[str, Any]]:
    """
    Parse a generateContent JSON body.

    Returns (answer_text_or_none, debug_metadata).
    Gemma 4 may return:
      - answer-only parts
      - thought + answer parts (thought flagged with "thought": true)
      - thought-only parts when output budget is exhausted
    """
    meta: dict[str, Any] = {
        "model": model,
        "candidate_count": 0,
        "part_count": 0,
        "thought_part_count": 0,
        "answer_part_count": 0,
        "finish_reason": None,
        "block_reason": None,
        "usage": data.get("usageMetadata"),
    }

    candidates = data.get("candidates") or []
    meta["candidate_count"] = len(candidates)
    if not candidates:
        prompt_feedback = data.get("promptFeedback") or {}
        meta["block_reason"] = prompt_feedback.get("blockReason")
        _log_stage("parse", purpose, model=model, result="no_candidates", meta=meta)
        return None, meta

    candidate = candidates[0] or {}
    meta["finish_reason"] = candidate.get("finishReason")

    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    meta["part_count"] = len(parts)

    thought_texts: list[str] = []
    answer_texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = (part.get("text") or "").strip()
        if not text:
            continue
        if part.get("thought") is True:
            meta["thought_part_count"] += 1
            thought_texts.append(text)
        else:
            meta["answer_part_count"] += 1
            answer_texts.append(text)

    answer_text = "\n".join(answer_texts).strip()
    thought_text = "\n".join(thought_texts).strip()

    if include_thoughts:
        combined = "\n".join(filter(None, [thought_text, answer_text])).strip()
        if combined:
            _log_stage(
                "parse",
                purpose,
                model=model,
                result="combined_with_thoughts",
                finish_reason=meta["finish_reason"],
                answer_len=len(combined),
            )
            return combined, meta

    if answer_text:
        _log_stage(
            "parse",
            purpose,
            model=model,
            result="answer_only",
            finish_reason=meta["finish_reason"],
            answer_len=len(answer_text),
            thought_parts=meta["thought_part_count"],
        )
        return answer_text, meta

    if thought_text:
        _log_stage(
            "parse",
            purpose,
            model=model,
            result="thought_only_no_answer",
            finish_reason=meta["finish_reason"],
            thought_len=len(thought_text),
        )
        return None, meta

    _log_stage(
        "parse",
        purpose,
        model=model,
        result="empty_parts",
        finish_reason=meta["finish_reason"],
    )
    return None, meta


def _model_attempt_order(preferred: str) -> list[str]:
    ordered: list[str] = []
    for model in (preferred, GEMMA_MODEL, *GEMMA4_MODELS):
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def _build_generation_config(
    max_tokens: int,
    *,
    temperature: float = 0.4,
    thinking_level: str = "MINIMAL",
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    # Required for Gemma 4 REST compatibility; prevents 500s and thought-only replies.
    config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return config


# ─── Low-level callers ────────────────────────────────────────────────────────


def _call_custom_endpoint(
    prompt: str,
    max_tokens: int,
    *,
    purpose: str,
) -> Optional[str]:
    """Call a self-hosted or proxied Gemma endpoint."""
    if not GEMMA_API_URL or not GEMMA_API_KEY:
        _log_stage("custom_skip", purpose, reason="missing_url_or_key")
        return None

    url = GEMMA_API_URL.rstrip("/")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": _build_generation_config(max_tokens),
    }
    headers = {
        "Authorization": f"Bearer {GEMMA_API_KEY}",
        "Content-Type": "application/json",
    }

    _log_stage("custom_request", purpose, url=url, max_tokens=max_tokens, prompt_len=len(prompt))

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT_SECONDS)
        _log_stage("custom_response", purpose, status=resp.status_code, body=_truncate(resp.text))
        if resp.status_code != 200:
            return None

        data = resp.json()
        if "candidates" in data:
            text, _ = _parse_google_ai_response(data, purpose=purpose, model="custom-endpoint")
            return text
        if isinstance(data, list) and data:
            generated = data[0].get("generated_text", "").replace(prompt, "").strip()
            return generated or None
    except requests.RequestException as exc:
        logger.warning("[gemma:%s:custom_error] %s", purpose, exc)
    return None


def _call_google_ai_studio(
    prompt: str,
    max_tokens: int,
    *,
    purpose: str,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    thinking_level: str = "MINIMAL",
    temperature: float = 0.4,
) -> Optional[str]:
    """Call Gemma 4 via Google AI Studio REST API."""
    key_to_use = _strip_env(api_key or GOOGLE_AI_STUDIO_KEY)
    if not key_to_use:
        _log_stage("studio_skip", purpose, reason="missing_api_key")
        return None

    models = _model_attempt_order(GEMMA_MODEL)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": _build_generation_config(
            max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
        ),
    }

    _log_stage(
        "studio_request",
        purpose,
        models=models,
        max_tokens=max_tokens,
        thinking_level=thinking_level,
        temperature=temperature,
        prompt_len=len(prompt),
        prompt_preview=_truncate(prompt, 400),
        payload=json.dumps(payload, ensure_ascii=False)[:800],
    )

    for model in models:
        url = f"{GOOGLE_AI_BASE}/{model}:generateContent?key={key_to_use}"

        for attempt in range(1, MAX_RETRIES + 1):
            _log_stage("studio_attempt", purpose, model=model, attempt=attempt, url=_truncate(url, 180))
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
            except requests.Timeout as exc:
                logger.warning(
                    "[gemma:%s:studio_timeout] model=%s attempt=%s error=%s",
                    purpose,
                    model,
                    attempt,
                    exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                break
            except requests.RequestException as exc:
                logger.warning(
                    "[gemma:%s:studio_error] model=%s attempt=%s error=%s",
                    purpose,
                    model,
                    attempt,
                    exc,
                )
                break

            _log_stage(
                "studio_response",
                purpose,
                model=model,
                attempt=attempt,
                status=resp.status_code,
                body=_truncate(resp.text),
            )

            if resp.status_code == 404:
                break  # model unavailable — try next model

            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            if resp.status_code != 200:
                break

            try:
                data = resp.json()
            except ValueError as exc:
                logger.warning(
                    "[gemma:%s:studio_json_error] model=%s error=%s body=%s",
                    purpose,
                    model,
                    exc,
                    _truncate(resp.text),
                )
                break

            text, meta = _parse_google_ai_response(data, purpose=purpose, model=model)
            if text:
                _log_stage(
                    "studio_success",
                    purpose,
                    model=model,
                    finish_reason=meta.get("finish_reason"),
                    answer_preview=_truncate(text, 300),
                )
                return text

            # Thought-only or empty answer — retry with a higher output budget (Gemma 4 quirk).
            if (
                not text
                and meta.get("thought_part_count", 0) > 0
                and max_tokens < MAX_OUTPUT_TOKENS
            ):
                boosted_tokens = min(max_tokens * 2, MAX_OUTPUT_TOKENS)
                if boosted_tokens > max_tokens:
                    _log_stage(
                        "studio_retry_tokens",
                        purpose,
                        model=model,
                        old_max=max_tokens,
                        new_max=boosted_tokens,
                    )
                    boosted_payload = {
                        "contents": payload["contents"],
                        "generationConfig": _build_generation_config(
                            boosted_tokens,
                            temperature=temperature,
                            thinking_level="MINIMAL",
                        ),
                    }
                    try:
                        retry_resp = requests.post(url, json=boosted_payload, timeout=timeout)
                        _log_stage(
                            "studio_retry_tokens_response",
                            purpose,
                            model=model,
                            status=retry_resp.status_code,
                            body=_truncate(retry_resp.text),
                        )
                        if retry_resp.status_code == 200:
                            retry_data = retry_resp.json()
                            retry_text, _ = _parse_google_ai_response(
                                retry_data, purpose=purpose, model=model
                            )
                            if retry_text:
                                return retry_text
                    except requests.RequestException as exc:
                        logger.warning(
                            "[gemma:%s:studio_retry_tokens_error] model=%s error=%s",
                            purpose,
                            model,
                            exc,
                        )

            # Thought-only or empty answer — retry once with MINIMAL thinking if not already.
            if thinking_level != "MINIMAL":
                _log_stage("studio_retry_minimal", purpose, model=model)
                minimal_payload = {
                    "contents": payload["contents"],
                    "generationConfig": _build_generation_config(
                        max_tokens,
                        temperature=temperature,
                        thinking_level="MINIMAL",
                    ),
                }
                try:
                    retry_resp = requests.post(url, json=minimal_payload, timeout=timeout)
                    _log_stage(
                        "studio_retry_response",
                        purpose,
                        model=model,
                        status=retry_resp.status_code,
                        body=_truncate(retry_resp.text),
                    )
                    if retry_resp.status_code == 200:
                        retry_data = retry_resp.json()
                        retry_text, _ = _parse_google_ai_response(
                            retry_data, purpose=purpose, model=model
                        )
                        if retry_text:
                            return retry_text
                except requests.RequestException as exc:
                    logger.warning(
                        "[gemma:%s:studio_retry_error] model=%s error=%s",
                        purpose,
                        model,
                        exc,
                    )

            break  # try next model

    _log_stage("studio_exhausted", purpose, result="all_models_failed")
    return None


def _call_huggingface(
    prompt: str,
    max_tokens: int,
    *,
    purpose: str,
) -> Optional[str]:
    """Call Gemma via HuggingFace Serverless Inference."""
    if not HF_API_KEY:
        _log_stage("hf_skip", purpose, reason="missing_api_key")
        return None

    for model_slug in ("gemma-2-9b-it", "gemma-7b-it"):
        url = f"{HF_INFERENCE_BASE}/{model_slug}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.4,
                "return_full_text": False,
            },
        }
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        _log_stage("hf_request", purpose, model=model_slug, url=url)

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            _log_stage(
                "hf_response",
                purpose,
                model=model_slug,
                status=resp.status_code,
                body=_truncate(resp.text),
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "").strip()
                    if text:
                        return text
                if isinstance(data, dict):
                    text = data.get("generated_text", "").strip()
                    if text:
                        return text
            elif resp.status_code == 503:
                continue
        except requests.RequestException as exc:
            logger.warning("[gemma:%s:hf_error] model=%s error=%s", purpose, model_slug, exc)

    return None


def call_gemma(
    prompt: str,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
    *,
    purpose: str = "generic",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    thinking_level: str = "MINIMAL",
    temperature: float = 0.4,
) -> Optional[str]:
    """Attempt to generate content via the best available endpoint."""
    _log_stage(
        "call_start",
        purpose,
        max_tokens=max_tokens,
        timeout=timeout,
        thinking_level=thinking_level,
        prompt_len=len(prompt),
    )

    result = _call_custom_endpoint(prompt, max_tokens, purpose=purpose)
    if result:
        _log_stage("call_done", purpose, provider="custom", answer_len=len(result))
        return result

    result = _call_google_ai_studio(
        prompt,
        max_tokens,
        purpose=purpose,
        api_key=api_key,
        timeout=timeout,
        thinking_level=thinking_level,
        temperature=temperature,
    )
    if result:
        _log_stage("call_done", purpose, provider="google_ai_studio", answer_len=len(result))
        return result

    result = _call_huggingface(prompt, max_tokens, purpose=purpose)
    if result:
        _log_stage("call_done", purpose, provider="huggingface", answer_len=len(result))
        return result

    logger.warning("[gemma:%s:call_failed] All AI providers unavailable.", purpose)
    return None


# ─── High-level features ──────────────────────────────────────────────────────


def generate_insights(
    business_name: str,
    context_data: dict,
    *,
    api_key: Optional[str] = None,
    report_scope: str = "full",
) -> str:
    """Generate AI-powered business insights in Bangla and English using Gemma 4."""
    context = dict(context_data)
    context.setdefault("business_name", business_name)

    total_revenue = float(context.get("total_revenue") or 0)
    total_profit = float(context.get("total_profit") or 0)
    aov = float(context.get("avg_order_value") or 0)
    unique_customers = int(context.get("unique_customers") or 0)
    at_risk_customers = int(context.get("at_risk_customers") or 0)
    margin = float(context.get("profit_margin_pct") or 0)
    if not margin and total_revenue > 0:
        margin = total_profit / total_revenue * 100
        context["profit_margin_pct"] = round(margin, 1)

    context_str = _format_report_context(context)

    if report_scope == "daily":
        scope_instruction = (
            "This is a daily email report. Lead with today's performance, then briefly "
            "compare it to recent/all-time context where useful."
        )
    else:
        scope_instruction = (
            "This is a comprehensive full business report. Use all-time totals for overall "
            "health, today's figures for current momentum, and last-7-days data for recent trends. "
            "Never describe all-time totals as if they happened today."
        )

    prompt = f"""You are a senior business consultant writing an AI analysis report for {business_name}.

{scope_instruction}

Use ONLY the business data below. Do not invent numbers, products, or customers.

Business data:
{context_str}

Write a genuine strategic analysis with interpretation and recommendations.
Do NOT simply restate each metric in its own sentence.

Include in each language section:
- Overall business health and profit/margin assessment
- Today's performance vs broader totals (when today's data exists)
- Customer retention risk (especially the {at_risk_customers} at-risk customers)
- Inventory or operational concerns if relevant
- 2-3 specific, actionable recommendations for growth, profit, or retention

Write 6-8 sentences per language section.
Output only the two sections below. No reasoning, no bullet points, no markdown.

Bangla Section:
English Section:"""

    result = call_gemma(
        prompt,
        max_tokens=INSIGHTS_MAX_OUTPUT_TOKENS,
        api_key=api_key,
        purpose="generate_insights",
        timeout=REPORT_TIMEOUT_SECONDS,
        thinking_level="MINIMAL",
        temperature=0.65,
    )
    if result:
        return result

    # Deterministic fallback — only when every provider fails.
    if margin > 30:
        margin_bn = "আপনার প্রফিট মার্জিন চমৎকার — বর্তমান কৌশল বজায় রাখুন।"
        margin_en = "Your profit margin is excellent — maintain your current strategy."
    elif margin > 15:
        margin_bn = "প্রফিট মার্জিন সন্তোষজনক, তবে ডেলিভারি খরচ কমিয়ে আরও বাড়ানো সম্ভব।"
        margin_en = "Profit margin is satisfactory, but can be improved by reducing delivery costs."
    else:
        margin_bn = "প্রফিট মার্জিন কম — মূল্য পুনর্নির্ধারণ বা ডেলিভারি চার্জ পর্যালোচনা করুন।"
        margin_en = "Profit margin is low — consider repricing or reviewing delivery charges."

    risk_bn = (
        f"⚠️ {at_risk_customers} জন কাস্টমার ৩০+ দিন ধরে ক্রয় করেননি — তাদের জন্য বিশেষ অফার চালু করুন।"
        if at_risk_customers > 0
        else "কাস্টমার রিটেনশন ভালো আছে — লয়্যালটি প্রোগ্রাম চালু করে এটি আরও শক্তিশালী করুন।"
    )

    risk_en = (
        f"⚠️ {at_risk_customers} customers haven't purchased in 30+ days — send them a special win-back offer."
        if at_risk_customers > 0
        else "Customer retention is strong — reinforce this by launching a loyalty program."
    )

    today_revenue = context.get("today_revenue")
    today_profit = context.get("today_profit")
    today_orders = context.get("today_orders")
    if report_scope == "daily" and today_revenue is not None:
        revenue_line_bn = f"আজ {business_name}-এর মোট রেভিনিউ ৳{float(today_revenue):,.0f} এবং নেট প্রফিট ৳{float(today_profit or 0):,.0f}।"
        revenue_line_en = (
            f"Today, {business_name} generated ৳{float(today_revenue):,.0f} in revenue "
            f"with ৳{float(today_profit or 0):,.0f} net profit across {int(today_orders or 0)} orders."
        )
    else:
        revenue_line_bn = f"{business_name}-এর সর্বকালীন মোট রেভিনিউ ৳{total_revenue:,.0f} এবং নেট প্রফিট ৳{total_profit:,.0f}।"
        revenue_line_en = (
            f"All-time, {business_name} has generated ৳{total_revenue:,.0f} in revenue "
            f"with ৳{total_profit:,.0f} net profit across {int(context.get('total_orders') or 0)} orders."
        )

    return (
        "**Bangla Section:**\n"
        f"{revenue_line_bn} "
        f"{margin_bn} "
        f"গড় অর্ডার ভ্যালু ৳{aov:,.0f} — আপ-সেলিং ও বান্ডেল অফার দিয়ে এটি ৳{aov * 1.2:,.0f}-এ নিয়ে যাওয়ার লক্ষ্য রাখুন। "
        f"মোট {unique_customers} জন ইউনিক কাস্টমারের মধ্যে {at_risk_customers} জন ঝুঁকিতে আছেন — {risk_bn}\n\n"
        "**English Section:**\n"
        f"{revenue_line_en} "
        f"{margin_en} "
        f"The average order value (AOV) is ৳{aov:,.0f} — aim to increase this to ৳{aov * 1.2:,.0f} through up-selling and bundle offers. "
        f"Among {unique_customers} unique customers, {at_risk_customers} are at risk — {risk_en}"
    )


def generate_forecast_insights(
    business_name: str,
    forecast_data: dict,
    *,
    api_key: Optional[str] = None,
) -> str:
    """Generate AI-powered insights for the 14-day demand forecast."""
    prompt = f"""You are a senior supply chain and business strategist for {business_name}.
I have generated a 14-day demand forecast based on historical sales data.

Forecast Data:
{json.dumps(forecast_data, indent=2)}

Please provide a concise, strategic analysis of this forecast.
Your analysis must include exactly two sections:

**Bangla Section:**
Write a 3-4 sentence summary in Bengali. Highlight expected revenue trends (e.g. growing or stable) and call out any specific products that are expected to sell well so the business can restock them.

**English Section:**
Write a 3-4 sentence summary in English covering the exact same points (revenue trends and top product restock warnings).

Do not include any other text, reasoning, or markdown outside of these two sections.
"""
    result = call_gemma(
        prompt,
        max_tokens=INSIGHTS_MAX_OUTPUT_TOKENS,
        api_key=api_key,
        purpose="generate_forecast_insights",
        timeout=REPORT_TIMEOUT_SECONDS,
        thinking_level="MINIMAL",
        temperature=0.65,
    )
    if result:
        return result
    
    # Fallback
    daily = forecast_data.get("daily_forecast", [])
    top_products = forecast_data.get("top_products_forecast", [])
    
    if daily and len(daily) > 0:
        avg_rev = sum(d.get("predicted_revenue", 0) for d in daily) / len(daily)
        total_predicted = sum(d.get("predicted_revenue", 0) for d in daily)
    else:
        avg_rev = 0
        total_predicted = 0

    product_names = [p.get("product_name") for p in top_products[:3]]
    products_str_bn = ", ".join(product_names) if product_names else "শীর্ষ পণ্যগুলোর"
    products_str_en = ", ".join(product_names) if product_names else "top products"
    
    return (
        "**Bangla Section:**\n"
        f"{business_name}-এর আগামী ১৪ দিনের পূর্বাভাস তৈরি করা হয়েছে। প্রতিদিন গড়ে প্রায় ৳{avg_rev:,.0f} রেভিনিউ আশা করা হচ্ছে (মোট ৳{total_predicted:,.0f})। "
        f"দয়া করে {products_str_bn}-এর স্টক চেক করুন যাতে স্টকের ঘাটতি না হয়।\n\n"
        "**English Section:**\n"
        f"A 14-day forecast has been generated for {business_name}. Expected average daily revenue is ৳{avg_rev:,.0f} (Total: ৳{total_predicted:,.0f}). "
        f"Please review your stock levels for {products_str_en} to avoid stock-outs."
    )

def generate_restock_plan(
    business_name: str,
    plan_data: dict,
    *,
    api_key: Optional[str] = None,
) -> str:
    """Generate an AI-written weekly purchase plan from pre-computed restock math."""
    prompt = f"""You are a senior inventory and supply chain advisor for {business_name}, a retail shop.
I have already computed the restock math from real sales data (30-day velocity, current stock, days of stock left, suggested order quantities). Do NOT recalculate or invent any numbers — use only the data below.

Restock Data:
{json.dumps(plan_data, indent=2)}

Write a practical weekly purchase plan for the shop owner.
Your answer must include exactly two sections:

**Bangla Section:**
Write 3-5 sentences in Bengali. Tell the owner exactly which products to buy this week and roughly how many units (use the suggested_order_qty values), warn about any product that will stock out within a few days, and tell them which products to SKIP buying because they have plenty of stock or are not selling.

**English Section:**
Write 3-5 sentences in English covering the same points: what to buy this week (with quantities), the most urgent stockout risks, and what to skip. If a total budget estimate exists, mention it.

Be direct and specific, like advice from a trusted manager. Do not include any other text, reasoning, or markdown outside of these two sections.
"""
    result = call_gemma(
        prompt,
        max_tokens=INSIGHTS_MAX_OUTPUT_TOKENS,
        api_key=api_key,
        purpose="generate_restock_plan",
        timeout=REPORT_TIMEOUT_SECONDS,
        thinking_level="MINIMAL",
        temperature=0.6,
    )
    if result:
        return result

    # Deterministic fallback — only when every provider fails.
    items = plan_data.get("plan", [])
    buy_items = [i for i in items if (i.get("suggested_order_qty") or 0) > 0][:3]
    skip_items = [i for i in items if (i.get("suggested_order_qty") or 0) == 0][:2]

    buy_bn = ", ".join(f"{i['name']} ({i['suggested_order_qty']} পিস)" for i in buy_items) or "কিছু নেই"
    buy_en = ", ".join(f"{i['name']} ({i['suggested_order_qty']} units)" for i in buy_items) or "nothing"
    skip_en = ", ".join(i["name"] for i in skip_items)
    total_cost = plan_data.get("totals", {}).get("total_cost", 0)

    skip_bn_line = f" পর্যাপ্ত স্টক থাকায় {skip_en} এখন কেনার দরকার নেই।" if skip_items else ""
    skip_en_line = f" Skip {skip_en} for now — you have enough stock." if skip_items else ""

    return (
        "**Bangla Section:**\n"
        f"এই সপ্তাহে {business_name}-এর জন্য রিস্টক পরিকল্পনা তৈরি করা হয়েছে। "
        f"বিক্রির গতি অনুযায়ী এখন কেনা দরকার: {buy_bn}। "
        f"আনুমানিক মোট খরচ ৳{total_cost:,.0f}।{skip_bn_line}\n\n"
        "**English Section:**\n"
        f"A weekly restock plan has been generated for {business_name}. "
        f"Based on your sales velocity, buy now: {buy_en}. "
        f"Estimated total purchase budget: ৳{total_cost:,.0f}.{skip_en_line}"
    )


def generate_pl_summary(business_name: str, pl_data: dict, api_key: Optional[str] = None) -> str:
    """
    Generate a formal financial summary for a Profit & Loss statement.
    """
    prompt = (
        f"You are a professional financial controller for {business_name}.\n"
        f"Analyze this Profit & Loss statement data:\n"
        f"- Period: {pl_data.get('period', 'Unknown')}\n"
        f"- Gross Revenue: {pl_data.get('gross_revenue', 0)}\n"
        f"- Total COGS: {pl_data.get('total_cogs', 0)}\n"
        f"- Delivery Costs: {pl_data.get('delivery_costs', 0)}\n"
        f"- Gross Profit: {pl_data.get('gross_profit', 0)}\n"
        f"- Total Operating Expenses: {pl_data.get('total_opex', 0)}\n"
        f"- Net Profit: {pl_data.get('net_profit', 0)}\n"
        f"\n"
        f"Write a single, formal paragraph (3-4 sentences) summarizing the financial performance. "
        f"Mention the gross margin and net margin if possible. Highlight the biggest takeaway regarding profitability. "
        f"Do not use bullet points or conversational pleasantries. Write strictly in a professional, executive tone."
    )
    
    result = call_gemma(
        prompt,
        max_tokens=250,
        api_key=api_key,
        purpose="pl_summary",
        timeout=REPORT_TIMEOUT_SECONDS,
        thinking_level="MINIMAL",
        temperature=0.7,
    )
    return result or "Profit & Loss data generated successfully. Financial summary unavailable at this time."

def generate_monthly_executive_summary(
    business_name: str,
    month_str: str,
    report_data: dict,
    *,
    api_key: Optional[str] = None,
) -> str:
    """Generate a formal executive summary for a monthly business report."""
    prompt = f"""You are a professional business analyst writing an executive summary for the investors and partners of {business_name}.
I have generated the business report for the month of {month_str}.

Report Data:
{json.dumps(report_data, indent=2)}

Please write a concise, formal executive summary (3-5 sentences). Highlight the total revenue, net profit margin, and key performing products. Keep the tone professional, objective, and encouraging.
Do not use formatting like bold or headers. Just return plain paragraph text.
"""
    result = call_gemma(
        prompt,
        max_tokens=INSIGHTS_MAX_OUTPUT_TOKENS,
        api_key=api_key,
        purpose="generate_monthly_executive_summary",
        timeout=REPORT_TIMEOUT_SECONDS,
        thinking_level="MINIMAL",
        temperature=0.7,
    )
    if result:
        # Strip out any bolding the model might stubbornly add
        return result.replace("**", "").strip()
    
    # Fallback
    revenue = report_data.get("total_revenue", 0)
    profit = report_data.get("net_profit", 0)
    return (
        f"In {month_str}, {business_name} recorded a total revenue of ৳{revenue:,.0f} and a net profit of ৳{profit:,.0f}. "
        "The business maintained a steady operational pace throughout the month, supported by consistent sales across top-performing product categories. "
        "We are well-positioned to leverage this momentum going forward."
    )


def _format_context_value(value: Any) -> str:
    """Render a context field value for the model prompt."""
    if value is None:
        return "N/A"
    if isinstance(value, list):
        if not value:
            return "None"
        if all(isinstance(item, str) for item in value):
            return "; ".join(value)
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {val}" for key, val in value.items())
    return str(value)


def _format_context_for_prompt(context_data: dict) -> str:
    """Render structured context with human-readable labels for the model."""
    sections = {
        "Business Overview": {
            "business_name": "Business name",
            "last_sale_date": "Most recent sale date",
            "data_gaps": "Metrics not tracked in this system",
        },
        "Sales & Daily Performance": {
            "total_orders": "Total orders (all time)",
            "delivered_orders": "Delivered orders",
            "pending_orders": "Pending orders",
            "returned_orders": "Returned orders",
            "cancelled_orders": "Cancelled orders",
            "return_rate_pct": "Return rate (%)",
            "total_revenue": "Total revenue (BDT, all time)",
            "total_profit": "Total profit (BDT, all time)",
            "total_delivery_cost": "Total delivery cost (BDT, all time)",
            "avg_order_value": "Average order value (BDT, all time)",
            "profit_margin_pct": "Profit margin (%, all time)",
            "net_cash_flow": "Net cash flow after investments (BDT)",
            "today_orders": "Today's orders",
            "today_revenue": "Today's revenue / sales (BDT)",
            "today_profit": "Today's profit (BDT)",
            "last_7_days_orders": "Orders in the last 7 days",
            "last_7_days_revenue": "Revenue in the last 7 days (BDT)",
            "last_7_days_profit": "Profit in the last 7 days (BDT)",
            "top_products": "Top selling products",
        },
        "Customers": {
            "unique_customers": "Unique customers (all time)",
            "at_risk_customers": "At-risk customers (no purchase in 30+ days)",
            "top_customers": "Top customers by spend",
        },
        "Products & Inventory": {
            "total_products_in_inventory": "Products in inventory",
            "total_stock_value": "Inventory stock value (BDT)",
            "out_of_stock_products": "Out-of-stock products",
            "low_stock_products": "Low-stock products",
        },
        "Investments & Capital": {
            "total_investment": "Total investment spend (BDT)",
            "investment_by_category": "Investment breakdown by category",
        },
    }

    lines: list[str] = []
    for section_name, labels in sections.items():
        section_lines: list[str] = []
        for key, label in labels.items():
            if key in context_data:
                section_lines.append(f"- {label}: {_format_context_value(context_data[key])}")
        if section_lines:
            lines.append(f"[{section_name}]")
            lines.extend(section_lines)
            lines.append("")
    return "\n".join(lines).strip()


def _format_report_context(context_data: dict) -> str:
    """Render the richer dataset used for full business reports."""
    return _format_context_for_prompt(context_data)


def _is_bangla_question(question: str) -> bool:
    """Heuristic: treat questions with Bengali script as Bangla."""
    return any("\u0980" <= char <= "\u09FF" for char in question)


def _build_structured_fallback(
    business_name: str,
    question: str,
    context_data: dict,
) -> str:
    """Deterministic three-section answer when the AI provider is unavailable."""
    q_lower = question.lower()
    today_rev = float(context_data.get("today_revenue") or 0)
    today_profit = float(context_data.get("today_profit") or 0)
    today_orders = int(context_data.get("today_orders") or 0)
    total_rev = float(context_data.get("total_revenue") or 0)
    total_profit = float(context_data.get("total_profit") or 0)
    margin = float(context_data.get("profit_margin_pct") or 0)
    at_risk = int(context_data.get("at_risk_customers") or 0)
    unique = int(context_data.get("unique_customers") or 0)
    last_7_rev = float(context_data.get("last_7_days_revenue") or 0)
    last_7_orders = int(context_data.get("last_7_days_orders") or 0)
    avg_7_day = last_7_rev / 7 if last_7_rev else 0
    today_margin = (today_profit / today_rev * 100) if today_rev else 0
    out_of_stock = context_data.get("out_of_stock_products") or "None"
    low_stock = context_data.get("low_stock_products") or "None"
    bn = _is_bangla_question(question)

    if any(token in q_lower for token in ("profit", "প্রফিট", "লাভ")):
        numbers = (
            f"আজকের মোট প্রফিট **৳{today_profit:,.2f}**। মোট নেট প্রফিট **৳{total_profit:,.0f}**।"
            if bn
            else f"Today's profit is **৳{today_profit:,.2f}**. All-time net profit is **৳{total_profit:,.0f}**."
        )
    elif any(token in q_lower for token in ("revenue", "রেভিনিউ", "sales", "সেলস", "বিক্রি")):
        numbers = (
            f"আজকের মোট সেলস (Revenue) **৳{today_rev:,.2f}**। আজকের মোট অর্ডার **{today_orders}টি** এবং প্রফিট **৳{today_profit:,.2f}**।"
            if bn
            else f"Today's revenue is **৳{today_rev:,.2f}** across **{today_orders}** orders with **৳{today_profit:,.2f}** profit."
        )
    elif any(token in q_lower for token in ("average order", "aov", "avg order", "গড় অর্ডার")):
        aov = float(context_data.get("avg_order_value") or 0)
        numbers = (
            f"গড় অর্ডার ভ্যালু (AOV) **৳{aov:,.2f}**।"
            if bn
            else f"Average order value is **৳{aov:,.2f}**."
        )
    elif any(token in q_lower for token in ("customer", "কাস্টমার", "গ্রাহক")):
        numbers = (
            f"মোট **{unique}** জন ইউনিক কাস্টমার। **{at_risk}** জন At-risk (৩০+ দিন কেনাকাটা নেই)।"
            if bn
            else f"You have **{unique}** unique customers; **{at_risk}** are at-risk (no purchase in 30+ days)."
        )
    else:
        numbers = (
            f"আজকের সেলস **৳{today_rev:,.2f}**, প্রফিট **৳{today_profit:,.2f}**, মোট অর্ডার **{today_orders}টি**। "
            f"সর্বকালীন রেভিনিউ **৳{total_rev:,.0f}**, প্রফিট **৳{total_profit:,.0f}**।"
            if bn
            else f"Today: **৳{today_rev:,.2f}** revenue, **৳{today_profit:,.2f}** profit, **{today_orders}** orders. "
            f"All-time: **৳{total_rev:,.0f}** revenue, **৳{total_profit:,.0f}** profit."
        )

    if bn:
        why = (
            f"আজকের প্রফিট মার্জিন প্রায় **{today_margin:.1f}%**, যা সামগ্রিক গড় মার্জিন (**{margin:.1f}%**) এর "
            f"{'উচ্চতর' if today_margin > margin else 'নিম্নতর'}। "
            f"গত ৭ দিনের গড় দৈনিক রেভিনিউ **৳{avg_7_day:,.2f}** ({last_7_orders} অর্ডার) — "
            f"আজকের পারফরম্যান্স {'এর উপরে' if today_rev >= avg_7_day else 'এর নিচে'}।"
        )
        recommendations = (
            f"১. **ইনভেন্টরি অপ্টিমাইজেশন:** Out-of-stock: {out_of_stock}। দ্রুত রিস্টক করুন।\n"
            f"২. **কাস্টমার রি-এনগেজমেন্ট:** **{at_risk}** জন At-risk কাস্টমার ({unique} জনের মধ্যে) — টার্গেটেড অফার চালু করুন।\n"
            f"৩. **লো-স্টক অ্যালার্ট:** Low-stock: {low_stock} — স্টক পূর্ণ করুন।"
        )
    else:
        why = (
            f"Today's margin is **{today_margin:.1f}%** vs overall **{margin:.1f}%**. "
            f"7-day daily revenue average is **৳{avg_7_day:,.2f}** ({last_7_orders} orders) — "
            f"today is {'above' if today_rev >= avg_7_day else 'below'} that baseline."
        )
        recommendations = (
            f"1. **Inventory optimization:** Restock out-of-stock items ({out_of_stock}).\n"
            f"2. **Customer re-engagement:** **{at_risk}** of **{unique}** customers are at-risk — launch win-back offers.\n"
            f"3. **Low-stock alert:** Replenish low-stock items ({low_stock})."
        )

    return (
        f"**The Numbers**\n{numbers}\n\n"
        f"**The 'Why'**\n{why}\n\n"
        f"**Strategic Recommendation**\n{recommendations}"
    )


def answer_query(
    business_name: str,
    question: str,
    context_data: dict,
    api_key: Optional[str] = None,
) -> str:
    """Answer a natural language business question using Gemma 4."""
    context_str = _format_context_for_prompt(context_data)
    question = (question or "").strip()

    prompt = f"""You are an elite Business Analyst and Chief Strategy AI for {business_name}.
You power "Ask Anything About Your Sales". Always deliver a structured executive briefing.

DATA (use ONLY these facts — never invent numbers):
{context_str}

RULES:
- Cross-functional: link sales, customers, products/inventory, and investments in every answer.
- Bold every key metric and BDT amount with ** (example: **৳17,500.00**, **4টি**, **94%**).
- For today's questions use today's fields; compare against last-7-days averages and all-time margin where relevant.
- Proactively mention at-risk customers, out-of-stock, and low-stock items when the data supports it.
- If CLV/CAC/churn are unavailable, note the gap briefly in The 'Why' section.

LANGUAGE:
- If the question is in Bangla: keep headings in English exactly as below, write all body text in Bangla.
  You may include English business terms in parentheses (Revenue, At-risk, AOV).
- If the question is in English: write everything in English.

REQUIRED FORMAT — always output exactly these three sections:

**The Numbers**
[2-4 sentences answering the question directly. Bold all figures.]

**The 'Why'**
[One analytical paragraph connecting today's performance to 7-day trends, profit margin vs overall margin ({context_data.get('profit_margin_pct', 'n/a')}%), inventory, and customer retention.]

**Strategic Recommendation**
[Exactly 2-3 numbered action items. For Bangla use ১. ২. ৩. Each item starts with a **bold action title** followed by specific advice referencing real data from the dataset.]

User question: {question}

Answer:"""

    logger.info(
        "[gemma:answer_query:start] business=%s question=%s context_keys=%s",
        business_name,
        _truncate(question, 200),
        list(context_data.keys()),
    )

    result = call_gemma(
        prompt,
        max_tokens=QUERY_MAX_OUTPUT_TOKENS,
        api_key=api_key,
        purpose="answer_query",
        timeout=QUERY_TIMEOUT_SECONDS,
        thinking_level="MINIMAL",
        temperature=0.35,
    )

    if result:
        logger.info(
            "[gemma:answer_query:success] answer_preview=%s",
            _truncate(result, 300),
        )
        return result

    logger.warning(
        "[gemma:answer_query:fallback] Gemma unavailable — using structured fallback for question=%s",
        _truncate(question, 200),
    )
    return _build_structured_fallback(business_name, question, context_data)

def generate_winback_messages(
    business_name: str,
    campaign: dict,
    target_customers: list[dict],
    api_key: Optional[str] = None,
) -> dict:
    """Generate personalized win-back messages for at-risk customers."""
    if not target_customers:
        return {"campaign_brief": "No at-risk customers found.", "messages": [], "at_risk_count": 0, "combined_lifetime_spend": 0}

    import urllib.parse

    # Prepare customer summary
    cust_data = []
    combined_spend = 0
    for c in target_customers:
        combined_spend += c.get("total_revenue", 0)
        fav = c.get("favorite_category", "products")
        days = c.get("days_since_last_order", campaign.get("inactive_days", 30))
        cust_data.append(f"Name: {c.get('name', 'Customer')} | Phone: {c.get('phone')} | Spend: {c.get('total_revenue')} | Inactive: {days} days | Fav Category: {fav}")

    cust_str = "\n".join(cust_data)
    discount = campaign.get("discount_percent", 10)
    lang = campaign.get("language", "bn")
    note = campaign.get("custom_note", "")

    prompt = f"""You are an elite Marketing AI for {business_name}.
Generate a win-back campaign for the following at-risk customers.

CAMPAIGN DETAILS:
- Offer: {discount}% off
- Language: {'Bangla' if lang == 'bn' else 'English'}
- Custom Note from Owner: {note}

CUSTOMERS (MAX {campaign.get('limit', 50)}):
{cust_str}

OUTPUT STRICT JSON ONLY:
{{
  "campaign_brief": "Short 2 sentence marketing brief explaining the angle of this campaign.",
  "messages": [
    {{
      "phone": "customer phone",
      "name": "customer name",
      "days_inactive": 0,
      "message": "The personalized WhatsApp message. Include the custom note, the discount, and mention their favorite category."
    }}
  ]
}}

Ensure output is valid JSON without markdown wrapping."""

    result = call_gemma(
        prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        api_key=api_key,
        purpose="winback_preview",
        timeout=REPORT_TIMEOUT_SECONDS,
        temperature=0.7,
    )
    
    try:
        if not result:
            raise ValueError("Empty response from Gemma")
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
            
        data = json.loads(result)
        
        # Hydrate WhatsApp URLs
        for msg in data.get("messages", []):
            encoded = urllib.parse.quote(msg.get("message", ""))
            phone = msg.get("phone", "")
            if not phone.startswith("+88"):
                phone = "+88" + phone
            msg["whatsapp_url"] = f"https://wa.me/{phone}?text={encoded}"
            
        data["at_risk_count"] = len(target_customers)
        data["combined_lifetime_spend"] = combined_spend
        return data
    except Exception as e:
        logger.error(f"Failed to parse winback JSON: {e}")
        return {
            "campaign_brief": "Failed to generate campaign with AI due to invalid formatting.",
            "messages": [],
            "at_risk_count": len(target_customers),
            "combined_lifetime_spend": combined_spend
        }
