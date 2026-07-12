"""
cost_estimator.py
Turns token counts into an estimated USD cost per model. This is an
*estimate*, not a bill: providers.py's get_completion() reports only a
single combined token count (input + output together, per each API's
usage field), so we apply a blended $/token rate per model rather than
pretending we have a true input/output split.

Prices are per 1,000,000 tokens, set from each provider's public list
pricing at the time this was written. Provider pricing changes over time —
treat PRICING as something you update periodically, not a hardcoded truth.
"""

from dataclasses import dataclass

# USD per 1M tokens. "blended" = simple average of input/output list price,
# used because get_completion() doesn't separate prompt vs completion tokens.
# GitHub Models (DeepSeek, Llama) are served free during their preview, so
# they're priced at $0 here — flip these to the real per-token rate if you
# swap to a paid DeepSeek/Llama endpoint directly.
PRICING = {
    "Gemini 3.5 Flash": {"input_per_1m": 0.075, "output_per_1m": 0.30},
    "DeepSeek R1":       {"input_per_1m": 0.0,   "output_per_1m": 0.0},
    "Llama 3.3 70B":     {"input_per_1m": 0.0,   "output_per_1m": 0.0},
}

DEFAULT_INPUT_OUTPUT_SPLIT = 0.6  # assume ~60% of a RAG call's tokens are
                                   # prompt/context, ~40% completion, absent
                                   # a real split from the API response.


@dataclass
class CostEstimate:
    model: str
    tokens: int
    usd: float
    is_free_tier: bool


def blended_rate_per_token(model: str) -> float:
    """$/token blended rate, mixing input/output list price by the assumed split."""
    prices = PRICING.get(model)
    if not prices:
        return 0.0
    blended_per_1m = (
        prices["input_per_1m"] * DEFAULT_INPUT_OUTPUT_SPLIT
        + prices["output_per_1m"] * (1 - DEFAULT_INPUT_OUTPUT_SPLIT)
    )
    return blended_per_1m / 1_000_000


def estimate_cost(model: str, tokens: int) -> CostEstimate:
    if not tokens:
        return CostEstimate(model=model, tokens=0, usd=0.0, is_free_tier=_is_free(model))
    rate = blended_rate_per_token(model)
    return CostEstimate(model=model, tokens=tokens, usd=round(tokens * rate, 6), is_free_tier=_is_free(model))


def _is_free(model: str) -> bool:
    prices = PRICING.get(model)
    return bool(prices) and prices["input_per_1m"] == 0 and prices["output_per_1m"] == 0


def format_cost(usd: float) -> str:
    if usd == 0:
        return "$0.00"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"


def total_cost(messages: list[dict]) -> float:
    """messages: st.session_state.messages — sums estimated cost across every
    assistant turn, using each message's own model (so switching models
    mid-conversation still totals correctly)."""
    total = 0.0
    for m in messages:
        if m.get("role") != "assistant" or not m.get("tokens"):
            continue
        est = estimate_cost(m.get("model", ""), m["tokens"])
        total += est.usd
    return round(total, 6)
