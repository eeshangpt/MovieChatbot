"""Derives a ground-truth training label from the downstream LLM's response
to a query -- the core of the production feedback loop.

Two strategies, applied in order:
  1. Heuristic refusal-pattern matching (cheap, fast, no extra LLM call).
     Reliable as long as your bot's system prompt makes refusals
     consistently detectable (which you control).
  2. LLM-as-judge fallback for responses the heuristic can't confidently
     classify (e.g. unusually short, or no pattern matched but also no
     clear engagement signal).
"""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Pattern, Tuple

from .exceptions import LabelDerivationError
from .schemas import Label, LabelSource
from .utils import get_logger

logger = get_logger(__name__)

LLMCallFn = Callable[[str], str]

# Default patterns assume a bot whose system prompt asks it to decline
# off-topic queries in a recognizable way. Override with patterns matching
# your actual refusal wording for higher heuristic accuracy.
DEFAULT_REFUSAL_PATTERNS: List[str] = [
    r"\bi can only (help|assist) with\b",
    r"\bi'?m only able to (help|assist) with\b",
    r"\b(that'?s|this is) outside (of )?(my|this) scope\b",
    r"\bi specialize (in|only in) (movies|tv|television)\b",
    r"\bi'?m not able to (help|assist) with that\b",
    r"\bi don'?t have information about that topic\b",
    r"\bi can'?t help with (that|topics) (that are )?unrelated to (movies|tv)\b",
    r"\bplease ask (me )?(a |something )?(about )?movies? or tv\b",
    r"\bi'?m designed to (only )?(answer|help with) (questions about )?movies\b",
]

_MIN_PLAUSIBLE_RESPONSE_LENGTH = 5  # chars; shorter is treated as ambiguous

_JUDGE_PROMPT_TEMPLATE = """You are auditing a chatbot that should only answer \
questions about movies and TV shows, and should decline anything else.

User query: {query}

Chatbot response: {response}

Did the chatbot's response actually engage with the query as a movie/TV \
question, or did it decline/redirect because the query was out of scope?

Respond with ONLY a JSON object of the form:
{{"label": "relevant"}} or {{"label": "out_of_domain"}}
No commentary, no markdown fences.
"""


def _compile_patterns(patterns: List[str]) -> List[Pattern]:
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as exc:
            raise LabelDerivationError(f"Invalid refusal pattern '{p}': {exc}") from exc
    return compiled


def derive_label_heuristic(
    query: str,
    llm_response: str,
    refusal_patterns: Optional[List[str]] = None,
) -> Optional[Label]:
    """Infer a label from refusal-pattern matching alone.

    Returns:
        Label.OUT_OF_DOMAIN if a refusal pattern matched.
        Label.RELEVANT if no refusal pattern matched and the response looks
            like genuine engagement (non-trivial length).
        None if the result is ambiguous (e.g. a suspiciously short/empty
            response) -- caller should fall back to the LLM judge.

    Raises:
        LabelDerivationError: if query/llm_response are empty, or a custom
            refusal pattern fails to compile as regex.
    """
    if not query or not query.strip():
        raise LabelDerivationError("derive_label_heuristic requires a non-empty query.")
    if llm_response is None:
        raise LabelDerivationError(
            "derive_label_heuristic requires a non-None llm_response."
        )

    patterns = _compile_patterns(refusal_patterns or DEFAULT_REFUSAL_PATTERNS)

    stripped_response = llm_response.strip()

    for pattern in patterns:
        if pattern.search(stripped_response):
            return Label.OUT_OF_DOMAIN

    if len(stripped_response) < _MIN_PLAUSIBLE_RESPONSE_LENGTH:
        # Too short to confidently call it genuine engagement (could be an
        # error message, an empty completion, etc.) -- defer to the judge.
        return None

    return Label.RELEVANT


def _parse_judge_response(response: str) -> Label:
    cleaned = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    elif not cleaned.startswith("{"):
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LabelDerivationError(
            f"Could not parse JSON from judge response: {exc}\nResponse was: {response[:300]}"
        ) from exc

    label_str = parsed.get("label") if isinstance(parsed, dict) else None
    if label_str not in (Label.RELEVANT.value, Label.OUT_OF_DOMAIN.value):
        raise LabelDerivationError(
            f"Judge response did not contain a valid 'label' field: {parsed}"
        )
    return Label(label_str)


def derive_label_llm_judge(
    query: str,
    llm_response: str,
    judge_call_fn: LLMCallFn,
    max_retries: int = 2,
) -> Label:
    """Ask a (typically cheap/small) LLM to judge whether the response
    actually engaged with the query as in-domain or declined it.

    Raises:
        LabelDerivationError: if every attempt fails to produce a parseable,
            valid label.
    """
    if not query or not query.strip():
        raise LabelDerivationError("derive_label_llm_judge requires a non-empty query.")
    if llm_response is None:
        raise LabelDerivationError(
            "derive_label_llm_judge requires a non-None llm_response."
        )

    prompt = _JUDGE_PROMPT_TEMPLATE.format(query=query, response=llm_response)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):
        try:
            raw = judge_call_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            last_error = LabelDerivationError(f"Judge LLM call failed: {exc}")
            logger.warning(
                "Attempt %d/%d: judge LLM call failed: %s",
                attempt,
                max_retries + 1,
                exc,
            )
            continue

        try:
            return _parse_judge_response(raw)
        except LabelDerivationError as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d: failed to parse judge response: %s",
                attempt,
                max_retries + 1,
                exc,
            )

    raise last_error or LabelDerivationError("Unknown error deriving label via judge.")


def derive_label(
    query: str,
    llm_response: str,
    judge_call_fn: Optional[LLMCallFn] = None,
    refusal_patterns: Optional[List[str]] = None,
    max_retries: int = 2,
) -> Tuple[Label, LabelSource]:
    """Full orchestration: heuristic first, LLM-judge fallback if ambiguous.

    Args:
        query: the original user query.
        llm_response: the downstream LLM's response to that query.
        judge_call_fn: callable used as the LLM-judge fallback. If None and
            the heuristic is ambiguous, raises LabelDerivationError rather
            than guessing -- callers should catch this and simply skip
            logging that example rather than poisoning the training set
            with a guessed label.
        refusal_patterns: override the default refusal-detection patterns.
        max_retries: retries for the judge fallback.

    Returns:
        (label, source) where source indicates which strategy resolved it.

    Raises:
        LabelDerivationError: if the heuristic is ambiguous and either no
            judge_call_fn was provided, or the judge itself fails.
    """
    heuristic_label = derive_label_heuristic(query, llm_response, refusal_patterns)
    if heuristic_label is not None:
        return heuristic_label, LabelSource.PRODUCTION_HEURISTIC

    if judge_call_fn is None:
        raise LabelDerivationError(
            "Heuristic label derivation was ambiguous and no judge_call_fn "
            "was provided to resolve it."
        )

    judge_label = derive_label_llm_judge(
        query, llm_response, judge_call_fn, max_retries
    )
    return judge_label, LabelSource.PRODUCTION_LLM_JUDGE
