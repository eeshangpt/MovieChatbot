"""Synthetic training data generation via an LLM.

This module is intentionally decoupled from any specific LLM SDK: callers
inject a simple `llm_call_fn(prompt: str) -> str` callable wrapping whatever
client they already use (Anthropic, OpenAI, a local model, etc).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Set

from .exceptions import DataGenerationError
from .schemas import Label, LabelSource, TrainingExample
from .utils import get_logger

logger = get_logger(__name__)

LLMCallFn = Callable[[str], str]

DEFAULT_POSITIVE_CATEGORIES: Dict[str, str] = {
    "movie_trivia": "general trivia questions about movies (plot, year, facts)",
    "tv_trivia": "general trivia questions about TV shows",
    "recommendations": "asking for a movie or TV show recommendation",
    "cast_crew": "questions about actors, directors, or crew of a movie or show",
    "release_streaming": "questions about release dates or where to stream something",
    "comparisons": "comparing two movies, shows, or franchises",
    "ratings_reviews": "questions about ratings, reviews, or whether a movie/show is good",
    "casual_phrasing": "casually phrased or typo-laden movie/TV questions",
}

DEFAULT_NEGATIVE_CATEGORIES: Dict[str, str] = {
    "cooking": "questions about cooking or recipes",
    "sports": "questions about sports",
    "coding": "programming or software engineering questions",
    "math_homework": "math or homework help questions",
    "weather": "questions about the weather",
    "finance": "questions about personal finance or investing",
    "travel": "questions about travel planning",
    "health": "general health or medical questions",
    "chitchat": "casual greetings or small talk unrelated to any topic",
    "news": "questions about current events or news",
    "hard_negative_celebrity": (
        "questions about a celebrity that have NOTHING to do with their "
        "movie/TV work, e.g. net worth, personal life, relationships"
    ),
    "hard_negative_books_music": (
        "questions about books or music, including ones with a movie/TV "
        "adaptation, but phrased without mentioning the adaptation"
    ),
    "hard_negative_other_entertainment": (
        "questions about video games or theatre -- entertainment, but not movies or TV"
    ),
}

_PROMPT_TEMPLATE = """Generate {n} diverse, realistic user queries for a chatbot.

Category: {category_description}

Requirements:
- Each query should sound like something a real user would type.
- Vary length, phrasing, formality, and sentence structure.
- Do not number them or add any commentary.
- Respond with ONLY a JSON array of {n} strings, nothing else.
"""


def _parse_llm_json_list(response: str) -> List[str]:
    """Extract a JSON list of strings from a (possibly fenced) LLM response."""
    cleaned = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    elif not cleaned.startswith("["):
        # Fall back to grabbing the first top-level [...] block.
        bracket_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if bracket_match:
            cleaned = bracket_match.group(0)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DataGenerationError(
            f"Could not parse JSON list from LLM response: {exc}\n"
            f"Response was: {response[:500]}"
        ) from exc

    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise DataGenerationError(
            f"Expected a JSON array of strings, got: {type(parsed)}"
        )
    return parsed


def _generate_for_category(
    llm_call_fn: LLMCallFn,
    category_description: str,
    n: int,
    max_retries: int,
) -> List[str]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        prompt = _PROMPT_TEMPLATE.format(n=n, category_description=category_description)
        try:
            response = llm_call_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            last_error = DataGenerationError(f"LLM call failed: {exc}")
            logger.warning(
                "Attempt %d/%d: LLM call failed: %s", attempt, max_retries + 1, exc
            )
            continue

        try:
            from langchain.messages import AIMessage

            if isinstance(response, AIMessage):
                response = response.content
            examples = _parse_llm_json_list(response)
            if examples:
                return examples
            last_error = DataGenerationError("LLM returned an empty list.")
        except DataGenerationError as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d: failed to parse LLM response: %s",
                attempt,
                max_retries + 1,
                exc,
            )

    raise last_error or DataGenerationError("Unknown error generating examples.")


def generate_synthetic_dataset(
    llm_call_fn: LLMCallFn,
    positive_categories: Optional[Dict[str, str]] = None,
    negative_categories: Optional[Dict[str, str]] = None,
    n_per_category: int = 20,
    max_retries: int = 2,
) -> List[TrainingExample]:
    """Generate a synthetic, labeled training set spanning diverse categories.

    Failures in an individual category are logged and skipped (not fatal) so
    that a single flaky LLM call doesn't abort the whole run. Raises
    DataGenerationError only if every category fails.

    Args:
        llm_call_fn: callable that sends a prompt to an LLM and returns its
            text response, e.g. `lambda prompt: my_client.generate(prompt)`.
        positive_categories: mapping of category name -> description for
            in-domain (movie/TV) examples. Defaults to DEFAULT_POSITIVE_CATEGORIES.
        negative_categories: mapping of category name -> description for
            out-of-domain examples. Defaults to DEFAULT_NEGATIVE_CATEGORIES.
        n_per_category: number of examples to request per category.
        max_retries: number of retries per category on failure/malformed output.

    Returns:
        List of TrainingExample with source=LabelSource.SYNTHETIC, deduplicated
        (case/whitespace-insensitive) across the entire run.

    Raises:
        DataGenerationError: if n_per_category < 1, or every category fails.
    """
    if n_per_category < 1:
        raise DataGenerationError("n_per_category must be >= 1.")

    # NOTE: use explicit `is None` checks rather than `x or default` -- an
    # empty dict ({}) is a deliberate "no categories of this type" choice
    # and must NOT silently fall back to the defaults.
    if positive_categories is None:
        positive_categories = DEFAULT_POSITIVE_CATEGORIES
    if negative_categories is None:
        negative_categories = DEFAULT_NEGATIVE_CATEGORIES
    if not positive_categories and not negative_categories:
        raise DataGenerationError(
            "Both positive_categories and negative_categories are empty; "
            "nothing to generate."
        )

    examples: List[TrainingExample] = []
    seen_normalized: Set[str] = set()
    failed_categories: List[str] = []

    plan = [
        (positive_categories, Label.RELEVANT),
        (negative_categories, Label.OUT_OF_DOMAIN),
    ]

    for categories, label in plan:
        for category_name, description in categories.items():
            try:
                raw_examples = _generate_for_category(
                    llm_call_fn, description, n_per_category, max_retries
                )
            except DataGenerationError as exc:
                logger.error(
                    "Skipping category '%s' after exhausting retries: %s",
                    category_name,
                    exc,
                )
                failed_categories.append(category_name)
                continue

            n_added = 0
            for text in raw_examples:
                normalized = text.strip().lower()
                if not normalized or normalized in seen_normalized:
                    continue
                seen_normalized.add(normalized)
                examples.append(
                    TrainingExample(
                        text=text.strip(),
                        label=label,
                        source=LabelSource.SYNTHETIC,
                        category=category_name,
                    )
                )
                n_added += 1
            logger.info(
                "Category '%s' (%s): generated %d unique examples.",
                category_name,
                label.value,
                n_added,
            )

    if not examples:
        raise DataGenerationError(
            f"Synthetic data generation produced zero examples. "
            f"Failed categories: {failed_categories}"
        )
    if failed_categories:
        logger.warning(
            "Synthetic data generation completed with %d failed categories: %s",
            len(failed_categories),
            failed_categories,
        )

    return examples
