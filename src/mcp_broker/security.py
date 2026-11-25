from __future__ import annotations

from mcp_broker.models import (
    Operation,
    ToolDescriptor,
    UserContext,
)

import re
import unicodedata

from dataclasses import dataclass, field
from enum import StrEnum

from langdetect import (
    DetectorFactory,
    LangDetectException,
    detect,
)


# Makes language detection return consistent results.
DetectorFactory.seed = 0


DEFAULT_ABUSIVE_TERMS = frozenset(
    {
        "idiot",
        "stupid",
        "bitch",
        "fuck",
        "fucking",
        "bastard",
        "moron",
    }
)


DEFAULT_HARMFUL_PHRASES = frozenset(
    {
        "hurt someone",
        "kill someone",
        "destroy the production server",
        "damage the production server",
        "delete all production data",
    }
)


class InputRejectionReason(StrEnum):
    """Reason why a user question was rejected."""

    EMPTY = "empty"
    TOO_LONG = "too_long"
    NON_ENGLISH = "non_english"
    ABUSIVE = "abusive"
    HARMFUL = "harmful"


@dataclass(
    frozen=True,
    slots=True,
)
class InputGuardConfig:
    """Configuration used by the chatbot input guard."""

    english_only: bool = True
    max_characters: int = 2000

    abusive_terms: frozenset[str] = field(
        default_factory=lambda: DEFAULT_ABUSIVE_TERMS
    )

    harmful_phrases: frozenset[str] = field(
        default_factory=lambda: DEFAULT_HARMFUL_PHRASES
    )

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than zero."
            )


@dataclass(
    frozen=True,
    slots=True,
)
class InputValidationResult:
    """Result returned after validating a user question."""

    allowed: bool

    # Contains the cleaned question only when input is allowed.
    normalized_text: str = ""

    reason: InputRejectionReason | None = None

    # Safe message that can be displayed to the user.
    user_message: str = ""


def _normalize_question(
    question: str | None,
) -> str:
    """Remove unnecessary spaces, tabs and new lines."""

    if question is None:
        return ""

    return " ".join(
        str(question).strip().split()
    )


def _contains_blocked_term(
    text: str,
    blocked_terms: frozenset[str],
) -> bool:
    """
    Check complete words instead of matching partial words.

    For example, blocking 'ass' must not block 'classify'.
    """

    normalized_text = text.casefold()

    for blocked_term in blocked_terms:
        normalized_term = str(
            blocked_term
        ).strip().casefold()

        if not normalized_term:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_term)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized_text,
        ):
            return True

    return False


def _contains_harmful_phrase(
    text: str,
    harmful_phrases: frozenset[str],
) -> bool:
    """Check for explicitly configured harmful phrases."""

    normalized_text = text.casefold()

    for phrase in harmful_phrases:
        normalized_phrase = str(
            phrase
        ).strip().casefold()

        if (
            normalized_phrase
            and normalized_phrase in normalized_text
        ):
            return True

    return False


def _contains_non_latin_script(
    text: str,
) -> bool:
    """
    Detect letters written in non-Latin scripts.

    Numbers, punctuation, API paths and machine identifiers
    are not treated as non-English.
    """

    for character in text:
        if not character.isalpha():
            continue

        character_name = unicodedata.name(
            character,
            "",
        )

        if "LATIN" not in character_name:
            return True

    return False


def _build_language_sample(
    text: str,
) -> str:
    """
    Remove technical identifiers that can confuse language detection.

    Examples:
    MACH-1
    Compressor_24_Temperature
    /machines/count
    site-01
    """

    technical_token_pattern = (
        r"\b[\w.-]*[_/\d-][\w./-]*\b"
        r"|/[A-Za-z0-9_./-]+"
    )

    sample = re.sub(
        technical_token_pattern,
        " ",
        text,
    )

    return " ".join(
        sample.split()
    )


def _is_english(
    text: str,
) -> bool:
    """Return True when the question is English."""
    if not text:
        return False
    letter = [
        char for char in text if char.isalpha()
    ]
    if not letter:
        return True 
    return all(
        char.isascii() for char in letter
    )
    # if _contains_non_latin_script(text):
    #     return False

    # sample = _build_language_sample(text)

    # words = re.findall(
    #     r"[^\W\d_]+",
    #     sample,
    #     flags=re.UNICODE,
    # )

    # # Language detection is unreliable for very short technical
    # # inputs such as "MACH-1 status", so allow them.
    # if len(words) < 3:
    #     return True

    # try:
    #     detected_language = detect(sample)
    # except LangDetectException:
    #     # Do not block valid technical questions when the language
    #     # detector cannot make a reliable decision.
    #     return True

    # return detected_language == "en"


def _blocked_result(
    reason: InputRejectionReason,
    message: str,
) -> InputValidationResult:
    """Create a safe blocked-input result."""

    return InputValidationResult(
        allowed=False,
        normalized_text="",
        reason=reason,
        user_message=message,
    )



def validate_user_question(
    question: str | None,
    *,
    config: InputGuardConfig | None = None,
) -> InputValidationResult:
    """
    Validate a user question before any chatbot processing.

    Blocked questions must not be sent to:
    - Conversation storage
    - Embedding generation
    - Tool retrieval
    - LLM
    - MCP servers
    """

    active_config = (
        config
        if config is not None
        else InputGuardConfig()
    )

    normalized_question = _normalize_question(
        question
    )

    if not normalized_question:
        return _blocked_result(
            InputRejectionReason.EMPTY,
            "Please enter a question.",
        )

    if (
        len(normalized_question)
        > active_config.max_characters
    ):
        return _blocked_result(
            InputRejectionReason.TOO_LONG,
            (
                "Your question is too long. "
                f"Maximum allowed length is "
                f"{active_config.max_characters} characters."
            ),
        )

    if _contains_blocked_term(
        normalized_question,
        active_config.abusive_terms,
    ):
        return _blocked_result(
            InputRejectionReason.ABUSIVE,
            (
                "Your question contains inappropriate language. "
                "Please ask your question respectfully."
            ),
        )

    if _contains_harmful_phrase(
        normalized_question,
        active_config.harmful_phrases,
    ):
        return _blocked_result(
            InputRejectionReason.HARMFUL,
            (
                "This request cannot be processed because it "
                "may cause harm."
            ),
        )

    if (
        active_config.english_only
        and not _is_english(normalized_question)
    ):
        return _blocked_result(
            InputRejectionReason.NON_ENGLISH,
            "Please ask your question in English only.",
        )

    return InputValidationResult(
        allowed=True,
        normalized_text=normalized_question,
        reason=None,
        user_message="",
    )

class AccessPolicy:
    """
    Authorization boundary applied before a tool can reach the LLM.

    This class does not decide tool relevance and does not force
    hierarchy values such as site, zone, line, or cell.
    """

    @staticmethod
    def is_allowed(
        user: UserContext,
        tool: ToolDescriptor,
    ) -> bool:
        if user is None or tool is None:
            return False

        policy = getattr(
            tool,
            "policy",
            None,
        )

        if policy is None:
            return False

        # Disabled tools are never visible.
        if not bool(
            getattr(
                policy,
                "enabled",
                True,
            )
        ):
            return False

        # WRITE / ADMIN tools require explicit permission.
        operation = getattr(
            policy,
            "operation",
            Operation.READ,
        )

        if (
            operation
            in {
                Operation.WRITE,
                Operation.ADMIN,
            }
            and not bool(
                getattr(
                    user,
                    "allow_write",
                    False,
                )
            )
        ):
            return False

        # Scopes use all-of semantics.
        required_scopes = set(
            getattr(
                policy,
                "required_scopes",
                set(),
            )
            or set()
        )

        user_scopes = set(
            getattr(
                user,
                "scopes",
                set(),
            )
            or set()
        )

        if not required_scopes.issubset(
            user_scopes
        ):
            return False

        # Roles use any-of semantics, matching the original behavior.
        required_roles = set(
            getattr(
                policy,
                "required_roles",
                set(),
            )
            or set()
        )

        user_roles = set(
            getattr(
                user,
                "roles",
                set(),
            )
            or set()
        )

        if (
            required_roles
            and required_roles.isdisjoint(
                user_roles
            )
        ):
            return False

        return True
