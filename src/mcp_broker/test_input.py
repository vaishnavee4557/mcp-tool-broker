from __future__ import annotations
import pytest 
from mcp_broker.security import (
    InputGuardConfig,
    InputRejectionReason,
    validate_user_question,
)
@pytest.fixture
def guard_config() -> InputGuardConfig:
    return InputGuardConfig(
        english_only=True,
        max_characters=200,
        abusive_terms=frozenset({"idiot", "stupid"}),
        harmful_phrases=frozenset({"destroy the production server", "hurt someone"}),
    )

@pytest.mark.parametrize(
    "question",
    [
        "How many machines are available?",
        "SHOW ME THE MACHINE COUNT",
        "What is the status of MACH-1?",
        "Show Compressor_24_Temperature for the last 7 days.",
        "Call GET /machines/count for site-01.",
        "How many mashines are running?",  # A spelling mistake is still valid
    ]
)
    
def test_allows_safe_english_and_technical_questions(
    question: str,
    guard_config: InputGuardConfig,
) -> None:
    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is True
    assert result.reason is None
    assert result.normalized_text == question
    assert result.user_message == ""

 
 
def test_normalizes_extra_whitespace(
    guard_config: InputGuardConfig,
) -> None:
    question = "   How many machines are available?   "

    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is True
    assert result.reason is None
    assert result.normalized_text == "How many machines are available?"
    assert result.user_message == ""

@pytest.mark.parametrize(
    "question",
    [ "" ,
        "   ",
        "\n\t",
        None,
    ],

)
def test_rejects_empty_questions(
    question: str | None,
    guard_config: InputGuardConfig,
) -> None:
    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is False
    assert result.reason is InputRejectionReason.EMPTY
    assert result.normalized_text == ""
    assert result.user_message


def test_rejects_question_over_character_limit(
    guard_config: InputGuardConfig,
) -> None:
    question = "a" * 201

    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is False
    assert result.reason is InputRejectionReason.TOO_LONG
    assert result.user_message


@pytest.mark.parametrize(
    "question",
    [
        "You are an idiot",
        "THIS BOT IS STUPID",
        "Why are you StUpId?",
    ],
)


def test_rejects_abusive_words_case_insenstively(
    question : str,
    guard_config: InputGuardConfig,
    


) -> None:
    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is False
    assert result.reason is InputRejectionReason.ABUSIVE
    assert result.user_message


def test_does_not_match_abusive_term_inside_safe_word(
    guard_config: InputGuardConfig,
) -> None:
    custom_config = InputGuardConfig(
        english_only=True,
        max_characters=200,
        abusive_terms=frozenset({"ass"}),
        harmful_phrases=guard_config.harmful_phrases,
    )

    result = validate_user_question(
        "Classify the machine alerts.",
        config=custom_config,
    )

    assert result.allowed is True
    assert result.reason is None


@pytest.mark.parametrize(
    "question",
    [
        "Tell me how to destroy the production server",
        "I want to HURT SOMEONE",
    ],
)
def test_rejects_explicit_harmful_requests(
    question: str,
    guard_config: InputGuardConfig,
) -> None:
    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is False
    assert result.reason is InputRejectionReason.HARMFUL
    assert result.user_message


@pytest.mark.parametrize(
    "question",
    [
        "मशीनों की संख्या कितनी है?",
        "¿Cuántas máquinas están funcionando actualmente?",
    ],
)
def test_rejects_non_english_questions(
    question: str,
    guard_config: InputGuardConfig,
) -> None:
    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is False
    assert result.reason is InputRejectionReason.NON_ENGLISH
    assert result.user_message


def test_language_restriction_can_be_disabled(
    guard_config: InputGuardConfig,
) -> None:
    multilingual_config = InputGuardConfig(
        english_only=False,
        max_characters=guard_config.max_characters,
        abusive_terms=guard_config.abusive_terms,
        harmful_phrases=guard_config.harmful_phrases,
    )

    result = validate_user_question(
        "मशीनों की संख्या कितनी है?",
        config=multilingual_config,
    )

    assert result.allowed is True
    assert result.reason is None


def test_blocked_result_does_not_repeat_complete_input(
    guard_config: InputGuardConfig,
) -> None:
    question = "You are an idiot"

    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is False
    assert result.normalized_text == ""
    assert question not in result.user_message


def test_boundary_value_at_maximum_length_is_allowed(
    guard_config: InputGuardConfig,
) -> None:
    question = "a" * guard_config.max_characters

    result = validate_user_question(
        question,
        config=guard_config,
    )

    assert result.allowed is True
    assert result.reason is None
