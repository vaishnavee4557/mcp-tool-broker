from __future__ import annotations

import re
import warnings 
from dataclasses import dataclass

from guardrails import Guard, OnFailAction , settings
from guardrails.errors import ValidationError

from guardrails_ai.detect_jailbreak import (
    DetectJailbreak,
)
from guardrails_ai.detect_pii import (
    DetectPII,
)
from guardrails_ai.restricttotopic import (
    RestrictToTopic,
)
from guardrails_ai.secrets_present import (
    SecretsPresent,
)
from guardrails_ai.toxic_language import (
    ToxicLanguage,
)
settings.disable_tracing = True 
warnings.filterwarnings(
    "ignore",
    message=(
        "The DetectSecrets validator works best "
        "with multiline code snippets.*"
    ),
    category=UserWarning,
)
# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    message: str = ""
    safe_text: str | None = None


# ============================================================
# FACTIGENT GUARDRAILS
# ============================================================

class FactigentGuardrails:
    """
    Single guardrail service for the chatbot.

    Models/validators are initialized ONCE when this class
    is instantiated.

    The same instances are reused for every user question.

    Input:
        User
        -> safety
        -> jailbreak
        -> topic
        -> PII
        -> secrets
        -> retriever

    Output:
        LLM final answer
        -> PII
        -> secrets
        -> confidential-data checks
        -> user
    """

    def __init__(
        self,
    ) -> None:

        print(
            "\nLoading Guardrails AI models..."
        )

        # ====================================================
        # CONTENT SAFETY
        # ====================================================

        self.content_safety_guard = (
            Guard()
            .use(
                ToxicLanguage(
                    threshold=0.5,
                    validation_method="full",
                    on_fail=OnFailAction.EXCEPTION,
                )
            )
        )

        # ====================================================
        # JAILBREAK
        # ====================================================

        self.jailbreak_guard = (
            Guard()
            .use(
                DetectJailbreak(
                    threshold=0.9,
                    device="cpu",
                    on_fail=OnFailAction.EXCEPTION,
                )
            )
        )

        # ====================================================
        # TOPIC CONTROL
        # ====================================================

        self.topic_guard = (
            Guard()
            .use(
                RestrictToTopic(
                    valid_topics=[
                        "industrial machines",
                        "machine telemetry",
                        "machine monitoring",
                        "machine sensors",
                        "machine parameters",
                        "machine temperature",
                        "machine vibration",
                        "machine humidity",
                        "machine voltage",
                        "machine power",
                        "machine frequency",
                        "machine level",
                        "machine health",
                        "machine risk",
                        "machine anomalies",
                        "predictive maintenance",
                        "industrial sites",
                        "industrial zones",
                        "industrial lines",
                        "industrial cells",
                        "Factigent users",
                        "Factigent roles",
                        "Factigent permissions",
                        "Factigent groups",
                        "Factigent organizations",
                        "Factigent connections",
                        "Factigent dataflows",
                        "Factigent business APIs",
                    ],
                    invalid_topics=[
                        "politics",
                        "general knowledge",
                        "weather",
                        "sports",
                        "movies",
                        "entertainment",
                        "celebrity news",
                        "unrelated geography",
                        "unrelated programming",
                        "personal advice",
                    ],
                    disable_classifier=False,
                    disable_llm=True,
                    device=-1,
                    on_fail=OnFailAction.EXCEPTION,
                )
            )
        )

        # ====================================================
        # PII
        # ====================================================
        self.pii_guard = (
            Guard()
            .use(
                DetectPII(
                    [
                        "EMAIL_ADDRESS",
                        "PHONE_NUMBER",
                        "CREDIT_CARD",
                        "IP_ADDRESS",
                    ],
                    on_fail = OnFailAction.EXCEPTION ,
                )
            )
        )

        # ====================================================
        # SECRETS
        # ====================================================

        self.secret_guard = (
            Guard()
            .use(
                SecretsPresent(
                    on_fail= OnFailAction.EXCEPTION,
                )
            )
        )
        print(
            "Guardrails AI ready."
        )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_input(
        self,
        text: str,
    ) -> GuardrailResult:

        normalized_text = self._normalize(
            text
        )

        if not normalized_text:
            return GuardrailResult(
                allowed=False,
                reason="empty",
                message=(
                    "Please enter a valid question."
                ),
            )

        # ----------------------------------------------------
        # CONTENT SAFETY
        # ----------------------------------------------------

        try:
            self.content_safety_guard.validate(
                normalized_text
            )

        except ValidationError:
            return GuardrailResult(
                allowed=False,
                reason="content_safety",
                message=(
                    "This request was blocked by "
                    "the content-safety policy."
                ),
            )

        # ----------------------------------------------------
        # JAILBREAK
        # ----------------------------------------------------

        try:
            self.jailbreak_guard.validate(
                normalized_text
            )

        except ValidationError:
            return GuardrailResult(
                allowed=False,
                reason="jailbreak",
                message=(
                    "This request appears to be attempting "
                    "to bypass the assistant's rules."
                ),
            )

        # ----------------------------------------------------
        # TOPIC / DOMAIN
        # ----------------------------------------------------

        try:
            self.topic_guard.validate(
                normalized_text
            )

        except ValidationError:
            return GuardrailResult(
                allowed=False,
                reason="off_topic",
                message=(
                    "I can only answer questions related "
                    "to supported Factigent business APIs "
                    "and machine data."
                ),
            )

        # ----------------------------------------------------
        # PII
        # ----------------------------------------------------

        try:
            self.pii_guard.validate(
                normalized_text
            )

        except ValidationError:
            return GuardrailResult(
                allowed=False,
                reason="pii",
                message=(
                    "Please remove personal or sensitive "
                    "information from your request."
                ),
            )

        # ----------------------------------------------------
        # SECRETS
        # ----------------------------------------------------

        try:
            self.secret_guard.validate(
                normalized_text
            )

        except ValidationError:
            return GuardrailResult(
                allowed=False,
                reason="secret",
                message=(
                    "Please do not include passwords, "
                    "API keys, tokens, or other secrets."
                ),
            )

        return GuardrailResult(
            allowed=True,
            safe_text=normalized_text,
        )

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    def validate_output(
        self,
        text: str,
    ) -> GuardrailResult:
        """
        Validate the FINAL answer before it is returned
        to the user.

        Output checks are focused on accidental
        confidential-data exposure.
        """

        normalized_text = self._normalize(
            text
        )

        if not normalized_text:
            return GuardrailResult(
                allowed=False,
                reason="empty_output",
                message=(
                    "The response could not be generated safely."
                ),
            )

        # ----------------------------------------------------
        # PII LEAKAGE
        # ----------------------------------------------------

        try:
            self.pii_guard.validate(
                normalized_text
            )

        except ValidationError:
            return GuardrailResult(
                allowed=False,
                reason="output_pii",
                message=(
                    "The generated response contained "
                    "sensitive personal information "
                    "and was blocked."
                ),
            )

        # ----------------------------------------------------
        # SECRET / TOKEN LEAKAGE
        # ----------------------------------------------------

        try:
            self.secret_guard.validate(
                normalized_text
            )

        except ValidationError:
            return GuardrailResult(
                allowed=False,
                reason="output_secret",
                message=(
                    "The generated response contained "
                    "confidential credentials or secrets "
                    "and was blocked."
                ),
            )

        # ----------------------------------------------------
        # INTERNAL / PRIVATE INFRASTRUCTURE
        #
        # Deterministic checks — no model call.
        # ----------------------------------------------------

        if self._contains_private_infrastructure(
            normalized_text
        ):
            return GuardrailResult(
                allowed=False,
                reason="internal_infrastructure",
                message=(
                    "The generated response contained "
                    "internal infrastructure information "
                    "and was blocked."
                ),
            )

        # ----------------------------------------------------
        # SYSTEM PROMPT / INTERNAL INSTRUCTION LEAKAGE
        #
        # Deterministic protection.
        # ----------------------------------------------------

        if self._contains_prompt_leakage(
            normalized_text
        ):
            return GuardrailResult(
                allowed=False,
                reason="prompt_leakage",
                message=(
                    "The generated response contained "
                    "internal assistant instructions "
                    "and was blocked."
                ),
            )

        return GuardrailResult(
            allowed=True,
            safe_text=normalized_text,
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return " ".join(
            str(
                text or ""
            ).strip().split()
        )

    @staticmethod
    def _contains_private_infrastructure(
        text: str,
    ) -> bool:
        """
        Detect obvious private/internal network information.

        Examples:
            10.x.x.x
            192.168.x.x
            172.16.x.x - 172.31.x.x
            localhost
            127.0.0.1
        """

        patterns = [
            r"\b10(?:\.\d{1,3}){3}\b",

            r"\b192\.168"
            r"(?:\.\d{1,3}){2}\b",

            r"\b172\."
            r"(?:1[6-9]|2\d|3[01])"
            r"(?:\.\d{1,3}){2}\b",

            r"\b127(?:\.\d{1,3}){3}\b",

            r"\blocalhost\b",
        ]

        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            is not None
            for pattern in patterns
        )

    @staticmethod
    def _contains_prompt_leakage(
        text: str,
    ) -> bool:
        """
        Catch obvious accidental disclosure of internal
        prompt/instruction terminology.

        This is a final safety net, not the primary
        jailbreak detector.
        """

        suspicious_phrases = (
            "system prompt:",
            "developer message:",
            "hidden instructions:",
            "internal instructions:",
            "tool selection prompt:",
            "factigent_system_prompt",
            "tool_selection_prompt",
        )

        lowered = text.casefold()

        return any(
            phrase in lowered
            for phrase in suspicious_phrases
        )