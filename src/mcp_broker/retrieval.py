from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from difflib import get_close_matches
from typing import Any
from mcp_broker.tool_knowledge import build_tool_knowledge_text
from mcp_broker.embeddings import EmbeddingProvider
from mcp_broker.models import (
    RetrievalCandidate,
    ToolDescriptor,
    UserContext,
)
from mcp_broker.security import AccessPolicy


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


_DOMAIN_VOCAB = {
    "temperature", "temp", "vibration", "voltage", "humidity",
    "pressure", "power", "frequency", "current", "level", "flow",
    "speed", "rpm", "sensor", "sensors", "reading", "readings",
    "telemetry", "measurement", "measurements", "parameter",
    "parameters", "machine", "machines", "anomaly", "anomalies",
    "abnormal", "risk", "risky", "health", "severity", "alert",
    "alerts", "maintenance", "ticket", "incident", "pipeline",
    "training", "trained", "model", "models", "feature",
    "config", "configuration",
}


def tokenize(text: str) -> list[str]:
    """
    Normalize normal text, snake_case, kebab-case and camelCase
    into comparable lowercase tokens.
    """
    normalized = _CAMEL_BOUNDARY_RE.sub(" ", str(text))
    normalized = (
        normalized
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(".", " ")
    )
    return _TOKEN_RE.findall(normalized.lower())


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    """
    True cosine similarity.
    """
    if not left or not right:
        return 0.0

    dot = sum(
        a * b
        for a, b in zip(left, right, strict=False)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


def _collect_schema_text(
    value: Any,
    parts: list[str],
    *,
    depth: int = 0,
) -> None:
    """
    Add useful parameter names/descriptions/enums from a JSON schema
    to the text used by the retriever.

    This is important for Swagger-generated tools because the operation
    summary alone is often too weak for retrieval.
    """
    if depth > 8:
        return

    if isinstance(value, dict):
        title = value.get("title")
        description = value.get("description")
        enum_values = value.get("enum")
        properties = value.get("properties")

        if isinstance(title, str):
            parts.append(title)

        if isinstance(description, str):
            parts.append(description)

        if isinstance(enum_values, list):
            parts.extend(
                str(item)
                for item in enum_values
                if isinstance(item, (str, int, float))
            )

        if isinstance(properties, dict):
            for property_name, property_schema in properties.items():
                parts.append(str(property_name))
                _collect_schema_text(
                    property_schema,
                    parts,
                    depth=depth + 1,
                )

        items = value.get("items")
        if items is not None:
            _collect_schema_text(
                items,
                parts,
                depth=depth + 1,
            )

        # Request bodies can contain nested schemas.
        for key in ("allOf", "anyOf", "oneOf"):
            children = value.get(key)
            if isinstance(children, list):
                for child in children:
                    _collect_schema_text(
                        child,
                        parts,
                        depth=depth + 1,
                    )


def build_tool_text(tool: ToolDescriptor) -> str:
    """
    Build richer searchable text from:
    - MCP tool name
    - MCP description
    - existing searchable_text
    - input parameter/schema names
    - policy domain
    - policy keywords
    """
    parts: list[str] = [
        tool.name,
        tool.description or "",
        getattr(tool, "searchable_text", "") or "",
    ]

    policy = getattr(tool, "policy", None)

    if policy is not None:
        domain = getattr(policy, "domain", "")
        keywords = getattr(policy, "keywords", [])

        if domain:
            parts.append(str(domain))

        if keywords:
            parts.extend(str(item) for item in keywords)

    input_schema = getattr(tool, "input_schema", None)

    if isinstance(input_schema, dict):
        _collect_schema_text(
            input_schema,
            parts,
        )

    return " ".join(
        part
        for part in parts
        if part
    )


def _expand_by_intent(query: str) -> str:
    """
    Add generic synonyms that commonly appear in industrial APIs.

    This is not tied to one endpoint name. It helps a query such as
    "show MAC1 temperature" match tools described as "telemetry",
    "sensor readings", "measurements", etc.
    """
    tokens = set(tokenize(query))
    extra: set[str] = set()

    sensor_words = {
        "temperature",
        "temp",
        "vibration",
        "voltage",
        "humidity",
        "pressure",
        "power",
        "frequency",
        "current",
        "level",
        "flow",
        "speed",
        "rpm",
        "sensor",
        "sensors",
        "reading",
        "readings",
        "telemetry",
        "measurement",
        "measurements",
        "parameter",
        "parameters",
        "value",
        "values",
    }

    if tokens.intersection(sensor_words):
        extra.update(
            {
                "sensor",
                "telemetry",
                "reading",
                "readings",
                "measurement",
                "measurements",
                "parameter",
                "parameters",
                "value",
                "values",
                "latest",
                "current",
                "machine",
                "data",
            }
        )

    machine_words = {"machine", "machines"}
    listing_words = {
        "all",
        "list",
        "show",
        "available",
        "names",
        "name",
        "count",
        "which",
        "what",
    }

    if (
        tokens.intersection(machine_words)
        and tokens.intersection(listing_words)
    ):
        extra.update(
            {
                "machine",
                "machines",
                "list",
                "available",
                "name",
                "names",
                "id",
                "ids",
            }
        )

    risk_words = {
        "risk",
        "risky",
        "health",
        "severity",
        "unhealthy",
        "alert",
        "alerts",
    }

    if tokens.intersection(risk_words):
        extra.update(
            {
                "risk",
                "risky",
                "health",
                "severity",
                "alert",
                "alerts",
                "anomaly",
                "machine",
            }
        )

    anomaly_words = {
        "anomaly",
        "anomalies",
        "abnormal",
        "abnormality",
        "unusual",
        "outlier",
        "outliers",
    }

    if tokens.intersection(anomaly_words):
        extra.update(
            {
                "anomaly",
                "anomalies",
                "abnormal",
                "outlier",
                "alert",
                "machine",
            }
        )

    ticket_words = {
        "ticket",
        "tickets",
        "maintenance",
        "incident",
        "servicenow",
        "jira",
    }

    action_words = {
        "create",
        "raise",
        "open",
        "submit",
    }

    if (
        tokens.intersection(ticket_words)
        and tokens.intersection(action_words)
    ):
        extra.update(
            {
                "create",
                "maintenance",
                "ticket",
                "incident",
                "machine",
            }
        )

    if not extra:
        return query

    return f"{query} {' '.join(sorted(extra))}"


def _fuzzy_expand_query(
    query: str,
    vocabulary: set[str],
) -> str:
    """
    Correct likely user typos using the vocabulary present in tool metadata.

    Example:
        temperatue -> temperature

    Only longer tokens are fuzzy matched so short machine IDs such as
    MAC1 are not accidentally rewritten.
    """
    query_tokens = tokenize(query)
    additions: list[str] = []

    for token in query_tokens:
        if len(token) < 5:
            continue

        if token in vocabulary:
            continue

        matches = get_close_matches(
            token,
            vocabulary,
            n=1,
            cutoff=0.78,
        )

        if matches:
            additions.append(matches[0])

    if not additions:
        return query

    return f"{query} {' '.join(additions)}"


def intent_adjustment(
    query: str,
    tool: ToolDescriptor,
) -> float:
    """
    Generic deterministic reranking.

    It boosts tools whose purpose matches the query and strongly penalizes
    obviously unrelated pipeline/training/config tools for telemetry queries.
    """
    query_tokens = set(tokenize(query))
    tool_text = build_tool_text(tool)
    tool_tokens = set(tokenize(tool_text))

    adjustment = 0.0

    sensor_query_words = {
        "temperature",
        "temp",
        "vibration",
        "voltage",
        "humidity",
        "pressure",
        "power",
        "frequency",
        "current",
        "level",
        "flow",
        "speed",
        "rpm",
        "sensor",
        "sensors",
        "reading",
        "readings",
        "telemetry",
        "measurement",
        "measurements",
    }

    sensor_tool_words = {
        "sensor",
        "sensors",
        "telemetry",
        "reading",
        "readings",
        "measurement",
        "measurements",
        "parameter",
        "parameters",
        "latest",
    }

    unrelated_pipeline_words = {
        "pipeline",
        "training",
        "trained",
        "model",
        "models",
        "feature",
        "config",
        "configuration",
        "encoder",
    }

    wants_sensor_data = bool(
        query_tokens.intersection(sensor_query_words)
    )

    if wants_sensor_data:
        matched_sensor_terms = len(
            tool_tokens.intersection(sensor_tool_words)
        )

        adjustment += min(
            0.95,
            0.22 * matched_sensor_terms,
        )

        if "data" in tool_tokens:
            adjustment += 0.10

        if "machine" in tool_tokens or "machines" in tool_tokens:
            adjustment += 0.08

        if tool_tokens.intersection(unrelated_pipeline_words):
            adjustment -= 0.95

        if tool_tokens.intersection(
            {"risk", "risky", "health", "severity"}
        ):
            adjustment -= 0.40

    machine_words = {"machine", "machines"}
    listing_words = {
        "all",
        "list",
        "show",
        "available",
        "names",
        "name",
        "count",
        "which",
        "what",
    }

    wants_machine_listing = bool(
        query_tokens.intersection(machine_words)
        and query_tokens.intersection(listing_words)
    )

    if wants_machine_listing:
        if tool_tokens.intersection({"machine", "machines"}):
            adjustment += 0.30

        if tool_tokens.intersection(
            {"list", "available", "names", "name"}
        ):
            adjustment += 0.45

        if tool_tokens.intersection(
            {
                "risky",
                "risk",
                "health",
                "severity",
                "anomaly",
                "alerts",
            }
        ):
            adjustment -= 0.70

        if tool_tokens.intersection(unrelated_pipeline_words):
            adjustment -= 0.55

    wants_risk = bool(
        query_tokens.intersection(
            {
                "risk",
                "risky",
                "health",
                "severity",
                "unhealthy",
                "alerts",
            }
        )
    )

    if wants_risk:
        if tool_tokens.intersection(
            {
                "risk",
                "risky",
                "health",
                "severity",
                "alerts",
                "anomaly",
            }
        ):
            adjustment += 0.75

    wants_anomaly = bool(
        query_tokens.intersection(
            {
                "anomaly",
                "anomalies",
                "abnormal",
                "unusual",
                "outlier",
            }
        )
    )

    if wants_anomaly:
        if tool_tokens.intersection(
            {
                "anomaly",
                "anomalies",
                "abnormal",
                "outlier",
                "alert",
            }
        ):
            adjustment += 0.75

    wants_ticket = bool(
        query_tokens.intersection(
            {
                "create",
                "raise",
                "open",
                "submit",
            }
        )
        and query_tokens.intersection(
            {
                "ticket",
                "incident",
                "maintenance",
            }
        )
    )

    if wants_ticket:
        if tool_tokens.intersection(
            {
                "ticket",
                "incident",
                "maintenance",
                "create",
            }
        ):
            adjustment += 0.90

    return max(
        -1.25,
        min(1.25, adjustment),
    )


class BM25Index:
    def __init__(
        self,
        documents: Iterable[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.docs = [
            tokenize(doc)
            for doc in documents
        ]
        self.k1 = k1
        self.b = b
        self.avg_length = (
            sum(map(len, self.docs))
            / max(len(self.docs), 1)
        )

        self.doc_freq: Counter[str] = Counter()

        for doc in self.docs:
            self.doc_freq.update(
                set(doc)
            )

    def score(
        self,
        query: str,
        doc_index: int,
    ) -> float:
        if not self.docs:
            return 0.0

        query_terms = tokenize(query)
        doc = self.docs[doc_index]
        frequencies = Counter(doc)
        score = 0.0
        total_docs = len(self.docs)

        for term in query_terms:
            df = self.doc_freq.get(
                term,
                0,
            )

            idf = math.log(
                1
                + (
                    total_docs - df + 0.5
                )
                / (df + 0.5)
            )

            tf = frequencies.get(
                term,
                0,
            )

            denominator = (
                tf
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * len(doc)
                    / max(self.avg_length, 1)
                )
            )

            if denominator:
                score += (
                    idf
                    * (
                        tf
                        * (self.k1 + 1)
                    )
                    / denominator
                )

        return score


class DomainRouter:
    """
    Cheap first-stage routing.
    """

    def rank(
        self,
        query: str,
        tools: list[ToolDescriptor],
    ) -> dict[str, float]:
        query_tokens = set(
            tokenize(query)
        )

        domain_terms: dict[
            str,
            set[str],
        ] = defaultdict(set)

        for tool in tools:
            policy = getattr(tool, "policy", None)

            domain = (
                getattr(policy, "domain", "")
                if policy is not None
                else ""
            ) or "general"

            keywords = (
                getattr(policy, "keywords", [])
                if policy is not None
                else []
            )

            domain_terms[domain].update(
                tokenize(domain)
            )

            domain_terms[domain].update(
                tokenize(
                    " ".join(
                        str(item)
                        for item in keywords
                    )
                )
            )

        scores: dict[str, float] = {}

        for domain, terms in domain_terms.items():
            if not terms:
                scores[domain] = 0.0
                continue

            overlap = len(
                query_tokens.intersection(terms)
            )

            scores[domain] = (
                overlap
                / math.sqrt(len(terms))
            )

        return scores


class HybridToolRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        lexical_weight: float = 0.55,
        embedding_weight: float = 0.30,
        domain_weight: float = 0.15,
        minimum_embedding_signal: float = 0.20,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.lexical_weight = lexical_weight
        self.embedding_weight = embedding_weight
        self.domain_weight = domain_weight
        self.minimum_embedding_signal = minimum_embedding_signal

        self.tools: list[ToolDescriptor] = []
        self.tool_texts: list[str] = []
        self.tool_embeddings: list[list[float]] = []
        self.vocabulary: set[str] = set()
        self.bm25 = BM25Index([])
        self.domain_router = DomainRouter()

        # Useful for CLI debugging.
        self.last_expanded_query = ""

    async def rebuild(
        self,
        tools: list[ToolDescriptor],
    ) -> None:
        self.tools = list(tools)

        self.tool_texts = [
            build_tool_knowledge_text(tool)
            for tool in self.tools
        ]

        self.vocabulary = {
            token
            for text in self.tool_texts
            for token in tokenize(text)
            if len(token) >= 3
        }

        self.bm25 = BM25Index(
            self.tool_texts
        )

        if self.tool_texts:
            self.tool_embeddings = (
                await self.embedding_provider.embed(
                    self.tool_texts
                )
            )
        else:
            self.tool_embeddings = []

    async def retrieve(
        self,
        query: str,
        user: UserContext,
        *,
        limit: int,
    ) -> list[RetrievalCandidate]:
        allowed_indices = [
            index
            for index, tool in enumerate(self.tools)
            if AccessPolicy.is_allowed(
                user,
                tool,
            )
        ]

        if not allowed_indices:
            return []

        # 1. Fix likely typos using tool vocabulary.
        expanded_query = _fuzzy_expand_query(
            query,
            self.vocabulary | _DOMAIN_VOCAB,
        )

        # 2. Add generic industrial synonyms.
        expanded_query = _expand_by_intent(
            expanded_query
        )

        self.last_expanded_query = expanded_query

        query_vectors = (
            await self.embedding_provider.embed(
                [expanded_query]
            )
        )

        if not query_vectors:
            return []

        query_embedding = query_vectors[0]

        allowed_tools = [
            self.tools[index]
            for index in allowed_indices
        ]

        domain_scores = self.domain_router.rank(
            expanded_query,
            allowed_tools,
        )

        lexical_raw = {
            index: self.bm25.score(
                expanded_query,
                index,
            )
            for index in allowed_indices
        }

        max_lexical = max(
            lexical_raw.values(),
            default=0.0,
        )

        if max_lexical <= 0.0:
            max_lexical = 1.0

        candidates: list[
            RetrievalCandidate
        ] = []

        for index in allowed_indices:
            tool = self.tools[index]

            lexical = (
                lexical_raw[index]
                / max_lexical
            )

            embedding = max(
                0.0,
                cosine_similarity(
                    query_embedding,
                    self.tool_embeddings[index],
                ),
            )

            policy = getattr(tool, "policy", None)
            domain_name = (
                getattr(policy, "domain", "")
                if policy is not None
                else ""
            ) or "general"

            domain = domain_scores.get(
                domain_name,
                0.0,
            )

            domain = min(
                1.0,
                max(0.0, domain),
            )

            adjustment = intent_adjustment(
                expanded_query,
                tool,
            )

            # Important safety/relevance gate:
            # Do not call a random API just because a hash embedding
            # produced a tiny accidental similarity such as 0.09.
            has_real_signal = (
                lexical > 0.0
                or domain > 0.0
                or adjustment > 0.0
                or embedding >= self.minimum_embedding_signal
            )

            if not has_real_signal:
                continue

            score = (
                self.lexical_weight * lexical
                + self.embedding_weight * embedding
                + self.domain_weight * domain
                + adjustment
            )

            candidates.append(
                RetrievalCandidate(
                    tool=tool,
                    score=score,
                    lexical_score=lexical,
                    embedding_score=embedding,
                    domain_score=domain,
                )
            )

        candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return candidates[:limit]
