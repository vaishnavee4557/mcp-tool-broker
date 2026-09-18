from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from difflib import get_close_matches
from typing import Any

from mcp_broker.models import (
    RetrievalCandidate,
    ToolDescriptor,
    UserContext,
)
from mcp_broker.pg_retriever import PgVectorToolRetriever
from mcp_broker.security import AccessPolicy


# ============================================================
# GENERIC TEXT NORMALIZATION
# ============================================================

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


# Generic language concepts used only for retrieval.
# These are not API names, machine names, sensor names, or endpoint rules.
_OPERATION_CONCEPTS: dict[str, set[str]] = {
    "count": {
        "count",
        "counts",
        "number",
        "numbers",
        "total",
        "totals",
        "quantity",
        "howmany",
    },
    "list": {
        "list",
        "lists",
        "listing",
        "listings",
        "available",
        "availability",
        "names",
        "name",
        "which",
    },
    "detail": {
        "detail",
        "details",
        "information",
        "info",
        "describe",
        "description",
    },
    "current": {
        "current",
        "latest",
        "recent",
        "newest",
        "now",
    },
    "history": {
        "history",
        "historical",
        "trend",
        "trends",
        "timeline",
        "chart",
        "charts",
        "graph",
        "graphs",
    },
    "risk": {
        "risk",
        "risky",
        "severity",
        "critical",
        "warning",
    },
    "health": {
        "health",
        "healthy",
        "status",
        "condition",
    },
    "anomaly": {
        "anomaly",
        "anomalies",
        "abnormal",
        "abnormality",
        "outlier",
        "outliers",
        "unusual",
    },
    "report": {
        "report",
        "reports",
        "summary",
        "summaries",
        "analytics",
    },
}


def _singular_forms(token: str) -> list[str]:
    forms = [token]

    if len(token) > 4 and token.endswith("ies"):
        forms.append(token[:-3] + "y")
    elif len(token) > 4 and token.endswith("ses"):
        forms.append(token[:-2])
    elif (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        forms.append(token[:-1])

    return forms


def tokenize(text: str) -> list[str]:
    """
    Convert arbitrary text into case-insensitive retrieval tokens.

    This function is retrieval-only. It never changes user values that are
    later sent to an API.
    """

    normalized = _CAMEL_BOUNDARY_RE.sub(
        " ",
        str(text or ""),
    )

    normalized = (
        normalized
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(".", " ")
        .replace(":", " ")
    )

    raw_tokens = _TOKEN_RE.findall(
        normalized.casefold()
    )

    tokens: list[str] = []

    for token in raw_tokens:
        tokens.extend(
            _singular_forms(token)
        )

    return list(
        dict.fromkeys(tokens)
    )


def _operation_labels(
    text: str,
) -> set[str]:
    """
    Convert natural-language operation words into generic concepts such as
    count/list/history/current. No endpoint-specific logic is used.
    """

    tokens = set(
        tokenize(text)
    )

    labels: set[str] = set()

    for label, words in _OPERATION_CONCEPTS.items():
        if tokens.intersection(words):
            labels.add(label)

    # "how many" is commonly split into two tokens.
    if "how" in tokens and "many" in tokens:
        labels.add("count")

    return labels


def _expand_operation_language(
    query: str,
) -> str:
    """
    Append canonical generic operation labels for lexical/semantic retrieval.

    Example:
        "number of available machines"
        -> "number of available machines count list"

    Original query remains unchanged outside retrieval.
    """

    labels = sorted(
        _operation_labels(query)
    )

    if not labels:
        return query

    return (
        str(query).strip()
        + " "
        + " ".join(labels)
    ).strip()


# ============================================================
# LIVE MCP / OPENAPI METADATA EXTRACTION
# ============================================================

def _collect_schema_text(
    value: Any,
    parts: list[str],
    *,
    depth: int = 0,
) -> None:
    if depth > 10:
        return

    if isinstance(value, dict):
        for field in (
            "title",
            "description",
            "type",
            "format",
        ):
            item = value.get(field)
            if isinstance(item, str) and item.strip():
                parts.append(item)

        # Keep required field names searchable, but do not duplicate/boost them.
        required = value.get("required")
        if isinstance(required, list):
            parts.extend(
                str(item)
                for item in required
                if isinstance(item, str) and item.strip()
            )

        enum_values = value.get("enum")
        if isinstance(enum_values, list):
            for item in enum_values:
                if isinstance(
                    item,
                    (str, int, float, bool),
                ):
                    parts.append(str(item))

        const_value = value.get("const")
        if isinstance(
            const_value,
            (str, int, float, bool),
        ):
            parts.append(
                str(const_value)
            )

        properties = value.get("properties")
        if isinstance(properties, dict):
            for (
                property_name,
                property_schema,
            ) in properties.items():
                parts.append(
                    str(property_name)
                )
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

        additional_properties = value.get(
            "additionalProperties"
        )
        if isinstance(
            additional_properties,
            (dict, list),
        ):
            _collect_schema_text(
                additional_properties,
                parts,
                depth=depth + 1,
            )

        for key in (
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
        ):
            children = value.get(key)
            if isinstance(children, list):
                for child in children:
                    _collect_schema_text(
                        child,
                        parts,
                        depth=depth + 1,
                    )

    elif isinstance(value, list):
        for child in value:
            _collect_schema_text(
                child,
                parts,
                depth=depth + 1,
            )


def build_schema_text(
    tool: ToolDescriptor,
) -> str:
    schema = getattr(
        tool,
        "input_schema",
        None,
    )

    if not isinstance(schema, dict):
        return ""

    parts: list[str] = []

    _collect_schema_text(
        schema,
        parts,
    )

    return " ".join(
        part
        for part in parts
        if str(part).strip()
    )


def _metadata_value(
    tool: ToolDescriptor,
    attribute: str,
) -> str:
    value = getattr(
        tool,
        attribute,
        None,
    )

    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return " ".join(
            str(item)
            for item in value
            if str(item).strip()
        )

    if isinstance(value, dict):
        return " ".join(
            f"{key} {item}"
            for key, item in value.items()
        )

    return str(value)


def build_tool_text(
    tool: ToolDescriptor,
) -> str:
    """
    Build retrieval text from live MCP/OpenAPI metadata.

    Safe generic sources are used when present:
    name, description, searchable_text, operation id, path, method, tags,
    input schema, output/response schema, and policy metadata.
    """

    parts: list[str] = [
        _metadata_value(tool, "name"),
        _metadata_value(tool, "description"),
        _metadata_value(tool, "searchable_text"),
        _metadata_value(tool, "operation_id"),
        _metadata_value(tool, "operationId"),
        _metadata_value(tool, "path"),
        _metadata_value(tool, "method"),
        _metadata_value(tool, "tags"),
        build_schema_text(tool),
    ]

    # Different registry implementations may expose response metadata using
    # different attribute names. getattr keeps this backward compatible.
    for attribute in (
        "output_schema",
        "response_schema",
        "result_schema",
    ):
        schema = getattr(
            tool,
            attribute,
            None,
        )

        if isinstance(schema, dict):
            schema_parts: list[str] = []
            _collect_schema_text(
                schema,
                schema_parts,
            )
            parts.append(
                " ".join(schema_parts)
            )

    policy = getattr(
        tool,
        "policy",
        None,
    )

    if policy is not None:
        domain = getattr(
            policy,
            "domain",
            "",
        )

        if domain:
            parts.append(
                str(domain)
            )

        keywords = getattr(
            policy,
            "keywords",
            [],
        )

        if isinstance(
            keywords,
            (list, tuple, set),
        ):
            parts.extend(
                str(item)
                for item in keywords
                if str(item).strip()
            )

    # Add canonical operation labels derived from the tool's own metadata.
    raw_text = " ".join(
        part
        for part in parts
        if part and str(part).strip()
    )

    operation_labels = sorted(
        _operation_labels(raw_text)
    )

    if operation_labels:
        parts.extend(
            operation_labels
        )

    return " ".join(
        str(part)
        for part in parts
        if part and str(part).strip()
    )


# ============================================================
# GENERIC QUERY ENRICHMENT
# ============================================================

def _fuzzy_expand_query(
    query: str,
    vocabulary: set[str],
    *,
    cutoff: float = 0.84,
) -> str:
    additions: list[str] = []

    for token in tokenize(query):
        if len(token) < 5:
            continue

        if any(
            character.isdigit()
            for character in token
        ):
            continue

        if token in vocabulary:
            continue

        matches = get_close_matches(
            token,
            vocabulary,
            n=1,
            cutoff=cutoff,
        )

        if matches:
            additions.append(
                matches[0]
            )

    if not additions:
        return query

    unique_additions = list(
        dict.fromkeys(additions)
    )

    return (
        f"{query} "
        + " ".join(
            unique_additions
        )
    )


# ============================================================
# GENERIC SCORE HELPERS
# ============================================================

def _clamp01(
    value: float,
) -> float:
    return min(
        1.0,
        max(
            0.0,
            float(value),
        ),
    )


def _token_overlap_score(
    query_tokens: set[str],
    document_tokens: set[str],
) -> float:
    if (
        not query_tokens
        or not document_tokens
    ):
        return 0.0

    overlap = len(
        query_tokens.intersection(
            document_tokens
        )
    )

    if overlap <= 0:
        return 0.0

    denominator = math.sqrt(
        len(query_tokens)
        * len(document_tokens)
    )

    if denominator <= 0.0:
        return 0.0

    return _clamp01(
        overlap / denominator
    )


def _operation_compatibility_score(
    query: str,
    tool_text: str,
) -> float:
    query_labels = _operation_labels(
        query
    )

    if not query_labels:
        return 0.0

    tool_labels = _operation_labels(
        tool_text
    )

    if not tool_labels:
        return 0.0

    overlap = len(
        query_labels.intersection(
            tool_labels
        )
    )

    return _clamp01(
        overlap
        / max(
            len(query_labels),
            1,
        )
    )


# ============================================================
# BM25
# ============================================================

class BM25Index:
    def __init__(
        self,
        documents: Iterable[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.docs = [
            tokenize(document)
            for document in documents
        ]

        self.k1 = float(k1)
        self.b = float(b)

        self.avg_length = (
            sum(
                map(
                    len,
                    self.docs,
                )
            )
            / max(
                len(self.docs),
                1,
            )
        )

        self.doc_freq: Counter[str] = Counter()

        for document in self.docs:
            self.doc_freq.update(
                set(document)
            )

    def score(
        self,
        query: str,
        doc_index: int,
    ) -> float:
        if (
            not self.docs
            or doc_index < 0
            or doc_index >= len(self.docs)
        ):
            return 0.0

        query_terms = tokenize(
            query
        )

        if not query_terms:
            return 0.0

        document = self.docs[
            doc_index
        ]

        frequencies = Counter(
            document
        )

        total_docs = len(
            self.docs
        )

        score = 0.0

        for term in query_terms:
            document_frequency = self.doc_freq.get(
                term,
                0,
            )

            inverse_document_frequency = math.log(
                1.0
                + (
                    total_docs
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            term_frequency = frequencies.get(
                term,
                0,
            )

            denominator = (
                term_frequency
                + self.k1
                * (
                    1.0
                    - self.b
                    + self.b
                    * len(document)
                    / max(
                        self.avg_length,
                        1.0,
                    )
                )
            )

            if denominator <= 0.0:
                continue

            score += (
                inverse_document_frequency
                * (
                    term_frequency
                    * (
                        self.k1
                        + 1.0
                    )
                )
                / denominator
            )

        return score


# ============================================================
# GENERIC POLICY-DOMAIN SCORE
# ============================================================

class DomainRouter:
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
            policy = getattr(
                tool,
                "policy",
                None,
            )

            domain = (
                getattr(
                    policy,
                    "domain",
                    "",
                )
                if policy is not None
                else ""
            ) or "general"

            keywords = (
                getattr(
                    policy,
                    "keywords",
                    [],
                )
                if policy is not None
                else []
            )

            domain_terms[
                domain
            ].update(
                tokenize(domain)
            )

            if isinstance(
                keywords,
                (list, tuple, set),
            ):
                for keyword in keywords:
                    domain_terms[
                        domain
                    ].update(
                        tokenize(
                            str(keyword)
                        )
                    )

        scores: dict[
            str,
            float,
        ] = {}

        for (
            domain,
            terms,
        ) in domain_terms.items():
            scores[
                domain
            ] = _token_overlap_score(
                query_tokens,
                terms,
            )

        return scores


# ============================================================
# PRODUCTION PGVECTOR + BM25 RETRIEVER
# ============================================================

class PgHybridToolRetriever:
    """
    Generic MCP tool retriever.

    Important:
    - It never requires site/zone/line/cell merely because those fields exist
      in a tool schema.
    - It only ranks tools here. Argument completion and hierarchy resolution
      belong to the orchestration/context-resolution layer.
    - No machine name, sensor name, endpoint name, or Factigent-specific
      routing rule is hardcoded.
    """

    def __init__(
        self,
        pg_retriever: PgVectorToolRetriever,
        *,
        lexical_weight: float = 0.40,
        embedding_weight: float = 0.35,
        schema_weight: float = 0.08,
        domain_weight: float = 0.02,
        operation_weight: float = 0.15,
        minimum_embedding_signal: float = 0.10,
        semantic_pool_size: int = 80,
        lexical_pool_size: int = 80,
    ) -> None:
        self.pg_retriever = pg_retriever

        weights = {
            "lexical": max(
                0.0,
                float(
                    lexical_weight
                ),
            ),
            "embedding": max(
                0.0,
                float(
                    embedding_weight
                ),
            ),
            "schema": max(
                0.0,
                float(
                    schema_weight
                ),
            ),
            "domain": max(
                0.0,
                float(
                    domain_weight
                ),
            ),
            "operation": max(
                0.0,
                float(
                    operation_weight
                ),
            ),
        }

        total_weight = sum(
            weights.values()
        )

        if total_weight <= 0.0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        self.lexical_weight = (
            weights["lexical"]
            / total_weight
        )
        self.embedding_weight = (
            weights["embedding"]
            / total_weight
        )
        self.schema_weight = (
            weights["schema"]
            / total_weight
        )
        self.domain_weight = (
            weights["domain"]
            / total_weight
        )
        self.operation_weight = (
            weights["operation"]
            / total_weight
        )

        self.minimum_embedding_signal = _clamp01(
            minimum_embedding_signal
        )

        self.semantic_pool_size = max(
            1,
            int(
                semantic_pool_size
            ),
        )

        self.lexical_pool_size = max(
            1,
            int(
                lexical_pool_size
            ),
        )

        self.tools: list[
            ToolDescriptor
        ] = []

        self.tool_texts: list[
            str
        ] = []

        self.schema_texts: list[
            str
        ] = []

        self.tool_token_sets: list[
            set[str]
        ] = []

        self.schema_token_sets: list[
            set[str]
        ] = []

        self.vocabulary: set[
            str
        ] = set()

        self.bm25 = BM25Index([])
        self.domain_router = DomainRouter()

        self.last_expanded_query = ""
        self.last_score_breakdown: list[
            dict[str, Any]
        ] = []

    async def rebuild(
        self,
        tools: list[ToolDescriptor],
    ) -> None:
        self.tools = list(
            tools
        )

        self.tool_texts = [
            build_tool_text(tool)
            for tool in self.tools
        ]

        self.schema_texts = [
            build_schema_text(tool)
            for tool in self.tools
        ]

        self.tool_token_sets = [
            set(
                tokenize(text)
            )
            for text in self.tool_texts
        ]

        self.schema_token_sets = [
            set(
                tokenize(text)
            )
            for text in self.schema_texts
        ]

        self.vocabulary = {
            token
            for token_set
            in self.tool_token_sets
            for token
            in token_set
            if len(token) >= 3
        }

        self.bm25 = BM25Index(
            self.tool_texts
        )

        self.last_expanded_query = ""
        self.last_score_breakdown = []

    async def retrieve(
        self,
        query: str,
        user: UserContext,
        *,
        limit: int,
    ) -> list[RetrievalCandidate]:
        if (
            limit <= 0
            or not self.tools
            or not str(
                query
                or ""
            ).strip()
        ):
            return []

        allowed_indices = [
            index
            for (
                index,
                tool,
            ) in enumerate(
                self.tools
            )
            if AccessPolicy.is_allowed(
                user,
                tool,
            )
        ]

        if not allowed_indices:
            return []

        allowed_index_set = set(
            allowed_indices
        )

        # Retrieval-only language enrichment.
        expanded_query = _expand_operation_language(
            str(query)
        )

        expanded_query = _fuzzy_expand_query(
            expanded_query,
            self.vocabulary,
        )

        self.last_expanded_query = expanded_query

        query_tokens = set(
            tokenize(
                expanded_query
            )
        )

        # Semantic recall from pgvector.
        semantic_results = await self.pg_retriever.retrieve(
            query=expanded_query,
            limit=max(
                self.semantic_pool_size,
                limit,
            ),
            read_only_only=False,
        )

        tool_name_to_index = {
            tool.name: index
            for (
                index,
                tool,
            ) in enumerate(
                self.tools
            )
        }

        semantic_scores: dict[
            str,
            float,
        ] = {}

        semantic_indices: set[
            int
        ] = set()

        for result in semantic_results:
            tool_name = getattr(
                result,
                "tool_name",
                None,
            )

            if not tool_name:
                continue

            index = tool_name_to_index.get(
                tool_name
            )

            if (
                index is None
                or index
                not in allowed_index_set
            ):
                continue

            similarity = _clamp01(
                getattr(
                    result,
                    "similarity",
                    0.0,
                )
            )

            previous = semantic_scores.get(
                tool_name,
                0.0,
            )

            if similarity > previous:
                semantic_scores[
                    tool_name
                ] = similarity

            semantic_indices.add(
                index
            )

        # Lexical scoring is cheap, so calculate it for every authorized live
        # tool. This prevents a valid exact tool from disappearing just because
        # the vector pool or a small lexical pool omitted it.
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

        lexical_denominator = (
            max_lexical
            if max_lexical > 0.0
            else 1.0
        )

        lexical_indices = [
            index
            for index in sorted(
                allowed_indices,
                key=lambda item: lexical_raw.get(
                    item,
                    0.0,
                ),
                reverse=True,
            )
            if lexical_raw.get(
                index,
                0.0,
            ) > 0.0
        ][
            : self.lexical_pool_size
        ]

        # Always consider authorized live tools that have lexical, semantic,
        # schema, or operation evidence. We initially include all allowed
        # indices and apply the signal gate below.
        candidate_indices = (
            set(allowed_indices)
            | semantic_indices
            | set(
                lexical_indices
            )
        )

        candidate_tools = [
            self.tools[index]
            for index
            in candidate_indices
        ]

        domain_scores = self.domain_router.rank(
            expanded_query,
            candidate_tools,
        )

        candidates: list[
            RetrievalCandidate
        ] = []

        breakdown_rows: list[
            dict[str, Any]
        ] = []

        for index in candidate_indices:
            tool = self.tools[
                index
            ]

            lexical = _clamp01(
                lexical_raw.get(
                    index,
                    0.0,
                )
                / lexical_denominator
            )

            embedding = _clamp01(
                semantic_scores.get(
                    tool.name,
                    0.0,
                )
            )

            schema = _token_overlap_score(
                query_tokens,
                self.schema_token_sets[
                    index
                ],
            )

            operation = _operation_compatibility_score(
                expanded_query,
                self.tool_texts[
                    index
                ],
            )

            policy = getattr(
                tool,
                "policy",
                None,
            )

            domain_name = (
                getattr(
                    policy,
                    "domain",
                    "",
                )
                if policy is not None
                else ""
            ) or "general"

            domain = _clamp01(
                domain_scores.get(
                    domain_name,
                    0.0,
                )
            )

            has_real_signal = (
                lexical > 0.0
                or schema > 0.0
                or operation > 0.0
                or domain > 0.0
                or embedding
                >= self.minimum_embedding_signal
            )

            if not has_real_signal:
                continue

            score = (
                self.lexical_weight
                * lexical
                + self.embedding_weight
                * embedding
                + self.schema_weight
                * schema
                + self.domain_weight
                * domain
                + self.operation_weight
                * operation
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

            breakdown_rows.append(
                {
                    "tool": tool.name,
                    "score": score,
                    "lexical": lexical,
                    "pgvector": embedding,
                    "schema": schema,
                    "domain": domain,
                    "operation": operation,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item.score,
                item.tool.name,
            )
        )

        breakdown_rows.sort(
            key=lambda row: (
                -float(
                    row.get(
                        "score",
                        0.0,
                    )
                ),
                str(
                    row.get(
                        "tool",
                        "",
                    )
                ),
            )
        )

        self.last_score_breakdown = breakdown_rows

        return candidates[
            :limit
        ]
