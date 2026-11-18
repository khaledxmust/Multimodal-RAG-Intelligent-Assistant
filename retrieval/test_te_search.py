import re
from qdrant_client import QdrantClient

from retrieval.embeddings import EmbeddingModel

from retrieval.vector_store import (
    COLLECTION_NAME,
    collection_exists,
    get_point_count,
    get_qdrant_client,
)


# ============================================================
# CONFIG
# ============================================================

TOP_K = 5


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [
    # Arabic
    "ما هي باقات WE Space؟",

    # English
    "What are the WE Space packages?",

    # Egyptian Arabic
    "ايه باقات النت الارضي من WE؟",

    # Arabic eKYC
    "هل خدمة eKYC متاحة طوال اليوم؟",

    # English eKYC
    "Can I use eKYC at any time?",
]

# ============================================================
# EGYPTIAN ARABIC QUERY NORMALIZATION
# ============================================================

EGYPTIAN_QUERY_REPLACEMENTS = [
    # Common Egyptian question forms
    (
        r"\b(?:ايه|إيه)\b",
        "ما هي",
    ),

    # Common Egyptian terms for fixed/home internet
    (
        r"\b(?:النت|الإنترنت)\s+(?:الارضي|الأرضي)\b",
        "الإنترنت المنزلي",
    ),

    (
        r"\bنت\s+(?:ارضي|أرضي)\b",
        "إنترنت منزلي",
    ),

    # Subscription phrasing
    (
        r"\b(?:اشترك|أشترك)\s+(?:ازاي|إزاي|ازاى|إزاى)\b",
        "كيفية الاشتراك",
    ),
]


def normalize_query(
    query: str,
) -> str:
    """
    Lightweight deterministic normalization for common
    Egyptian Arabic telecom wording.

    This does NOT translate the query and does not require
    an external API or another model.
    """

    normalized = query.strip()

    for pattern, replacement in EGYPTIAN_QUERY_REPLACEMENTS:

        normalized = re.sub(
            pattern,
            replacement,
            normalized,
            flags=re.IGNORECASE,
        )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


# ============================================================
# SEARCH
# ============================================================

def search(
    client: QdrantClient,
    embedding_model: EmbeddingModel,
    query: str,
    limit: int = TOP_K,
):
    """
    Normalize common Egyptian Arabic phrasing before
    embedding, then retrieve the most similar TE chunks.
    """

    normalized_query = normalize_query(
        query
    )

    query_vector = (
        embedding_model
        .encode_queries(
            [normalized_query]
        )[0]
        .tolist()
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return (
        response.points,
        normalized_query,
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_result(
    rank: int,
    result,
):

    payload = (
        result.payload
        or {}
    )

    print()

    print(
        f"[{rank}] "
        f"Score: {result.score:.4f}"
    )

    print(
        "Title:",
        payload.get(
            "title",
            "",
        ),
    )

    print(
        "Language:",
        payload.get(
            "content_language",
            "",
        ),
    )

    print(
        "Source kind:",
        payload.get(
            "source_kind",
            "",
        ),
    )

    section_label = (
        payload.get(
            "section_label",
            "",
        )
        or ""
    )

    if section_label:

        print(
            "Section:",
            section_label,
        )

    chunk_index = payload.get(
        "chunk_index",
        0,
    )

    chunk_total = payload.get(
        "chunk_total",
        1,
    )

    print(
        "Chunk:",
        f"{chunk_index + 1}/{chunk_total}",
    )

    print(
        "URL:",
        payload.get(
            "citation_url",
            "",
        ),
    )

    text = (
        payload.get(
            "text",
            "",
        )
        or ""
    )

    print(
        "Text:"
    )

    print(
        text[:700]
    )

    if len(
        text
    ) > 700:

        print(
            "..."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "TE RETRIEVAL TEST"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    print(
        "Connecting to Qdrant..."
    )

    client = (
        get_qdrant_client()
    )

    print(
        "Connected."
    )

    # --------------------------------------------------------
    # Validate collection
    # --------------------------------------------------------

    if not collection_exists(
        client
    ):

        raise RuntimeError(
            f"Qdrant collection "
            f"'{COLLECTION_NAME}' "
            f"does not exist. "
            f"Run the indexing script first."
        )

    point_count = (
        get_point_count(
            client
        )
    )

    if point_count == 0:

        raise RuntimeError(
            f"Qdrant collection "
            f"'{COLLECTION_NAME}' "
            f"exists but contains 0 points. "
            f"Run the indexing script first."
        )

    print(
        "Collection:",
        COLLECTION_NAME,
    )

    print(
        "Indexed points:",
        point_count,
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()

    print(
        "Loading embedding model..."
    )

    embedding_model = (
        EmbeddingModel()
    )

    print(
        "Embedding model ready."
    )

    # --------------------------------------------------------
    # Run test queries
    # --------------------------------------------------------

    for query in TEST_QUERIES:

        print()

        print(
            "=" * 80
        )

        print(
            "QUERY:"
        )

        print(
            query
        )

        print(
            "=" * 80
        )

        (
            results,
            normalized_query,
        ) = search(
            client=client,
            embedding_model=embedding_model,
            query=query,
        )

        if normalized_query != query:
            print(
                "Normalized query:"
            )

            print(
                normalized_query
            )

            print()


        if not results:

            print(
                "No results."
            )

            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print_result(
                rank,
                result,
            )

    print()

    print(
        "=" * 80
    )

    print(
        "RETRIEVAL TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()