import json
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# This file lives in:
#   TE-AI/evaluation/run_retrieval_benchmark.py
#
# When a Python file is executed directly, Python adds the
# script's own directory (evaluation/) to sys.path, not the
# project root. Add TE-AI/ explicitly so imports such as
# retrieval.*, generation.*, etc. work reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import get_qdrant_client
from generation.rag_assistant import retrieve


CASES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_cases.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_benchmark_results.md"
)


def payload_text(
    payload: dict,
) -> str:

    return " ".join(
        [
            str(
                payload.get(
                    "title",
                    "",
                )
            ),
            str(
                payload.get(
                    "section_label",
                    "",
                )
            ),
            str(
                payload.get(
                    "section",
                    "",
                )
            ),
            str(
                payload.get(
                    "text",
                    "",
                )
            ),
            str(
                payload.get(
                    "url",
                    "",
                )
            ),
            str(
                payload.get(
                    "final_url",
                    "",
                )
            ),
        ]
    )


def main():

    config = json.loads(
        CASES_PATH.read_text(
            encoding="utf-8",
        )
    )

    threshold = float(
        config.get(
            "minimum_retrieval_score",
            0.80,
        )
    )

    client = get_qdrant_client()
    embedding_model = EmbeddingModel()

    rows = []

    for case in config["cases"]:

        start = time.perf_counter()

        results, normalized_query = retrieve(
            client=client,
            embedding_model=embedding_model,
            query=case["query"],
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if results:

            top = results[0]

            top_score = float(
                top.score
            )

            payload = (
                top.payload
                or {}
            )

            top_title = (
                payload.get(
                    "title"
                )
                or payload.get(
                    "source_name"
                )
                or payload.get(
                    "source"
                )
                or ""
            )

            combined = payload_text(
                payload
            )

        else:

            top_score = 0.0
            top_title = ""
            combined = ""

        expected_any = case.get(
            "expected_any",
            [],
        )

        exact_match = (
            any(
                expected.lower()
                in combined.lower()
                for expected in expected_any
            )
            if expected_any
            else True
        )

        score_ok = (
            top_score
            >= threshold
        )

        passed = (
            bool(
                results
            )
            and score_ok
            and exact_match
        )

        rows.append(
            {
                "id":
                    case[
                        "id"
                    ],

                "language":
                    case[
                        "language"
                    ],

                "query":
                    case[
                        "query"
                    ],

                "normalized":
                    normalized_query,

                "top_title":
                    top_title,

                "top_score":
                    top_score,

                "latency":
                    elapsed,

                "score_ok":
                    score_ok,

                "expected_match":
                    exact_match,

                "pass":
                    passed,
            }
        )

        print(
            f"{case['id']}: "
            f"{'PASS' if passed else 'FAIL'} | "
            f"score={top_score:.4f} | "
            f"{elapsed:.3f}s | "
            f"{top_title}"
        )

    pass_count = sum(
        1
        for row in rows
        if row[
            "pass"
        ]
    )

    total = len(
        rows
    )

    avg_latency = (
        sum(
            row[
                "latency"
            ]
            for row in rows
        )
        / total
        if total
        else 0.0
    )

    avg_score = (
        sum(
            row[
                "top_score"
            ]
            for row in rows
        )
        / total
        if total
        else 0.0
    )

    lines = [
        "# Retrieval Benchmark Results",
        "",
        (
            "This report is generated from the current "
            "configured local RAG stack."
        ),
        "",
        f"- Cases: **{total}**",
        f"- Passed: **{pass_count}/{total}**",
        (
            "- Pass rate: "
            f"**{(pass_count / total * 100) if total else 0:.1f}%**"
        ),
        f"- Average top-1 score: **{avg_score:.4f}**",
        (
            "- Average retrieval latency: "
            f"**{avg_latency:.3f}s**"
        ),
        (
            "- Minimum score threshold: "
            f"**{threshold:.2f}**"
        ),
        "",
        (
            "| ID | Lang | Query | Normalized | Top source | "
            "Score | Latency | Expected entity | Result |"
        ),
        (
            "|---|---|---|---|---|---:|---:|---|---|"
        ),
    ]

    for row in rows:

        lines.append(
            (
                "| {id} | {language} | {query} | "
                "{normalized} | {top_title} | "
                "{top_score:.4f} | {latency:.3f}s | "
                "{entity} | {result} |"
            ).format(
                id=row[
                    "id"
                ],
                language=row[
                    "language"
                ],
                query=row[
                    "query"
                ].replace(
                    "|",
                    "\\|",
                ),
                normalized=str(
                    row[
                        "normalized"
                    ]
                ).replace(
                    "|",
                    "\\|",
                ),
                top_title=str(
                    row[
                        "top_title"
                    ]
                ).replace(
                    "|",
                    "\\|",
                ),
                top_score=row[
                    "top_score"
                ],
                latency=row[
                    "latency"
                ],
                entity=(
                    "PASS"
                    if row[
                        "expected_match"
                    ]
                    else "FAIL"
                ),
                result=(
                    "PASS"
                    if row[
                        "pass"
                    ]
                    else "FAIL"
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- `Score` is the top retrieved vector/reranked "
                "score returned by the current retrieval stack."
            ),
            (
                "- `Expected entity` checks whether the top "
                "retrieved payload contains the expected official "
                "product/entity name."
            ),
            (
                "- `Result` requires a non-empty result, "
                "score >= threshold, and expected entity match."
            ),
            (
                "- This benchmark evaluates retrieval only. "
                "ASR, generation, OCR, and TTS latency should "
                "be reported separately."
            ),
            "",
            "## Recommended submission metrics",
            "",
            (
                "Use this report together with:"
            ),
            "",
            (
                "- ASR processing time from the notebook."
            ),
            (
                "- LLM generation time from the Streamlit Details section."
            ),
            (
                "- OCR ingestion success examples."
            ),
            (
                "- Piper vs VoiceTut synthesis time."
            ),
            "",
            (
                "Do not present a metric as measured unless it was "
                "actually produced by the current run."
            ),
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
