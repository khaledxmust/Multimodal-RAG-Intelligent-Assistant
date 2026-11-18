import argparse
import time
from pathlib import Path

from generation.rag_assistant import (
    MIN_RETRIEVAL_SCORE,
    OLLAMA_MODEL,
    build_context,
    build_low_confidence_response,
    check_ollama,
    generate_answer,
    print_sources,
    retrieve,
)

from retrieval.embeddings import EmbeddingModel

from retrieval.vector_store import (
    get_point_count,
    get_qdrant_client,
)

from voice.asr import LocalASR

from voice.tts import (
    TTS_ENGINES,
    TTS_NONE,
    synthesize_answer,
)


# ============================================================
# VOICE RAG
# ============================================================

def run_voice_rag(
    audio_path: str | Path,
    tts_engine: str = TTS_NONE,
    tts_speaker: str | None = None,
):

    total_start = time.perf_counter()

    print(
        "=" * 80
    )

    print(
        "TE VOICE RAG ASSISTANT"
    )

    print(
        "=" * 80
    )

    print(
        "TTS engine:",
        TTS_ENGINES.get(
            tts_engine,
            tts_engine,
        ),
    )

    if (
            tts_engine == "voicetut"
            and tts_speaker
    ):
        print(
            "VoiceTut speaker:",
            tts_speaker,
        )

    # ========================================================
    # SERVICES
    # ========================================================

    client = get_qdrant_client()

    point_count = get_point_count(
        client
    )

    if point_count == 0:

        raise RuntimeError(
            "Qdrant collection is empty."
        )

    print(
        "Qdrant points:",
        point_count,
    )

    check_ollama()

    print(
        "Ollama: connected"
    )

    print(
        "Ollama model:",
        OLLAMA_MODEL,
    )

    # ========================================================
    # ASR
    # ========================================================

    print()

    print(
        "Loading ASR model..."
    )

    asr = LocalASR()

    print()

    print(
        "Transcribing audio..."
    )

    asr_start = time.perf_counter()

    transcription = asr.transcribe(
        audio_path=audio_path,
    )

    asr_time = (
        time.perf_counter()
        - asr_start
    )

    question = (
        transcription.get(
            "text",
            "",
        )
        .strip()
    )

    if not question:

        raise RuntimeError(
            "No speech could be transcribed "
            "from the audio."
        )

    print()

    print(
        "Detected language:",
        transcription.get(
            "language"
        ),
    )

    probability = transcription.get(
        "language_probability"
    )

    if probability is not None:

        print(
            "Language probability:",
            f"{probability:.4f}",
        )

    print()

    print(
        "Transcription:"
    )

    print(
        question
    )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    print()

    print(
        "Loading embedding model..."
    )

    embedding_model = EmbeddingModel()

    print(
        "Embedding model ready."
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieval_start = (
        time.perf_counter()
    )

    (
        results,
        normalized_query,
    ) = retrieve(
        client=client,
        embedding_model=embedding_model,
        query=question,
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    if normalized_query != question:

        print()

        print(
            "Normalized query:",
            normalized_query,
        )

    context, sources = build_context(
        results
    )

    best_score = (
        sources[0]["score"]
        if sources
        else 0.0
    )

    # ========================================================
    # GENERATION
    # ========================================================

    generation_time = 0.0

    if (
        not sources
        or best_score
        < MIN_RETRIEVAL_SCORE
    ):

        answer = (
            build_low_confidence_response(
                question
            )
        )

        generation_skipped = True

    else:

        generation_skipped = False

        print()

        print(
            "Generating answer..."
        )

        generation_start = (
            time.perf_counter()
        )

        answer = generate_answer(
            question=question,
            context=context,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

    # ========================================================
    # TTS
    # ========================================================

    tts_time = 0.0

    audio_output_path = None

    if tts_engine != TTS_NONE:

        print()

        print(
            "Generating spoken response..."
        )

        tts_start = (
            time.perf_counter()
        )

        tts_result = synthesize_answer(
            text=answer,
            engine=tts_engine,
            speaker=tts_speaker,
            output_path=(
                "voice/output/"
                "assistant_response.wav"
            ),
        )

        tts_time = (
            time.perf_counter()
            - tts_start
        )

        audio_output_path = (
            tts_result.get(
                "output_path"
            )
        )

    # ========================================================
    # RESULT
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "VOICE RAG RESULT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Assistant:"
    )

    print(
        answer
    )

    print_sources(
        answer=answer,
        sources=sources,
    )

    if audio_output_path:

        print()

        print(
            "Audio response:"
        )

        print(
            audio_output_path
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    print()

    print(
        "Timing:"
    )

    print(
        f"    ASR: "
        f"{asr_time:.2f}s"
    )

    print(
        f"    Retrieval: "
        f"{retrieval_time:.2f}s"
    )

    if generation_skipped:

        print(
            "    Generation: "
            "0.00s (skipped)"
        )

    else:

        print(
            f"    Generation: "
            f"{generation_time:.2f}s"
        )

    if tts_engine == TTS_NONE:

        print(
            "    TTS: "
            "0.00s (disabled)"
        )

    else:

        print(
            f"    TTS: "
            f"{tts_time:.2f}s"
        )

    print(
        f"    Total: "
        f"{total_time:.2f}s"
    )

    return {
        "transcription":
            question,

        "detected_language":
            transcription.get(
                "language"
            ),

        "language_probability":
            probability,

        "normalized_query":
            normalized_query,

        "answer":
            answer,

        "sources":
            sources,

        "tts_engine":
            tts_engine,

        "audio_output_path":
            audio_output_path,

        "asr_seconds":
            asr_time,

        "retrieval_seconds":
            retrieval_time,

        "generation_seconds":
            generation_time,

        "tts_seconds":
            tts_time,

        "total_seconds":
            total_time,
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Telecom Egypt local "
            "voice-to-RAG assistant."
        )
    )

    parser.add_argument(
        "audio",
        help=(
            "Path to WAV, MP3, M4A "
            "or another supported audio file."
        ),
    )

    parser.add_argument(
        "--tts",
        default=TTS_NONE,
        choices=list(
            TTS_ENGINES.keys()
        ),
        help=(
            "TTS engine for spoken response. "
            "Options: none, piper, voicetut."
        ),
    )

    parser.add_argument(
        "--speaker",
        default=None,
        help=(
            "VoiceTut speaker name. "
            "Example: Hossam, Mohamed, Sayed, Asmaa."
        ),
    )

    args = parser.parse_args()

    run_voice_rag(
        audio_path=args.audio,
        tts_engine=args.tts,
        tts_speaker=args.speaker,
    )


if __name__ == "__main__":
    main()