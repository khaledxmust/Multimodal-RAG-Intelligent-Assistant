from config.hardware import choose_ctranslate2_device
import argparse
import re
import time
from pathlib import Path

from faster_whisper import WhisperModel


# ============================================================
# CONFIG
# ============================================================

MODEL_SIZE = "small"

CPU_DEVICE = "cpu"
CPU_COMPUTE_TYPE = "int8"

GPU_MIN_VRAM_GB = 1.5
GPU_COMPUTE_TYPE = "float16"

BEAM_SIZE = 5

# This assistant is intentionally bilingual.
SUPPORTED_AUTO_LANGUAGES = {
    "ar",
    "en",
}

DOMAIN_PROMPT = (
    "Telecom Egypt WE. "
    "باقات النت الأرضي، الإنترنت المنزلي، "
    "WE Space, WE Life, WE Sonic, "
    "eKYC, My WE, WE Pay."
)

HOTWORDS = (
    "WE Telecom Egypt "
    "WE Space WE Life WE Sonic "
    "eKYC My WE WE Pay "
    "باقات النت الأرضي الإنترنت المنزلي"
)


# ============================================================
# TRANSCRIPTION NORMALIZATION
# ============================================================

def normalize_telecom_transcription(
    text: str,
) -> str:
    """
    Apply conservative Telecom Egypt-specific corrections
    to common ASR confusions.

    This is intentionally NOT a general spelling corrector.
    We only fix patterns where the intended Telecom term
    is highly likely from the surrounding context.
    """

    if not text:
        return ""

    normalized = text.strip()

    # --------------------------------------------------------
    # Common mixed Arabic/English confusion:
    #
    # Spoken:
    #   باقات النت الأرضي من WE
    #
    # Whisper may produce:
    #   باقات النت الأرضي منوي
    # --------------------------------------------------------

    normalized = re.sub(
        r"("
        r"(?:باقات\s+)?"
        r"(?:النت|الإنترنت)"
        r"\s+"
        r"(?:الأرضي|الارضي|المنزلي)"
        r")"
        r"\s+منوي\b",
        r"\1 من WE",
        normalized,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Handle phrase without "النت".
    # --------------------------------------------------------

    normalized = re.sub(
        r"("
        r"الإنترنت"
        r"\s+"
        r"(?:المنزلي|الأرضي|الارضي)"
        r")"
        r"\s+منوي\b",
        r"\1 من WE",
        normalized,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Collapse duplicated Egyptian question filler.
    #
    #   إيه إيه خدمة WE Life
    #   -> إيه خدمة WE Life
    # --------------------------------------------------------

    normalized = re.sub(
        r"\b(?:إيه|ايه)\s+(?:إيه|ايه)\b",
        "إيه",
        normalized,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Common ASR confusion:
    #
    #   خدمت -> خدمة
    # --------------------------------------------------------

    normalized = re.sub(
        r"\bخدمت\b",
        "خدمة",
        normalized,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Normalize whitespace.
    # --------------------------------------------------------

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


# ============================================================
# ASR MODEL
# ============================================================

class LocalASR:

    def __init__(
        self,
        model_size: str = MODEL_SIZE,
    ):

        self.model_size = model_size

        hardware = choose_ctranslate2_device(
            min_vram_gb=GPU_MIN_VRAM_GB,
        )

        preferred_device = hardware[
            "device"
        ]

        preferred_compute_type = hardware[
            "compute_type"
        ]

        print(
            f"Loading Whisper model: {model_size}"
        )

        print(
            "Preferred device:",
            preferred_device,
        )

        print(
            "Preferred compute type:",
            preferred_compute_type,
        )

        print(
            "Hardware decision:",
            hardware["reason"],
        )

        try:

            self.model = WhisperModel(
                model_size,
                device=preferred_device,
                compute_type=preferred_compute_type,
            )

            self.device = preferred_device
            self.compute_type = preferred_compute_type

            print(
                "Whisper model ready."
            )

        except Exception as error:

            if preferred_device != "cuda":
                raise

            print()
            print(
                "Whisper GPU initialization failed:"
            )
            print(
                error
            )
            print(
                "Falling back to CPU int8..."
            )

            self.model = WhisperModel(
                model_size,
                device=CPU_DEVICE,
                compute_type=CPU_COMPUTE_TYPE,
            )

            self.device = CPU_DEVICE
            self.compute_type = CPU_COMPUTE_TYPE

            print(
                "Whisper CPU fallback ready."
            )

    def _start_transcription(
        self,
        audio_path: Path,
        language: str | None,
    ):

        return self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
            condition_on_previous_text=False,
            initial_prompt=DOMAIN_PROMPT,
            hotwords=HOTWORDS,
        )

    def _run_transcription(
        self,
        audio_path: Path,
        language: str | None,
    ):

        try:

            segments_generator, info = (
                self._start_transcription(
                    audio_path=audio_path,
                    language=language,
                )
            )

            # Force generator execution so CUDA runtime failures
            # are caught inside this method.
            segments = list(
                segments_generator
            )

            return segments, info

        except Exception as error:

            if self.device != "cuda":
                raise

            print()
            print(
                "Whisper GPU inference failed:"
            )
            print(
                error
            )
            print(
                "Retrying transcription on CPU int8..."
            )

            self.model = WhisperModel(
                self.model_size,
                device=CPU_DEVICE,
                compute_type=CPU_COMPUTE_TYPE,
            )

            self.device = CPU_DEVICE
            self.compute_type = CPU_COMPUTE_TYPE

            segments_generator, info = (
                self._start_transcription(
                    audio_path=audio_path,
                    language=language,
                )
            )

            segments = list(
                segments_generator
            )

            return segments, info

    @staticmethod
    def _segments_text(
        segments,
    ) -> str:

        return " ".join(
            (
                segment.text
                or ""
            ).strip()
            for segment in segments
            if (
                segment.text
                or ""
            ).strip()
        ).strip()

    @staticmethod
    def _candidate_score(
        segments,
        text: str,
        language: str,
    ) -> float:
        """
        Score a forced Arabic/English retry.

        avg_logprob is the primary signal. A small script-consistency
        bonus helps mixed Arabic/English Telecom phrases choose Arabic
        instead of an unrelated European language transliteration.
        """

        logprobs = [
            float(
                segment.avg_logprob
            )
            for segment in segments
            if getattr(
                segment,
                "avg_logprob",
                None,
            ) is not None
        ]

        if logprobs:

            score = (
                sum(
                    logprobs
                )
                / len(
                    logprobs
                )
            )

        else:

            score = -10.0

        arabic_chars = len(
            re.findall(
                r"[\u0600-\u06FF]",
                text,
            )
        )

        latin_chars = len(
            re.findall(
                r"[A-Za-z]",
                text,
            )
        )

        total_letters = (
            arabic_chars
            + latin_chars
        )

        if total_letters > 0:

            arabic_ratio = (
                arabic_chars
                / total_letters
            )

            latin_ratio = (
                latin_chars
                / total_letters
            )

            if language == "ar":

                score += (
                    0.20
                    * arabic_ratio
                )

            elif language == "en":

                score += (
                    0.12
                    * latin_ratio
                )

        return score

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> dict:

        audio_path = Path(
            audio_path
        ).resolve()

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found: "
                f"{audio_path}"
            )

        if (
            language is not None
            and language not in SUPPORTED_AUTO_LANGUAGES
        ):

            raise ValueError(
                "Supported ASR languages are "
                "'ar', 'en', or automatic detection."
            )

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Explicit user override: use it directly.
        # ----------------------------------------------------

        if language is not None:

            segments_generator, info = (
                self._run_transcription(
                    audio_path=audio_path,
                    language=language,
                )
            )

            selected_language = language

            language_probability = (
                float(
                    info.language_probability
                )
                if (
                    info
                    and info.language_probability
                    is not None
                )
                else None
            )

            auto_detected_language = None
            auto_language_probability = None
            language_selection = "forced"

        # ----------------------------------------------------
        # Automatic bilingual mode.
        # ----------------------------------------------------

        else:

            segments_generator, info = (
                self._run_transcription(
                    audio_path=audio_path,
                    language=None,
                )
            )

            auto_detected_language = (
                info.language
                if info
                else None
            )

            auto_language_probability = (
                float(
                    info.language_probability
                )
                if (
                    info
                    and info.language_probability
                    is not None
                )
                else None
            )

            # Keep Whisper's result when it is already Arabic/English.
            if (
                auto_detected_language
                in SUPPORTED_AUTO_LANGUAGES
            ):

                selected_language = (
                    auto_detected_language
                )

                language_probability = (
                    auto_language_probability
                )

                language_selection = "auto"

            # Whisper sometimes labels short Egyptian utterances as
            # French/German/etc. Restrict recovery to Arabic/English.
            else:

                auto_text = (
                    self._segments_text(
                        segments_generator
                    )
                )

                print()
                print(
                    "Unsupported auto-detected language:",
                    auto_detected_language,
                    (
                        f"({auto_language_probability:.4f})"
                        if auto_language_probability
                        is not None
                        else ""
                    ),
                )

                print(
                    "Restricting recovery to Arabic/English..."
                )

                arabic_chars_in_auto = len(
                    re.findall(
                        r"[\u0600-\u06FF]",
                        auto_text,
                    )
                )

                # If the first pass already emitted Arabic script,
                # retry Arabic only. Otherwise compare ar vs en.
                if arabic_chars_in_auto >= 3:

                    candidate_languages = [
                        "ar",
                    ]

                else:

                    candidate_languages = [
                        "ar",
                        "en",
                    ]

                candidates = []

                for candidate_language in (
                    candidate_languages
                ):

                    candidate_segments, candidate_info = (
                        self._run_transcription(
                            audio_path=audio_path,
                            language=candidate_language,
                        )
                    )

                    candidate_text = (
                        self._segments_text(
                            candidate_segments
                        )
                    )

                    candidate_score = (
                        self._candidate_score(
                            segments=candidate_segments,
                            text=candidate_text,
                            language=candidate_language,
                        )
                    )

                    print(
                        f"Candidate {candidate_language}: "
                        f"score={candidate_score:.4f} "
                        f"text={candidate_text}"
                    )

                    candidates.append(
                        {
                            "language":
                                candidate_language,

                            "segments":
                                candidate_segments,

                            "info":
                                candidate_info,

                            "score":
                                candidate_score,
                        }
                    )

                best_candidate = max(
                    candidates,
                    key=lambda item: item[
                        "score"
                    ],
                )

                selected_language = (
                    best_candidate[
                        "language"
                    ]
                )

                segments_generator = (
                    best_candidate[
                        "segments"
                    ]
                )

                info = (
                    best_candidate[
                        "info"
                    ]
                )

                # Do not present the original French/German
                # probability as if it were an Arabic/English
                # confidence value.
                language_probability = None

                language_selection = (
                    "restricted_ar_en_fallback"
                )

                print(
                    "Selected ASR language:",
                    selected_language,
                )

        # ----------------------------------------------------
        # Build segments + text.
        # ----------------------------------------------------

        segments = []
        text_parts = []

        for segment in segments_generator:

            text = (
                segment.text
                or ""
            ).strip()

            if not text:
                continue

            text_parts.append(
                text
            )

            segments.append(
                {
                    "start":
                        round(
                            segment.start,
                            2,
                        ),

                    "end":
                        round(
                            segment.end,
                            2,
                        ),

                    "text":
                        text,
                }
            )

        raw_transcription = (
            " ".join(
                text_parts
            )
            .strip()
        )

        transcription = (
            normalize_telecom_transcription(
                raw_transcription
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        duration = (
            float(
                info.duration
            )
            if (
                info
                and info.duration
                is not None
            )
            else None
        )

        return {
            "text":
                transcription,

            "raw_text":
                raw_transcription,

            "language":
                selected_language,

            "language_probability":
                language_probability,

            "language_selection":
                language_selection,

            "auto_detected_language":
                auto_detected_language,

            "auto_language_probability":
                auto_language_probability,

            "duration_seconds":
                duration,

            "processing_seconds":
                elapsed,

            "segments":
                segments,
        }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Local bilingual Arabic/English speech-to-text "
            "using Faster-Whisper."
        )
    )

    parser.add_argument(
        "audio",
        help=(
            "Path to an audio file "
            "(WAV, MP3, M4A, etc.)."
        ),
    )

    parser.add_argument(
        "--language",
        default=None,
        choices=[
            "ar",
            "en",
        ],
        help=(
            "Optional language override. "
            "Leave empty for restricted Arabic/English auto mode."
        ),
    )

    args = parser.parse_args()

    asr = LocalASR()

    result = asr.transcribe(
        audio_path=args.audio,
        language=args.language,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "TRANSCRIPTION RESULT"
    )
    print(
        "=" * 80
    )

    print(
        "Selected language:",
        result[
            "language"
        ],
    )

    print(
        "Language selection:",
        result[
            "language_selection"
        ],
    )

    if (
        result[
            "auto_detected_language"
        ]
        is not None
    ):

        print(
            "Initial auto detection:",
            result[
                "auto_detected_language"
            ],
        )

    if (
        result[
            "auto_language_probability"
        ]
        is not None
    ):

        print(
            "Initial auto probability:",
            f"{result['auto_language_probability']:.4f}",
        )

    if (
        result[
            "duration_seconds"
        ]
        is not None
    ):

        print(
            "Audio duration:",
            f"{result['duration_seconds']:.2f}s",
        )

    print(
        "Processing time:",
        f"{result['processing_seconds']:.2f}s",
    )

    print()
    print(
        "Raw Whisper text:"
    )
    print(
        result[
            "raw_text"
        ]
    )

    print()
    print(
        "Normalized text:"
    )
    print(
        result[
            "text"
        ]
    )

    if result[
        "segments"
    ]:

        print()
        print(
            "Segments:"
        )

        for segment in result[
            "segments"
        ]:

            print(
                f"[{segment['start']:.2f}s "
                f"- {segment['end']:.2f}s] "
                f"{segment['text']}"
            )


if __name__ == "__main__":
    main()
