import argparse
import re
import time
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

# ============================================================
# CONFIG
# ============================================================

VOICE_DIR = (
    Path(__file__).resolve().parent
    / "models"
)

from config.settings import PIPER_MODEL

ARABIC_MODEL = PIPER_MODEL

SPEECH_LENGTH_SCALE = 0.70

# ============================================================
# TEXT CLEANING
# ============================================================


def normalize_text_for_tts(
    text: str,
) -> str:
    """
    Convert product names and mixed Arabic/English terms
    into speech-friendly forms.

    This affects only spoken audio.
    The displayed RAG answer remains unchanged.
    """

    replacements = [
        # Longer expressions first
        (
            r"\bWE\s+SPACE\b",
            "وي اسبيس",
        ),
        (
            r"\bWE\s+LIFE\b",
            "وي لايف",
        ),
        (
            r"\bWE\s+SONIC\b",
            "وي سونيك",
        ),
        (
            r"\bWE\s+Pay\b",
            "وي باي",
        ),
        (
            r"\bMy\s+WE\b",
            "ماي وي",
        ),
        (
            r"\beKYC\b",
            "إي كي واي سي",
        ),

        # Generic WE must come last
        (
            r"\bWE\b",
            "وي",
        ),
    ]

    normalized = text

    for pattern, replacement in replacements:

        normalized = re.sub(
            pattern,
            replacement,
            normalized,
            flags=re.IGNORECASE,
        )

    return normalized


def clean_text_for_speech(
    text: str,
) -> str:
    """
    Remove citation markers and formatting that should
    not be spoken by TTS.
    """

    text = re.sub(
        r"\[\d+\]",
        "",
        text,
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# TTS
# ============================================================

class LocalTTS:

    def __init__(
        self,
        model_path: str | Path = ARABIC_MODEL,
    ):

        self.model_path = Path(
            model_path
        ).resolve()

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"TTS model not found: "
                f"{self.model_path}"
            )

        print(
            "Loading Piper voice:"
        )

        print(
            self.model_path.name
        )

        start = time.perf_counter()

        self.voice = PiperVoice.load(
            str(
                self.model_path
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Piper voice ready "
            f"in {elapsed:.2f}s."
        )

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
    ) -> dict:

        clean_text = clean_text_for_speech(
            text
        )

        speech_text = normalize_text_for_tts(
            clean_text
        )

        if not clean_text:

            raise ValueError(
                "No text available for speech."
            )

        output_path = Path(
            output_path
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        start = time.perf_counter()

        with wave.open(
            str(
                output_path
            ),
            "wb",
        ) as wav_file:

            synthesis_config = SynthesisConfig(
                length_scale=SPEECH_LENGTH_SCALE,
            )

            self.voice.synthesize_wav(
                speech_text,
                wav_file,
                syn_config=synthesis_config,
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        return {
            "text":
                speech_text,

            "output_path":
                str(
                    output_path
                ),

            "processing_seconds":
                elapsed,
        }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Local Arabic text-to-speech "
            "using Piper."
        )
    )

    parser.add_argument(
        "text",
        help="Text to synthesize.",
    )

    parser.add_argument(
        "--output",
        default=(
            "voice/output/"
            "tts_test.wav"
        ),
        help="Output WAV path.",
    )

    args = parser.parse_args()

    tts = LocalTTS()

    result = tts.synthesize(
        text=args.text,
        output_path=args.output,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "TTS RESULT"
    )

    print(
        "=" * 80
    )

    print(
        "Text:",
        result[
            "text"
        ],
    )

    print(
        "Output:",
        result[
            "output_path"
        ],
    )

    print(
        "Processing time:",
        f"{result['processing_seconds']:.2f}s",
    )


if __name__ == "__main__":
    main()