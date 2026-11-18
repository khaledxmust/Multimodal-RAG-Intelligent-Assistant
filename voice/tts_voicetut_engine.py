import argparse
import gc
import re
import time
from pathlib import Path

from config.hardware import choose_torch_device


# ============================================================
# CONFIG
# ============================================================

VOICE_DIR = Path(
    __file__
).resolve().parent

MODEL_DIR = (
    VOICE_DIR
    / "models"
    / "voicetut"
)

OUTPUT_DIR = (
    VOICE_DIR
    / "output"
)

DEFAULT_SPEAKER = "Mohamed"

DEFAULT_NUM_STEP = 8

DEFAULT_SPEED = 1.0

# VoiceTut is much larger than the embedding/ASR models.
# Do not attempt CUDA unless the GPU has at least this much VRAM.
VOICETUT_MIN_VRAM_GB = 4.0


# Built-in VoiceTut voices
VOICETUT_SPEAKERS = [
    # Male
    "Abdelrahman",
    "Abdullah",
    "Kamal",
    "Hossam",
    "Mohamed",
    "Omar",
    "Sayed",
    "Zaki",
    "Aly",
    "Essam",
    "Ahmed",

    # Female
    "Asmaa",
    "Esraa",
    "Hanan",
    "Sarah",
    "Yasmin",
    "Omnia",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text_for_speech(
    text: str,
) -> str:
    """
    Prepare the RAG answer for speech.

    The displayed answer remains unchanged.
    Only the spoken version is cleaned.
    """

    if not text:
        return ""

    cleaned = text.strip()

    # Remove citations:
    # [1], [2], [1][2], etc.
    cleaned = re.sub(
        r"\[\d+\]",
        "",
        cleaned,
    )

    # Remove URLs
    cleaned = re.sub(
        r"https?://\S+",
        "",
        cleaned,
    )

    # Normalize whitespace
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    return cleaned


# ============================================================
# VOICETUT ENGINE
# ============================================================

class LocalVoiceTutTTS:

    engine_name = "voicetut"

    def __init__(
        self,
        speaker: str = DEFAULT_SPEAKER,
        device: str | None = None,
        dtype: str | None = None,
    ):

        # ----------------------------------------------------
        # Validate speaker
        # ----------------------------------------------------

        if speaker not in VOICETUT_SPEAKERS:

            raise ValueError(
                f"Unknown VoiceTut speaker: "
                f"{speaker}"
            )

        # ----------------------------------------------------
        # Validate model directory
        # ----------------------------------------------------

        if not MODEL_DIR.exists():

            raise FileNotFoundError(
                f"VoiceTut model directory "
                f"not found: {MODEL_DIR}"
            )

        self.speaker = speaker

        # ----------------------------------------------------
        # AUTO HARDWARE SELECTION
        # ----------------------------------------------------

        if device is None:

            hardware = choose_torch_device(
                min_vram_gb=VOICETUT_MIN_VRAM_GB,
            )

            selected_device = hardware[
                "device"
            ]

            self.hardware_decision = (
                hardware["reason"]
            )

            print(
                "VoiceTut hardware decision:"
            )

            print(
                self.hardware_decision
            )

        else:

            selected_device = (
                device
                .strip()
                .lower()
            )

            if selected_device not in {
                "cpu",
                "cuda",
            }:

                raise ValueError(
                    "VoiceTut device must be "
                    "'cpu' or 'cuda'."
                )

            self.hardware_decision = (
                "Device manually selected."
            )

            print(
                "VoiceTut device manually selected:",
                selected_device,
            )

        # ----------------------------------------------------
        # DTYPE
        # ----------------------------------------------------

        if dtype is None:

            selected_dtype = (
                "float16"
                if selected_device == "cuda"
                else "float32"
            )

        else:

            selected_dtype = dtype

        self.device = selected_device
        self.dtype = selected_dtype

        # ----------------------------------------------------
        # MODEL INFO
        # ----------------------------------------------------

        print(
            "Loading VoiceTut-TTS..."
        )

        print(
            "Model:",
            MODEL_DIR,
        )

        print(
            "Device:",
            self.device,
        )

        print(
            "Dtype:",
            self.dtype,
        )

        print(
            "Speaker:",
            speaker,
        )

        start = time.perf_counter()

        # Import only when this backend
        # is actually used.
        from voicetut_tts import (
            VoiceTutTTS as VoiceTutModel
        )

        self._model_class = (
            VoiceTutModel
        )

        # ----------------------------------------------------
        # INITIAL MODEL LOAD
        # ----------------------------------------------------

        try:

            self.model = (
                self._load_model(
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        except Exception as error:

            if self.device != "cuda":
                raise

            print()

            print(
                "VoiceTut GPU initialization failed:"
            )

            print(
                error
            )

            print(
                "Falling back to CPU float32..."
            )

            self._clear_cuda_cache()

            self.device = "cpu"
            self.dtype = "float32"

            self.model = (
                self._load_model(
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"VoiceTut-TTS ready "
            f"in {elapsed:.2f}s."
        )

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(
        self,
        device: str,
        dtype: str,
    ):

        return (
            self._model_class
            .from_pretrained(
                str(MODEL_DIR),
                device=device,
                dtype=dtype,
            )
        )

    # ========================================================
    # CUDA CLEANUP
    # ========================================================

    @staticmethod
    def _clear_cuda_cache():

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception:
            pass

    # ========================================================
    # CPU FALLBACK
    # ========================================================

    def _fallback_to_cpu(
        self,
    ):

        print()

        print(
            "Reloading VoiceTut on CPU..."
        )

        old_model = self.model

        self.model = None

        del old_model

        self._clear_cuda_cache()

        self.device = "cpu"
        self.dtype = "float32"

        self.model = (
            self._load_model(
                device=self.device,
                dtype=self.dtype,
            )
        )

        print(
            "VoiceTut CPU fallback ready."
        )

    # ========================================================
    # SYNTHESIS
    # ========================================================

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        speaker: str | None = None,
        num_step: int = DEFAULT_NUM_STEP,
        speed: float = DEFAULT_SPEED,
    ) -> dict:

        speech_text = (
            clean_text_for_speech(
                text
            )
        )

        if not speech_text:

            raise ValueError(
                "No text available "
                "for speech synthesis."
            )

        selected_speaker = (
            speaker
            or self.speaker
        )

        if (
            selected_speaker
            not in VOICETUT_SPEAKERS
        ):

            raise ValueError(
                f"Unknown VoiceTut speaker: "
                f"{selected_speaker}"
            )

        output_path = Path(
            output_path
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()

        print(
            "VoiceTut synthesis:"
        )

        print(
            "Device:",
            self.device,
        )

        print(
            "Dtype:",
            self.dtype,
        )

        print(
            "Speaker:",
            selected_speaker,
        )

        print(
            "Text:",
            speech_text,
        )

        start = time.perf_counter()

        # ----------------------------------------------------
        # SYNTHESIS WITH RUNTIME FALLBACK
        # ----------------------------------------------------

        try:

            self.model.synthesize(
                speech_text,

                speaker=
                    selected_speaker,

                num_step=
                    num_step,

                speed=
                    speed,

                output=
                    str(output_path),
            )

        except Exception as error:

            if self.device != "cuda":
                raise

            print()

            print(
                "VoiceTut GPU synthesis failed:"
            )

            print(
                error
            )

            print(
                "Retrying on CPU..."
            )

            self._fallback_to_cpu()

            self.model.synthesize(
                speech_text,

                speaker=
                    selected_speaker,

                num_step=
                    num_step,

                speed=
                    speed,

                output=
                    str(output_path),
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        return {
            "engine":
                self.engine_name,

            "speaker":
                selected_speaker,

            "device":
                self.device,

            "dtype":
                self.dtype,

            "text":
                speech_text,

            "output_path":
                str(output_path),

            "processing_seconds":
                elapsed,
        }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Local Egyptian VoiceTut-TTS test."
        )
    )

    parser.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER,
        choices=VOICETUT_SPEAKERS,
        help=(
            "Built-in VoiceTut speaker."
        ),
    )

    parser.add_argument(
        "--text",
        default=(
            "أهلاً بحضرتك في وي، "
            "باقات الإنترنت المنزلي عندنا "
            "تشمل وي سبيس ووي لايف "
            "ووي سونيك."
        ),
        help=(
            "Text to synthesize."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            OUTPUT_DIR
            / "voicetut_test.wav"
        ),
        help=(
            "Output WAV path."
        ),
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=DEFAULT_NUM_STEP,
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
    )

    args = parser.parse_args()

    tts = LocalVoiceTutTTS(
        speaker=args.speaker,
    )

    result = tts.synthesize(
        text=args.text,
        output_path=args.output,
        num_step=args.num_step,
        speed=args.speed,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "VOICETUT RESULT"
    )

    print(
        "=" * 80
    )

    print(
        "Speaker:",
        result["speaker"],
    )

    print(
        "Device:",
        result["device"],
    )

    print(
        "Dtype:",
        result["dtype"],
    )

    print(
        "Output:",
        result["output_path"],
    )

    print(
        "Synthesis time:",
        f"{result['processing_seconds']:.2f}s",
    )


if __name__ == "__main__":
    main()