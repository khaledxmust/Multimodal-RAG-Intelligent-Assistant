import time
from pathlib import Path

import requests


# ============================================================
# CONFIG
# ============================================================

from config.settings import VOICETUT_URL

DEFAULT_SPEAKER = "Mohamed"

DEFAULT_NUM_STEP = 8

DEFAULT_SPEED = 1.0


VOICETUT_SPEAKERS = [
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
    "Asmaa",
    "Esraa",
    "Hanan",
    "Sarah",
    "Yasmin",
    "Omnia",
]


# ============================================================
# VOICETUT CLIENT
# ============================================================

class VoiceTutTTS:

    engine_name = "voicetut"

    def __init__(
        self,
        speaker: str = DEFAULT_SPEAKER,
        base_url: str = VOICETUT_URL,
        num_step: int = DEFAULT_NUM_STEP,
        speed: float = DEFAULT_SPEED,
    ):

        if speaker not in VOICETUT_SPEAKERS:

            raise ValueError(
                f"Unknown VoiceTut speaker: "
                f"{speaker}"
            )

        self.speaker = speaker

        self.base_url = (
            base_url.rstrip("/")
        )

        self.num_step = num_step

        self.speed = speed

    # ========================================================
    # HEALTH
    # ========================================================

    def health(
        self,
    ) -> dict:

        try:

            response = requests.get(
                f"{self.base_url}/health",
                timeout=10,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:

            raise RuntimeError(
                "VoiceTut service is not available. "
                "Start it with:\n"
                "python -m voice.voicetut_server"
            ) from error

    # ========================================================
    # SYNTHESIS
    # ========================================================

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
    ) -> dict:

        if not text.strip():

            raise ValueError(
                "Text cannot be empty."
            )

        output_path = Path(
            output_path
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "text":
                text,

            "speaker":
                self.speaker,

            "num_step":
                self.num_step,

            "speed":
                self.speed,
        }

        start = time.perf_counter()

        try:

            response = requests.post(
                f"{self.base_url}/synthesize",
                json=payload,
                timeout=300,
            )

        except requests.RequestException as error:

            raise RuntimeError(
                "Could not connect to "
                "VoiceTut service."
            ) from error

        request_seconds = (
            time.perf_counter()
            - start
        )

        if not response.ok:

            raise RuntimeError(
                "VoiceTut synthesis failed.\n"
                f"Status: {response.status_code}\n"
                f"Response: {response.text}"
            )

        output_path.write_bytes(
            response.content
        )

        server_seconds = (
            response.headers.get(
                "X-TTS-Seconds"
            )
        )

        try:

            processing_seconds = (
                float(server_seconds)
                if server_seconds
                else request_seconds
            )

        except ValueError:

            processing_seconds = (
                request_seconds
            )

        return {
            "engine":
                self.engine_name,

            "speaker":
                self.speaker,

            "text":
                text,

            "output_path":
                str(output_path),

            "processing_seconds":
                processing_seconds,

            "request_seconds":
                request_seconds,
        }