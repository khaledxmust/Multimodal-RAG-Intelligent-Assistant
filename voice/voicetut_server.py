import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from voice.tts_voicetut_engine import (
    DEFAULT_NUM_STEP,
    DEFAULT_SPEAKER,
    VOICETUT_SPEAKERS,
    LocalVoiceTutTTS,
)


# ============================================================
# CONFIG
# ============================================================

VOICE_DIR = Path(
    __file__
).resolve().parent

OUTPUT_DIR = (
    VOICE_DIR
    / "output"
    / "voicetut"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="VoiceTut Local TTS Service",
    version="1.0",
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING VOICETUT SERVICE")
print("=" * 80)
print()

tts = LocalVoiceTutTTS(
    speaker=DEFAULT_SPEAKER,
)

print()
print("VoiceTut service ready.")
print()


# ============================================================
# REQUEST MODEL
# ============================================================

class SynthesisRequest(BaseModel):

    text: str

    speaker: str = DEFAULT_SPEAKER

    num_step: int = DEFAULT_NUM_STEP

    speed: float = 1.0


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "ok",

        "engine": "voicetut",

        "default_speaker":
            DEFAULT_SPEAKER,

        "default_num_step":
            DEFAULT_NUM_STEP,

        "speakers":
            VOICETUT_SPEAKERS,
    }


# ============================================================
# SYNTHESIS
# ============================================================

@app.post(
    "/synthesize"
)
def synthesize(
    request: SynthesisRequest,
):

    if not request.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    if (
        request.speaker
        not in VOICETUT_SPEAKERS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown speaker: "
                f"{request.speaker}"
            ),
        )

    timestamp = int(
        time.time() * 1000
    )

    output_path = (
        OUTPUT_DIR
        / f"response_{timestamp}.wav"
    )

    try:

        result = tts.synthesize(
            text=request.text,
            output_path=output_path,
            speaker=request.speaker,
            num_step=request.num_step,
            speed=request.speed,
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return FileResponse(
        path=result[
            "output_path"
        ],
        media_type="audio/wav",
        filename=output_path.name,
        headers={
            "X-TTS-Engine":
                "voicetut",

            "X-TTS-Speaker":
                request.speaker,

            "X-TTS-Seconds":
                str(
                    result[
                        "processing_seconds"
                    ]
                ),
        },
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
    )