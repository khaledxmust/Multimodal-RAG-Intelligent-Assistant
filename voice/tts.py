from pathlib import Path


# ============================================================
# TTS ENGINE NAMES
# ============================================================

TTS_NONE = "none"
TTS_PIPER = "piper"
TTS_VOICETUT = "voicetut"


TTS_ENGINES = {
    TTS_NONE:
        "No audio",

    TTS_PIPER:
        "Fast Arabic - Piper",

    TTS_VOICETUT:
        "Professional Egyptian - VoiceTut",
}


# ============================================================
# BASE INTERFACE
# ============================================================

class TTSEngine:

    engine_name = "base"

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
    ) -> dict:

        raise NotImplementedError


# ============================================================
# NO-AUDIO ENGINE
# ============================================================

class NoAudioTTS(TTSEngine):

    engine_name = TTS_NONE

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
    ) -> dict:

        return {
            "engine":
                self.engine_name,

            "text":
                text,

            "output_path":
                None,

            "processing_seconds":
                0.0,
        }


# ============================================================
# ENGINE FACTORY
# ============================================================

def get_tts_engine(
    engine: str,
    speaker: str | None = None,
) -> TTSEngine:

    engine = (
        engine
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # No audio
    # --------------------------------------------------------

    if engine == TTS_NONE:

        return NoAudioTTS()

    # --------------------------------------------------------
    # Piper
    # --------------------------------------------------------

    if engine == TTS_PIPER:

        from voice.tts_piper import LocalTTS

        return LocalTTS()

    # --------------------------------------------------------
    # VoiceTut
    # --------------------------------------------------------

    if engine == TTS_VOICETUT:

        from voice.tts_voicetut import (
            DEFAULT_SPEAKER,
            VoiceTutTTS,
        )

        return VoiceTutTTS(
            speaker=(
                speaker
                or DEFAULT_SPEAKER
            )
        )

    # --------------------------------------------------------
    # Invalid option
    # --------------------------------------------------------

    raise ValueError(
        f"Unknown TTS engine: {engine}. "
        f"Available engines: "
        f"{', '.join(TTS_ENGINES.keys())}"
    )


# ============================================================
# HIGH-LEVEL SYNTHESIS
# ============================================================

def synthesize_answer(
    text: str,
    engine: str = TTS_PIPER,
    speaker: str | None = None,
    output_path: str | Path = (
        "voice/output/assistant_response.wav"
    ),
) -> dict:

    tts = get_tts_engine(
        engine=engine,
        speaker=speaker,
    )

    result = tts.synthesize(
        text=text,
        output_path=output_path,
    )

    result.setdefault(
        "engine",
        engine,
    )

    return result