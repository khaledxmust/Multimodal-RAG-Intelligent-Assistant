import base64
import hashlib
import re
import sys
import tempfile
import time
from pathlib import Path

import requests
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.hardware import (
    choose_ctranslate2_device,
    get_nvidia_gpu_info,
    get_torch_cuda_info,
)


import generation.rag_assistant as rag_backend

from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import (
    get_point_count,
    get_qdrant_client,
)

from ingestion.user_documents import (
    SUPPORTED_EXTENSIONS,
    ingest_document,
)

# Speech recognition is loaded only when recording/transcription is used.

from voice.tts import (
    TTS_NONE,
    TTS_PIPER,
    TTS_VOICETUT,
    synthesize_answer,
)


# ============================================================
# CONFIG
# ============================================================

from config.settings import OLLAMA_URL, QDRANT_URL, VOICETUT_URL

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BACKGROUND_PATH = ASSETS_DIR / "we_ai_bg.jpg"
LOGO_PATH = ASSETS_DIR / "we_logo.png"

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
# PAGE
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG Intelligent Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# ASSET HELPERS
# ============================================================

def file_data_uri(path: Path, mime_type: str) -> str:
    if not path.exists():
        return ""

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


BACKGROUND_URI = file_data_uri(
    BACKGROUND_PATH,
    "image/jpeg",
)

LOGO_URI = file_data_uri(
    LOGO_PATH,
    "image/png",
)

def svg_icon(svg: str) -> str:
    encoded = base64.b64encode(
        svg.encode("utf-8")
    ).decode("ascii")

    return (
        "data:image/svg+xml;base64,"
        + encoded
    )


ICON_HOME = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="white" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 10.5 12 3l9 7.5"/>
<path d="M5.5 9.5V21h13V9.5"/>
<path d="M9.5 21v-6h5v6"/>
</svg>
""")

ICON_CHAT = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="white" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
<path d="M21 11.5a8.5 8.5 0 0 1-9 8.5 9.4 9.4 0 0 1-4-.9L3 21l1.7-4.6A8.4 8.4 0 1 1 21 11.5Z"/>
<path d="M8 11h.01M12 11h.01M16 11h.01"/>
</svg>
""")

ICON_DOC = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="white" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
<path d="M6 2h8l4 4v16H6z"/>
<path d="M14 2v5h5M9 12h6M9 16h6"/>
</svg>
""")

ICON_MIC = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="white" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
<rect x="9" y="3" width="6" height="12" rx="3"/>
<path d="M5 11a7 7 0 0 0 14 0M12 18v3M8.5 21h7"/>
</svg>
""")

ICON_GEAR = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="3"/>
<path d="M19.4 15a1.8 1.8 0 0 0 .36 2l.06.06-2.76 2.76-.06-.06a1.8 1.8 0 0 0-2-.36 1.8 1.8 0 0 0-1.1 1.66V21H10v-.09A1.8 1.8 0 0 0 8.9 19.25a1.8 1.8 0 0 0-2 .36l-.06.06-2.76-2.76.06-.06a1.8 1.8 0 0 0 .36-2A1.8 1.8 0 0 0 2.84 13.8H2v-3.9h.84A1.8 1.8 0 0 0 4.5 8.8a1.8 1.8 0 0 0-.36-2l-.06-.06L6.84 4l.06.06a1.8 1.8 0 0 0 2 .36A1.8 1.8 0 0 0 10 2.76V2h4v.76a1.8 1.8 0 0 0 1.1 1.66 1.8 1.8 0 0 0 2-.36l.06-.06 2.76 2.76-.06.06a1.8 1.8 0 0 0-.36 2 1.8 1.8 0 0 0 1.66 1.1H22v3.9h-.84A1.8 1.8 0 0 0 19.4 15Z"/>
</svg>
""")


# ============================================================
# SERVICE CHECKS
# ============================================================

def check_service(
    url: str,
    timeout: float = 1.5,
) -> bool:
    try:
        response = requests.get(
            url,
            timeout=timeout,
        )
        return response.ok
    except requests.RequestException:
        return False


def check_ollama() -> bool:
    return check_service(
        f"{OLLAMA_URL}/api/tags"
    )


def check_qdrant() -> bool:
    return check_service(
        f"{QDRANT_URL}/healthz"
    )


def check_voicetut() -> bool:
    return check_service(
        f"{VOICETUT_URL}/health"
    )


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "messages": [],
    "prompt_text": "",
    "pending_prompt": None,
    "pending_input_type": "text",
    "pending_voice_meta": None,
    "clear_prompt_on_rerun": False,
    "answer_mode": "Standard",
    "tts_engine": "None",
    "voicetut_speaker": "Hossam",
    "document_ingestion_results": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def submit_prompt():
    prompt = (
        st.session_state
        .get("prompt_text", "")
        .strip()
    )

    if not prompt:
        return

    st.session_state.pending_prompt = (
        prompt
    )

    st.session_state.pending_input_type = (
        "text"
    )

    st.session_state.pending_voice_meta = (
        None
    )


# ============================================================
# TTS HELPERS
# ============================================================

def get_selected_tts_engine() -> str:

    selected = (
        st.session_state
        .get(
            "tts_engine",
            "None",
        )
    )

    mapping = {
        "None":
            TTS_NONE,

        "Piper":
            TTS_PIPER,

        "VoiceTut":
            TTS_VOICETUT,
    }

    return mapping.get(
        selected,
        TTS_NONE,
    )


def prepare_tts_text(
    answer: str,
) -> str:
    """
    Convert the displayed Markdown answer into clean spoken text.

    The UI keeps citations and Markdown; TTS does not read citation
    numbers, asterisks, heading markers, or Markdown list markers.
    """

    spoken = (
        answer
        or ""
    )

    # Remove citation markers such as [1], [2].
    spoken = re.sub(
        r"\[\d+\]",
        "",
        spoken,
    )

    # Remove Markdown emphasis / heading characters.
    spoken = re.sub(
        r"[*_`#]+",
        "",
        spoken,
    )

    # Replace Markdown bullets with natural line breaks.
    spoken = re.sub(
        r"(?m)^\s*[-+]\s+",
        "",
        spoken,
    )

    # Replace ordered-list prefixes.
    spoken = re.sub(
        r"(?m)^\s*\d+[.)]\s+",
        "",
        spoken,
    )

    # Collapse excessive spacing while preserving sentence breaks.
    spoken = re.sub(
        r"[ \t]+",
        " ",
        spoken,
    )

    spoken = re.sub(
        r"\n{3,}",
        "\n\n",
        spoken,
    )

    return spoken.strip()


def synthesize_ui_answer(
    answer: str,
) -> dict | None:

    engine = (
        get_selected_tts_engine()
    )

    if engine == TTS_NONE:
        return None

    spoken_text = (
        prepare_tts_text(
            answer
        )
    )

    if not spoken_text:
        return None

    speaker = None

    if engine == TTS_VOICETUT:

        speaker = (
            st.session_state
            .get(
                "voicetut_speaker",
                "Hossam",
            )
        )

    identity = (
        f"{engine}|"
        f"{speaker or ''}|"
        f"{spoken_text}"
    )

    audio_id = (
        hashlib.sha1(
            identity.encode(
                "utf-8"
            )
        )
        .hexdigest()[:16]
    )

    output_path = (
        PROJECT_ROOT
        / "voice"
        / "output"
        / "ui"
        / f"{audio_id}.wav"
    )

    return synthesize_answer(
        text=spoken_text,
        engine=engine,
        speaker=speaker,
        output_path=output_path,
    )


def render_assistant_audio(
    tts_result: dict | None,
):

    if not tts_result:
        return

    output_path = (
        tts_result.get(
            "output_path"
        )
    )

    if not output_path:
        return

    audio_path = Path(
        output_path
    )

    if not audio_path.exists():
        return

    st.audio(
        audio_path.read_bytes(),
        format="audio/wav",
    )

    label = (
        "VoiceTut"
        if tts_result.get(
            "engine"
        ) == TTS_VOICETUT
        else "Piper"
    )

    if tts_result.get(
        "speaker"
    ):

        label += (
            " · "
            + str(
                tts_result[
                    "speaker"
                ]
            )
        )

    st.caption(
        label
    )


# ============================================================
# RAG RESOURCES
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def get_asr_model():
    from voice.asr import LocalASR

    return LocalASR()


def transcribe_uploaded_audio(
    recorded_audio,
) -> dict:
    """
    Persist Streamlit microphone audio temporarily and pass it
    through the existing LocalASR backend.
    """

    if recorded_audio is None:

        raise ValueError(
            "No audio was recorded."
        )

    audio_bytes = (
        recorded_audio
        .getvalue()
    )

    if not audio_bytes:

        raise ValueError(
            "The recorded audio is empty."
        )

    with tempfile.TemporaryDirectory(
        prefix="te_ai_voice_",
    ) as temp_dir:

        audio_path = (
            Path(
                temp_dir
            )
            / "microphone_input.wav"
        )

        audio_path.write_bytes(
            audio_bytes
        )

        asr = (
            get_asr_model()
        )

        return asr.transcribe(
            audio_path=audio_path,
            language=None,
        )


@st.cache_resource(
    show_spinner=False,
)
def get_rag_resources():

    client = (
        get_qdrant_client()
    )

    point_count = (
        get_point_count(
            client
        )
    )

    if point_count == 0:

        raise RuntimeError(
            "The Qdrant knowledge collection is empty."
        )

    embedding_model = (
        EmbeddingModel()
    )

    return (
        client,
        embedding_model,
        point_count,
    )


def clean_answer_for_display(
    answer: str,
) -> str:
    """
    Remove only redundant citation-only lines at the very end.

    Example:
        ... feature description [1].

        [1]

    becomes:
        ... feature description [1].
    """

    cleaned = (
        answer
        or ""
    ).strip()

    if not cleaned:
        return ""

    lines = (
        cleaned
        .splitlines()
    )

    while lines:

        last_line = (
            lines[-1]
            .strip()
        )

        if re.fullmatch(
            r"(?:\[\d+\]\s*)+",
            last_line,
        ):

            lines.pop()
            continue

        break

    return (
        "\n".join(
            lines
        )
        .strip()
    )


def get_display_sources(
    answer: str,
    sources: list[dict],
) -> list[dict]:
    """
    Return only sources cited in the answer.

    If the model omitted citations, use the top retrieved
    source as the same fallback used by the CLI.
    """

    citation_numbers = []

    if hasattr(
        rag_backend,
        "extract_citation_numbers",
    ):

        citation_numbers = (
            rag_backend
            .extract_citation_numbers(
                answer
            )
        )

    else:

        citation_numbers = [
            int(number)
            for number in re.findall(
                r"\[(\d+)\]",
                answer,
            )
        ]

    source_lookup = {
        source["number"]:
            source
        for source in sources
    }

    valid_numbers = [
        number
        for number in citation_numbers
        if number in source_lookup
    ]

    if (
        not valid_numbers
        and sources
    ):

        valid_numbers = [
            sources[0]["number"]
        ]

    return [
        source_lookup[number]
        for number in valid_numbers
    ]


def run_text_rag(
    question: str,
) -> dict:

    total_start = (
        time.perf_counter()
    )

    (
        client,
        embedding_model,
        point_count,
    ) = get_rag_resources()

    retrieval_start = (
        time.perf_counter()
    )

    (
        results,
        normalized_query,
    ) = rag_backend.retrieve(
        client=client,
        embedding_model=embedding_model,
        query=question,
    )

    retrieval_seconds = (
        time.perf_counter()
        - retrieval_start
    )

    if not results:

        return {
            "answer": (
                "I couldn't find relevant information "
                "in the available knowledge base."
            ),
            "sources": [],
            "normalized_query":
                normalized_query,
            "best_score":
                None,
            "retrieval_seconds":
                retrieval_seconds,
            "generation_seconds":
                0.0,
            "total_seconds":
                time.perf_counter()
                - total_start,
            "point_count":
                point_count,
        }

    best_score = float(
        results[0].score
    )

    context, sources = (
        rag_backend.build_context(
            results
        )
    )

    min_score = float(
        getattr(
            rag_backend,
            "MIN_RETRIEVAL_SCORE",
            0.80,
        )
    )

    if best_score < min_score:

        query_language = (
            rag_backend
            .detect_query_language(
                question
            )
        )

        if query_language == "ar":

            answer = (
                "لم أجد معلومات موثوقة كافية "
                "في المصادر المتاحة للإجابة "
                "على هذا السؤال."
            )

        else:

            answer = (
                "I couldn't find sufficiently relevant "
                "information in the available sources "
                "to answer this question reliably."
            )

        generation_seconds = 0.0

    else:

        generation_start = (
            time.perf_counter()
        )

        answer = (
            rag_backend.generate_answer(
                question=question,
                context=context,
            )
        )

        answer = (
            clean_answer_for_display(
                answer
            )
        )

        generation_seconds = (
            time.perf_counter()
            - generation_start
        )

    display_sources = (
        get_display_sources(
            answer=answer,
            sources=sources,
        )
    )

    return {
        "answer":
            answer,

        "sources":
            display_sources,

        "normalized_query":
            normalized_query,

        "best_score":
            best_score,

        "retrieval_seconds":
            retrieval_seconds,

        "generation_seconds":
            generation_seconds,

        "total_seconds":
            time.perf_counter()
            - total_start,

        "point_count":
            point_count,
    }


def render_role_label(
    role: str,
):

    if role == "user":

        css_class = (
            "chat-role-user"
        )

        label = "You"

    else:

        css_class = (
            "chat-role-ai"
        )

        label = "RAG Assistant"

    st.html(
        f"""
        <div class="chat-role {css_class}">
            {label}
        </div>
        """
    )


def render_sources(
    sources: list[dict],
):

    if not sources:
        return

    with st.expander(
        f"Sources · {len(sources)}",
        expanded=False,
    ):

        for source in sources:

            number = source[
                "number"
            ]

            title = (
                source.get(
                    "title",
                    "",
                )
                or "Source"
            )

            section = (
                source.get(
                    "section",
                    "",
                )
                or ""
            )

            url = (
                source.get(
                    "url",
                    "",
                )
                or ""
            )

            score = float(
                source.get(
                    "score",
                    0.0,
                )
            )

            st.markdown(
                f"**[{number}] {title}**"
            )

            if section:

                st.caption(
                    section
                )

            if url:

                st.markdown(
                    f"[Open source]({url})"
                )

            st.caption(
                f"Relevance: {score:.4f}"
            )


# ============================================================
# USER DOCUMENT INGESTION
# ============================================================

def ingest_uploaded_files(
    uploaded_files,
) -> list[dict]:
    """
    Save Streamlit uploads temporarily, then reuse the existing
    production ingestion pipeline with the cached Qdrant client
    and embedding model.
    """

    if not uploaded_files:
        return []

    (
        client,
        embedding_model,
        _,
    ) = get_rag_resources()

    results = []

    with tempfile.TemporaryDirectory(
        prefix="te_ai_upload_",
    ) as temp_dir:

        temp_dir_path = Path(
            temp_dir
        )

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):

            safe_name = (
                Path(
                    uploaded_file.name
                )
                .name
            )

            extension = (
                Path(
                    safe_name
                )
                .suffix
                .lower()
            )

            if extension not in SUPPORTED_EXTENSIONS:

                results.append(
                    {
                        "source_name":
                            safe_name,

                        "success":
                            False,

                        "error":
                            "Unsupported file type.",
                    }
                )

                continue

            temp_path = (
                temp_dir_path
                / f"{index}_{safe_name}"
            )

            temp_path.write_bytes(
                uploaded_file.getvalue()
            )

            try:

                result = (
                    ingest_document(
                        file_path=temp_path,
                        display_name=safe_name,
                        client=client,
                        embedding_model=embedding_model,
                    )
                )

                results.append(
                    {
                        **result,
                        "success":
                            True,
                    }
                )

            except Exception as error:

                results.append(
                    {
                        "source_name":
                            safe_name,

                        "success":
                            False,

                        "error":
                            str(
                                error
                            ),
                    }
                )

    return results


# ============================================================
# STATUS
# ============================================================

qdrant_ok = check_qdrant()
ollama_ok = check_ollama()
voicetut_ok = check_voicetut()

gpu = get_nvidia_gpu_info()
torch_cuda = get_torch_cuda_info()

asr_device = choose_ctranslate2_device(
    min_vram_gb=1.5,
)


# ============================================================
# CSS
# ============================================================

st.html(
    f"""
    <style>
        :root {{
            --bg: #050718;
            --panel: rgba(11, 13, 37, 0.72);
            --panel-strong: rgba(10, 12, 32, 0.90);
            --line: rgba(180, 143, 255, 0.22);
            --line-strong: rgba(164, 94, 255, 0.70);
            --purple: #8b5cf6;
            --violet: #a855f7;
            --pink: #e44cff;
            --cyan: #66d9ff;
            --green: #2ef2a0;
            --red: #ff5f77;
            --amber: #f4c96b;
            --text: #f7f7fb;
            --muted: #a6adc2;
        }}

        html, body, [class*="css"] {{
            font-family:
                "Segoe UI Variable Display",
                "Segoe UI Variable",
                "Aptos",
                "Inter",
                sans-serif;
        }}

        .stApp {{
            background:
                linear-gradient(
                    180deg,
                    rgba(4, 6, 18, 1.00) 0%,
                    rgba(4, 6, 18, 1.00) 34%,
                    rgba(4, 6, 18, 0.97) 45%,
                    rgba(4, 6, 18, 0.48) 67%,
                    rgba(4, 6, 18, 0.28) 100%
                ),
                url("{BACKGROUND_URI}");
            background-size: cover;
            background-position: center 74%;
            background-attachment: fixed;
            color: var(--text);
        }}

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        footer {{
            display: none !important;
        }}

        .block-container {{
            max-width: 1180px;
            padding-top: 3rem;
            padding-bottom: 2rem;
            padding-left: 5.8rem;
            padding-right: 1.8rem;
        }}

        /* ----------------------------------------------------
           FIXED COLLAPSED ICON RAIL
        ---------------------------------------------------- */

        .we-rail {{
            position: fixed;
            z-index: 999;
            top: 0;
            left: 0;
            width: 74px;
            height: 100vh;

            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;

            background:
                linear-gradient(
                    180deg,
                    rgba(6, 8, 26, 0.94),
                    rgba(5, 6, 19, 0.86)
                );

            border-right:
                1px solid rgba(180, 143, 255, 0.14);

            backdrop-filter: blur(18px);
        }}

        .rail-icons {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 20px;
        }}

        .rail-icon {{
            position: relative;
            width: 42px;
            height: 42px;

            display: flex;
            align-items: center;
            justify-content: center;

            color: #cfd3df;

            border:
                1px solid transparent;

            border-radius: 13px;

            transition:
                transform 160ms ease,
                color 160ms ease,
                background 160ms ease,
                box-shadow 160ms ease,
                border-color 160ms ease;
        }}

        .rail-icon:hover {{
            color: white;
            transform: translateY(-1px);
            background: rgba(126, 64, 255, 0.16);
            border-color: rgba(164, 94, 255, 0.32);
        }}

        .rail-icon.active {{
            color: white;
            background:
                linear-gradient(
                    145deg,
                    rgba(97, 48, 255, 0.78),
                    rgba(159, 55, 255, 0.62)
                );

            border-color:
                rgba(196, 143, 255, 0.64);

            box-shadow:
                0 0 22px rgba(126, 67, 255, 0.38);
        }}

        .rail-icon img {{
            width: 21px;
            height: 21px;
            display: block;
            opacity: 0.94;
        }}

        .rail-tip {{
            position: absolute;
            left: 52px;

            padding: 6px 9px;

            border:
                1px solid rgba(255,255,255,0.08);

            border-radius: 8px;

            background:
                rgba(8,10,25,0.94);

            color: #f4f4f7;

            font-size: 12px;
            white-space: nowrap;

            opacity: 0;
            pointer-events: none;

            transform: translateX(-4px);

            transition:
                opacity 150ms ease,
                transform 150ms ease;
        }}

        .rail-icon:hover .rail-tip {{
            opacity: 1;
            transform: translateX(0);
        }}

        /* ----------------------------------------------------
           BRAND
        ---------------------------------------------------- */

        .brand-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;

            min-height: 132px;

            margin-top: 0.2rem;
        }}

        .brand-logo {{
            width: 108px;
            max-height: 118px;
            object-fit: contain;
            filter:
                drop-shadow(
                    0 8px 18px rgba(126, 67, 255, 0.18)
                );
        }}

        .brand-divider {{
            width: 1px;
            height: 72px;
            margin: 0 22px;

            background:
                linear-gradient(
                    180deg,
                    transparent,
                    rgba(196, 184, 255, 0.60),
                    transparent
                );
        }}

        .brand-name {{
            font-size: 1.55rem;
            font-weight: 430;
            letter-spacing: -0.025em;
            color: #f3f3f8;
        }}

        /* ----------------------------------------------------
           HERO
        ---------------------------------------------------- */

        .hero {{
            text-align: center;
            margin-top: 2.2rem;
        }}

        .hero-title {{
            margin: 0;

            font-family:
                "Segoe UI Variable Display",
                "Aptos Display",
                "Inter",
                sans-serif;

            font-size:
                clamp(2.25rem, 4.2vw, 3.8rem);

            line-height: 1.04;
            font-weight: 670;
            letter-spacing: -0.055em;

            color: #fbfbfe;
        }}

        .hero-gradient {{
            background:
                linear-gradient(
                    92deg,
                    #dfc0ff 0%,
                    #b56dff 46%,
                    #8c4dff 100%
                );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .hero-copy {{
            max-width: 700px;
            margin: 1rem auto 0;

            color: #aeb5ca;
            font-size: 0.98rem;
            line-height: 1.7;
            letter-spacing: 0.01em;
        }}

        /* ----------------------------------------------------
           CHAT BAR
        ---------------------------------------------------- */

        .st-key-prompt_shell {{
            margin-top: 4.0rem;
            margin-left: auto;
            margin-right: auto;

            max-width: 820px;

            padding:
                8px 10px;

            border:
                1px solid rgba(163, 94, 255, 0.62);

            border-radius:
                22px;

            background:
                linear-gradient(
                    100deg,
                    rgba(17, 20, 49, 0.82),
                    rgba(11, 14, 38, 0.76)
                );

            box-shadow:
                0 0 0 1px rgba(131, 80, 255, 0.10),
                0 0 30px rgba(114, 54, 255, 0.12);

            backdrop-filter:
                blur(20px);
        }}

        .st-key-prompt_shell
        [data-testid="stHorizontalBlock"] {{
            gap: 0.45rem;
            align-items: center;
        }}

        .st-key-prompt_shell
        div[data-testid="stTextInput"] {{
            margin: 0;
        }}

        .st-key-prompt_shell
        div[data-testid="stTextInput"] input {{
            height: 42px;

            background: transparent !important;

            border: none !important;
            outline: none !important;

            color: #f7f7fb !important;

            font-size: 0.98rem;

            box-shadow: none !important;
        }}

        .st-key-prompt_shell
        div[data-testid="stTextInput"] input::placeholder {{
            color: #8d95ac !important;
        }}

        /* Keep the prompt focus state purple, never Streamlit red. */
        .st-key-prompt_shell
        div[data-testid="stTextInput"]
        div[data-baseweb="input"] {{
            border:
                1px solid transparent !important;

            box-shadow:
                none !important;

            background:
                rgba(255,255,255,0.035)
                !important;

            border-radius:
                12px !important;
        }}

        .st-key-prompt_shell
        div[data-testid="stTextInput"]
        div[data-baseweb="input"]:focus-within {{
            border-color:
                rgba(177, 104, 255, 0.92)
                !important;

            box-shadow:
                0 0 0 1px
                rgba(177, 104, 255, 0.28),
                0 0 16px
                rgba(129, 69, 255, 0.14)
                !important;
        }}

        .st-key-prompt_shell
        div[data-testid="stTextInput"]
        input:focus,
        .st-key-prompt_shell
        div[data-testid="stTextInput"]
        input:focus-visible,
        .st-key-prompt_shell
        div[data-testid="stTextInput"]
        input[aria-invalid="true"] {{
            border:
                none !important;

            outline:
                none !important;

            box-shadow:
                none !important;
        }}

        .st-key-prompt_shell
        button {{
            min-height: 40px;

            color: #ececf5;

            background:
                rgba(255,255,255,0.035);

            border:
                1px solid rgba(255,255,255,0.06);

            border-radius: 13px;

            box-shadow: none;

            transition:
                background 150ms ease,
                border-color 150ms ease,
                transform 150ms ease;
        }}

        .st-key-prompt_shell
        [data-testid="stPopover"] button svg:last-child {{
            display: none !important;
        }}

        .st-key-prompt_shell
        [data-testid="stPopover"] button {{
            min-width: 40px !important;
            width: 40px !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }}

        .st-key-prompt_shell
        [data-testid="stPopover"] button p {{
            width: 100%;
            text-align: center;
            margin: 0;
        }}

        .st-key-prompt_shell
        button:hover {{
            color: white;

            background:
                rgba(137, 80, 255, 0.16);

            border-color:
                rgba(168, 99, 255, 0.34);

            transform:
                translateY(-1px);
        }}

        .st-key-prompt_shell
        div[data-baseweb="select"] > div {{
            min-height: 40px;

            color: #dfe2eb;

            background: transparent;

            border: none;

            box-shadow: none;
        }}

        /* Send button */
        .st-key-send_button button {{
            width: 44px;
            height: 44px;

            padding: 0;

            font-size: 1.15rem;

            border-radius: 50%;

            border:
                1px solid rgba(209, 179, 255, 0.68);

            background:
                linear-gradient(
                    145deg,
                    #7248ff,
                    #8c4dff
                );

            box-shadow:
                0 0 18px rgba(125, 70, 255, 0.36);
        }}

        /* ----------------------------------------------------
           SECONDARY INFO BELOW BAR
        ---------------------------------------------------- */

        .mini-info {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;

            min-height: 24px;

            margin-top: 10px;

            color: #858da4;
            font-size: 0.78rem;
        }}

        .mini-sep {{
            width: 3px;
            height: 3px;
            border-radius: 50%;
            background: rgba(197, 179, 255, 0.55);
        }}

        /* ----------------------------------------------------
           SERVICE DOTS
        ---------------------------------------------------- */

        .service-row {{
            position: fixed;
            z-index: 500;

            left: 50%;
            bottom: 32px;

            transform: translateX(-50%);

            display: flex;
            align-items: center;
            justify-content: center;

            gap: 24px;
        }}

        .service-item {{
            position: relative;

            width: 12px;
            height: 12px;

            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .service-dot {{
            width: 9px;
            height: 9px;

            border-radius: 50%;

            transition:
                width 180ms ease,
                height 180ms ease,
                transform 180ms ease,
                box-shadow 180ms ease;
        }}

        .service-dot.online {{
            background: var(--green);

            box-shadow:
                0 0 12px rgba(46, 242, 160, 0.72);
        }}

        .service-dot.offline {{
            background: var(--red);

            box-shadow:
                0 0 9px rgba(255, 95, 119, 0.50);
        }}

        .service-dot.optional {{
            background: var(--amber);

            box-shadow:
                0 0 9px rgba(244, 201, 107, 0.50);
        }}

        .service-item:hover
        .service-dot {{
            width: 14px;
            height: 14px;

            transform:
                translateY(-1px);

            box-shadow:
                0 0 18px currentColor;
        }}

        .service-card {{
            position: absolute;

            left: 50%;
            bottom: 26px;

            transform:
                translate(-50%, 8px)
                scale(0.96);

            min-width: 148px;

            display: flex;
            align-items: center;

            gap: 10px;

            padding:
                10px 12px;

            border:
                1px solid rgba(188, 164, 255, 0.20);

            border-radius:
                12px;

            background:
                rgba(8, 11, 30, 0.94);

            box-shadow:
                0 16px 40px rgba(0, 0, 0, 0.35);

            backdrop-filter:
                blur(18px);

            opacity: 0;
            visibility: hidden;
            pointer-events: none;

            transition:
                opacity 160ms ease,
                transform 160ms ease,
                visibility 160ms ease;
        }}

        .service-item:hover
        .service-card {{
            opacity: 1;
            visibility: visible;

            transform:
                translate(-50%, 0)
                scale(1);
        }}

        .service-icon {{
            width: 31px;
            height: 31px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 9px;

            color: #e5d8ff;

            background:
                rgba(135, 77, 255, 0.14);

            border:
                1px solid rgba(168, 101, 255, 0.22);

            font-size: 15px;
            font-weight: 700;
        }}

        .service-name {{
            color: #f2f2f7;
            font-size: 0.79rem;
            font-weight: 620;
            line-height: 1.15;
        }}

        .service-status {{
            margin-top: 3px;

            color: #9da5ba;
            font-size: 0.69rem;
        }}

        .service-status .ok {{
            color: var(--green);
        }}

        .service-status .bad {{
            color: var(--red);
        }}

        /* ----------------------------------------------------
           CHAT HISTORY
        ---------------------------------------------------- */

        .st-key-chat_history {{
            max-width: 880px;
            margin: 1.15rem auto 5rem;
        }}

        [data-testid="stChatMessage"] {{
            border:
                1px solid rgba(255,255,255,0.055);

            border-radius: 16px;

            background:
                rgba(7, 10, 28, 0.48);

            backdrop-filter: blur(12px);
        }}

        /* Minimal chat: remove Streamlit's default avatar circles. */
        [data-testid="stChatMessage"]
        [data-testid*="stChatMessageAvatar"] {{
            display: none !important;
        }}

        [data-testid="stChatMessage"] {{
            padding:
                1rem 1.15rem !important;
        }}

        [data-testid="stChatMessage"]
        [data-testid="stMarkdownContainer"] {{
            width: 100%;
            unicode-bidi: plaintext;
            text-align: start;
        }}

        .chat-role {{
            margin-bottom: 0.48rem;

            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .chat-role-user {{
            color: #b7bdcf;
        }}

        .chat-role-ai {{
            color: #bb83ff;
        }}

        /* Cleaner structured answer typography. */
        [data-testid="stChatMessage"]
        ul,
        [data-testid="stChatMessage"]
        ol {{
            margin-top: 0.55rem;
            margin-bottom: 0.55rem;
            padding-inline-start: 1.35rem;
        }}

        [data-testid="stChatMessage"]
        li {{
            margin-bottom: 0.42rem;
            line-height: 1.65;
        }}

        [data-testid="stChatMessage"]
        p {{
            line-height: 1.7;
        }}

        [data-testid="stChatMessage"]
        strong {{
            color: #f3eaff;
        }}

        /* ----------------------------------------------------
           POPOVERS
        ---------------------------------------------------- */

        [data-testid="stPopoverBody"] {{
            background:
                rgba(8, 10, 27, 0.96);

            border:
                1px solid rgba(170, 116, 255, 0.18);

            border-radius: 14px;
        }}

        /* ----------------------------------------------------
           RESPONSIVE
        ---------------------------------------------------- */

        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 5.2rem;
                padding-right: 1rem;
            }}

            .brand-wrap {{
                transform: scale(0.9);
            }}

            .hero-title {{
                font-size: 2.35rem;
            }}

            .st-key-prompt_shell {{
                max-width: 100%;
            }}
        }}
    </style>
    """
)


# ============================================================
# LEFT ICON RAIL
# ============================================================

st.html(
    f"""
    <div class="we-rail">
        <div class="rail-icons">

            <div class="rail-icon active">
                <img src="{ICON_HOME}" alt="Home">
                <span class="rail-tip">Home</span>
            </div>

            <div class="rail-icon">
                <img src="{ICON_CHAT}" alt="Chat">
                <span class="rail-tip">Chat</span>
            </div>

            <div class="rail-icon">
                <img src="{ICON_DOC}" alt="Documents">
                <span class="rail-tip">Documents</span>
            </div>

            <div class="rail-icon">
                <img src="{ICON_MIC}" alt="Voice">
                <span class="rail-tip">Voice</span>
            </div>

            <div class="rail-icon">
                <img src="{ICON_GEAR}" alt="Settings">
                <span class="rail-tip">Settings</span>
            </div>

        </div>
    </div>
    """
)


# ============================================================
# BRAND LOCKUP
# ============================================================

if LOGO_URI:
    logo_html = (
        f'<img class="brand-logo" '
        f'src="{LOGO_URI}" '
        f'alt="WE">'
    )
else:
    logo_html = (
        '<div style="font-size:2rem;font-weight:700;">WE</div>'
    )

st.html(
    f"""
    <div class="brand-wrap">
        {logo_html}

        <div class="brand-divider"></div>

        <div class="brand-name">
            AI Assistant
        </div>
    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <h1 class="hero-title">
            Your local
            <span class="hero-gradient">
                AI workspace
            </span>
        </h1>

        <div class="hero-copy">
            Ask in Arabic, English, or Egyptian dialect.<br>
            Search knowledge, analyze documents,
            and get things done locally.
        </div>
    </div>
    """
)


# ============================================================
# FUNCTIONAL PROMPT BAR
# ============================================================

with st.container(
    key="prompt_shell",
):

    add_col, text_col, mode_col, mic_col, tts_col, send_col = (
        st.columns(
            [0.65, 7.0, 1.45, 0.7, 0.7, 0.72],
            vertical_alignment="center",
        )
    )

    with add_col:

        with st.popover(
            "＋",
            use_container_width=True,
        ):

            st.caption(
                "Add knowledge"
            )

            uploaded_files = (
                st.file_uploader(
                    "Upload files",
                    type=[
                        "pdf",
                        "docx",
                        "txt",
                        "png",
                        "jpg",
                        "jpeg",
                    ],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key="ui_uploads",
                )
            )

            if uploaded_files:

                st.caption(
                    f"{len(uploaded_files)} file(s) selected"
                )

            index_documents = (
                st.button(
                    "Index selected files",
                    disabled=(
                        not uploaded_files
                    ),
                    use_container_width=True,
                    key="index_uploaded_documents",
                )
            )

            if index_documents:

                with st.spinner(
                    "Adding documents to the knowledge base..."
                ):

                    ingestion_results = (
                        ingest_uploaded_files(
                            uploaded_files
                        )
                    )

                st.session_state[
                    "document_ingestion_results"
                ] = ingestion_results

            for result in (
                st.session_state[
                    "document_ingestion_results"
                ]
            ):

                if result.get(
                    "success"
                ):

                    st.success(
                        (
                            f"{result['source_name']} · "
                            f"{result['chunks_indexed']} chunks · "
                            f"{result['qdrant_points']} total points"
                        ),
                        icon="✅",
                    )

                else:

                    st.error(
                        (
                            f"{result.get('source_name', 'File')}: "
                            f"{result.get('error', 'Ingestion failed.')}"
                        )
                    )

    with text_col:

        if st.session_state.get(
            "clear_prompt_on_rerun",
            False,
        ):

            st.session_state.prompt_text = ""

            st.session_state.clear_prompt_on_rerun = (
                False
            )

        st.text_input(
            "Message",
            key="prompt_text",
            placeholder="Ask me anything...",
            label_visibility="collapsed",
        )

    with mode_col:

        st.selectbox(
            "Mode",
            options=[
                "Standard",
                "Concise",
                "Detailed",
            ],
            key="answer_mode",
            label_visibility="collapsed",
        )

    with mic_col:

        with st.popover(
            "🎙",
            use_container_width=True,
        ):

            st.caption(
                "Voice input"
            )

            recorded_audio = (
                st.audio_input(
                    "Record",
                    label_visibility="collapsed",
                    key="ui_audio_input",
                )
            )

            if recorded_audio:

                transcribe_and_ask = (
                    st.button(
                        "Transcribe & ask",
                        use_container_width=True,
                        key="transcribe_voice_question",
                    )
                )

                if transcribe_and_ask:

                    try:

                        with st.spinner(
                            "Transcribing locally..."
                        ):

                            voice_result = (
                                transcribe_uploaded_audio(
                                    recorded_audio
                                )
                            )

                        transcription = (
                            voice_result
                            .get(
                                "text",
                                "",
                            )
                            .strip()
                        )

                        if not transcription:

                            raise RuntimeError(
                                "Whisper did not detect usable speech."
                            )

                        st.session_state.pending_prompt = (
                            transcription
                        )

                        st.session_state.pending_input_type = (
                            "voice"
                        )

                        st.session_state.pending_voice_meta = (
                            voice_result
                        )

                        st.rerun()

                    except Exception as error:

                        st.error(
                            "Voice transcription failed: "
                            f"{error}"
                        )

    with tts_col:

        with st.popover(
            "🔊",
            use_container_width=True,
        ):

            st.caption(
                "Text-to-speech"
            )

            st.selectbox(
                "TTS engine",
                options=[
                    "None",
                    "Piper",
                    "VoiceTut",
                ],
                key="tts_engine",
            )

            if (
                st.session_state.tts_engine
                == "VoiceTut"
            ):

                st.selectbox(
                    "Voice",
                    options=VOICETUT_SPEAKERS,
                    key="voicetut_speaker",
                )

                if not voicetut_ok:
                    st.warning(
                        "VoiceTut service is offline."
                    )

    with send_col:

        with st.container(
            key="send_button",
        ):

            send_clicked = (
                st.button(
                    "↑",
                    key="send_prompt",
                    use_container_width=True,
                )
            )

            # Submission is intentionally explicit.
            #
            # Do not attach submit_prompt to text_input.on_change:
            # Streamlit can fire on_change when the input loses focus.
            # That caused a pasted draft to be sent when the user
            # opened or changed Mode / TTS / Voice controls.
            if send_clicked:

                submit_prompt()

                if st.session_state.get(
                    "pending_prompt"
                ):

                    st.session_state.clear_prompt_on_rerun = (
                        True
                    )

                    st.rerun()


# ============================================================
# SMALL ACTIVE SETTINGS LINE
# ============================================================

tts_label = (
    "Text only"
    if st.session_state.tts_engine == "None"
    else st.session_state.tts_engine
)

st.html(
    f"""
    <div class="mini-info">
        <span>{st.session_state.answer_mode}</span>
        <span class="mini-sep"></span>
        <span>{tts_label}</span>
        <span class="mini-sep"></span>
        <span>ASR {asr_device['device'].upper()}</span>
    </div>
    """
)


# ============================================================
# CHAT HISTORY
# ============================================================

with st.container(
    key="chat_history",
):

    # --------------------------------------------------------
    # Existing conversation
    # --------------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            render_role_label(
                message["role"]
            )

            if (
                message["role"]
                == "user"
                and message.get(
                    "input_type"
                )
                == "voice"
            ):

                voice_meta = (
                    message.get(
                        "voice_meta"
                    )
                    or {}
                )

                language = (
                    voice_meta.get(
                        "language"
                    )
                    or "auto"
                )

                probability = (
                    voice_meta.get(
                        "language_probability"
                    )
                )

                voice_label = (
                    f"Voice transcription · {language}"
                )

                if probability is not None:

                    voice_label += (
                        f" · {probability:.2f}"
                    )

                st.caption(
                    voice_label
                )

            st.markdown(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
            ):

                render_sources(
                    message.get(
                        "sources",
                        [],
                    )
                )

                render_assistant_audio(
                    message.get(
                        "tts_result"
                    )
                )

                tts_error = (
                    message.get(
                        "tts_error"
                    )
                )

                if tts_error:

                    st.caption(
                        "Audio unavailable: "
                        + tts_error
                    )

                timing = (
                    message.get(
                        "timing"
                    )
                )

                if timing:

                    with st.expander(
                        "Details",
                        expanded=False,
                    ):

                        timing_parts = []

                        if timing.get(
                            "asr"
                        ) is not None:

                            timing_parts.append(
                                "ASR "
                                f"{timing['asr']:.2f}s"
                            )

                        if timing.get(
                            "tts"
                        ) is not None:

                            timing_parts.append(
                                "TTS "
                                f"{timing['tts']:.2f}s"
                            )

                        timing_parts.extend(
                            [
                                "Retrieval "
                                f"{timing['retrieval']:.2f}s",

                                "Generation "
                                f"{timing['generation']:.2f}s",

                                "Total "
                                f"{timing['total']:.2f}s",
                            ]
                        )

                        st.caption(
                            " · ".join(
                                timing_parts
                            )
                        )

    # --------------------------------------------------------
    # New prompt waiting for RAG
    # --------------------------------------------------------

    pending_prompt = (
        st.session_state
        .get(
            "pending_prompt"
        )
    )

    pending_input_type = (
        st.session_state
        .get(
            "pending_input_type",
            "text",
        )
    )

    pending_voice_meta = (
        st.session_state
        .get(
            "pending_voice_meta"
        )
    )

    if pending_prompt:

        st.session_state.messages.append(
            {
                "role":
                    "user",

                "content":
                    pending_prompt,

                "input_type":
                    pending_input_type,

                "voice_meta":
                    pending_voice_meta,
            }
        )

        with st.chat_message(
            "user"
        ):

            render_role_label(
                "user"
            )

            if (
                pending_input_type
                == "voice"
            ):

                language = (
                    (
                        pending_voice_meta
                        or {}
                    )
                    .get(
                        "language"
                    )
                    or "auto"
                )

                probability = (
                    (
                        pending_voice_meta
                        or {}
                    )
                    .get(
                        "language_probability"
                    )
                )

                voice_label = (
                    f"Voice transcription · {language}"
                )

                if probability is not None:

                    voice_label += (
                        f" · {probability:.2f}"
                    )

                st.caption(
                    voice_label
                )

            st.markdown(
                pending_prompt
            )

        with st.chat_message(
            "assistant"
        ):

            render_role_label(
                "assistant"
            )

            try:

                with st.spinner(
                    "Searching the local knowledge base..."
                ):

                    rag_result = (
                        run_text_rag(
                            pending_prompt
                        )
                    )

                st.markdown(
                    rag_result[
                        "answer"
                    ]
                )

                render_sources(
                    rag_result[
                        "sources"
                    ]
                )

                tts_result = None
                tts_error = None

                if (
                    get_selected_tts_engine()
                    != TTS_NONE
                ):

                    try:

                        with st.spinner(
                            "Generating audio locally..."
                        ):

                            tts_result = (
                                synthesize_ui_answer(
                                    rag_result[
                                        "answer"
                                    ]
                                )
                            )

                        render_assistant_audio(
                            tts_result
                        )

                    except Exception as error:

                        tts_error = str(
                            error
                        )

                        st.warning(
                            "Text answer is ready, "
                            "but audio generation failed. "
                            f"{tts_error}"
                        )

                with st.expander(
                    "Details",
                    expanded=False,
                ):

                    if (
                        rag_result[
                            "normalized_query"
                        ]
                        != pending_prompt
                    ):

                        st.caption(
                            "Normalized query: "
                            + rag_result[
                                "normalized_query"
                            ]
                        )

                    current_asr_seconds = (
                        (
                            pending_voice_meta
                            or {}
                        )
                        .get(
                            "processing_seconds"
                        )
                    )

                    current_tts_seconds = (
                        (
                            tts_result
                            or {}
                        )
                        .get(
                            "processing_seconds"
                        )
                    )

                    current_total_seconds = (
                        rag_result[
                            "total_seconds"
                        ]
                    )

                    timing_parts = []

                    if current_asr_seconds is not None:

                        current_total_seconds += (
                            current_asr_seconds
                        )

                        timing_parts.append(
                            "ASR "
                            f"{current_asr_seconds:.2f}s"
                        )

                    if current_tts_seconds is not None:

                        current_total_seconds += (
                            current_tts_seconds
                        )

                        timing_parts.append(
                            "TTS "
                            f"{current_tts_seconds:.2f}s"
                        )

                    timing_parts.extend(
                        [
                            "Retrieval "
                            f"{rag_result['retrieval_seconds']:.2f}s",

                            "Generation "
                            f"{rag_result['generation_seconds']:.2f}s",

                            "Total "
                            f"{current_total_seconds:.2f}s",
                        ]
                    )

                    st.caption(
                        " · ".join(
                            timing_parts
                        )
                    )

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            rag_result[
                                "answer"
                            ],

                        "sources":
                            rag_result[
                                "sources"
                            ],

                        "tts_result":
                            tts_result,

                        "tts_error":
                            tts_error,

                        "timing": {
                            "asr":
                                (
                                    (
                                        pending_voice_meta
                                        or {}
                                    )
                                    .get(
                                        "processing_seconds"
                                    )
                                ),

                            "tts":
                                (
                                    (
                                        tts_result
                                        or {}
                                    )
                                    .get(
                                        "processing_seconds"
                                    )
                                ),

                            "retrieval":
                                rag_result[
                                    "retrieval_seconds"
                                ],

                            "generation":
                                rag_result[
                                    "generation_seconds"
                                ],

                            "total":
                                (
                                    rag_result[
                                        "total_seconds"
                                    ]
                                    +
                                    (
                                        (
                                            pending_voice_meta
                                            or {}
                                        )
                                        .get(
                                            "processing_seconds"
                                        )
                                        or 0.0
                                    )
                                    +
                                    (
                                        (
                                            tts_result
                                            or {}
                                        )
                                        .get(
                                            "processing_seconds"
                                        )
                                        or 0.0
                                    )
                                ),
                        },
                    }
                )

            except Exception as error:

                error_message = (
                    "The assistant could not complete "
                    "the request. "
                    f"{error}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            error_message,

                        "sources":
                            [],
                    }
                )

        st.session_state.pending_prompt = (
            None
        )

        st.session_state.pending_input_type = (
            "text"
        )

        st.session_state.pending_voice_meta = (
            None
        )


# ============================================================
# SERVICE DOTS + HOVER CARDS
# ============================================================

services = [
    {
        "name": "Qdrant",
        "icon": "Q",
        "online": qdrant_ok,
        "offline_label": "Offline",
        "online_label": "Running",
        "optional": False,
    },
    {
        "name": "Ollama",
        "icon": "O",
        "online": ollama_ok,
        "offline_label": "Offline",
        "online_label": "Online",
        "optional": False,
    },
    {
        "name": "VoiceTut",
        "icon": "V",
        "online": voicetut_ok,
        "offline_label": "Optional / Offline",
        "online_label": "Online",
        "optional": True,
    },
]

service_html = []

for service in services:

    if service["online"]:
        dot_class = "online"
        status_class = "ok"
        status_text = service["online_label"]

    elif service["optional"]:
        dot_class = "optional"
        status_class = ""
        status_text = service["offline_label"]

    else:
        dot_class = "offline"
        status_class = "bad"
        status_text = service["offline_label"]

    service_html.append(
        f"""
        <div class="service-item">

            <div class="service-card">

                <div class="service-icon">
                    {service["icon"]}
                </div>

                <div>
                    <div class="service-name">
                        {service["name"]}
                    </div>

                    <div class="service-status">
                        <span class="{status_class}">
                            ●
                        </span>
                        {status_text}
                    </div>
                </div>

            </div>

            <div class="service-dot {dot_class}">
            </div>

        </div>
        """
    )

st.html(
    """
    <div class="service-row">
    """
    + "".join(service_html)
    +
    """
    </div>
    """
)
