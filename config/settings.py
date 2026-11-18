"""Portable settings shared by the application and command line entry points."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')
DATA_DIR = PROJECT_ROOT / 'ingestion' / 'data'

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://127.0.0.1:6333').rstrip('/')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION', 'multimodal_rag_demo')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')
HF_LOCAL_FILES_ONLY = os.getenv('HF_LOCAL_FILES_ONLY', 'false').lower() in {'1', 'true', 'yes'}
VOICETUT_URL = os.getenv('VOICETUT_URL', 'http://127.0.0.1:8001').rstrip('/')
PIPER_MODEL = Path(os.getenv('PIPER_MODEL', str(PROJECT_ROOT / 'voice' / 'models' / 'ar_JO-kareem-medium.onnx'))).expanduser()
if not PIPER_MODEL.is_absolute():
    PIPER_MODEL = PROJECT_ROOT / PIPER_MODEL
OCR_LANGUAGES = os.getenv('OCR_LANGUAGES', 'ara+eng')
TESSERACT_CMD = os.getenv('TESSERACT_CMD', '')
