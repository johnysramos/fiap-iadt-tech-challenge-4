"""Configuração do projeto: chave e modelos da OpenAI (lidos do .env), caminhos e os números fixos do pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini")

MODELS_DIR = ROOT / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
OUTPUTS_DIR = ROOT / "outputs"

# Formatos aceitos
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Tamanho máximo do áudio
TRANSCRIBE_MAX_MB = 25

# Frames por segundo para análise de vídeo (quanto menor, mais rápido, mas menos preciso)
ANALYSIS_FPS = 10
YOLO_CONFIDENCE = 0.4

# Faixas de severidade do risco
SEVERITY_ATTENTION = 30
SEVERITY_CRITICAL = 70
