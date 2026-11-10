"""Classificador e coletor de metadados de arquivos de mídia (áudio e vídeo)."""
from pathlib import Path

import cv2

from . import config


def classify(path: Path) -> str:
    """Diz se o arquivo é 'audio' ou 'video' pela extensão."""
    if not path.is_file():
        raise ValueError(f"Arquivo não encontrado: {path}")
    ext = path.suffix.lower()
    if ext in config.AUDIO_EXTENSIONS:
        size_mb = path.stat().st_size / 2**20
        if size_mb > config.TRANSCRIBE_MAX_MB:
            raise ValueError(f"O áudio tem {size_mb:.1f} MB e a transcrição aceita no máximo {config.TRANSCRIBE_MAX_MB} MB.")
        return "audio"
    if ext in config.VIDEO_EXTENSIONS:
        return "video"
    accepted = sorted(config.AUDIO_EXTENSIONS | config.VIDEO_EXTENSIONS)
    raise ValueError(
        f"Apenas arquivos de áudio ou vídeo são aceitos ({', '.join(accepted)}). Recebido: {ext or 'sem extensão'}"
    )


def video_metadata(path: Path) -> dict:
    """Taxa de quadros, quantidade de frames, resolução e duração, lidos com o OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {
        "fps": round(fps, 3),
        "frames": frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_s": round(frames / fps, 2) if fps else 0.0,
    }
    cap.release()
    return meta


def audio_metadata(path: Path) -> dict:
    """Para o áudio só registramos o formato e o tamanho."""
    return {"format": path.suffix.lower().lstrip("."), "size_mb": round(path.stat().st_size / 2**20, 2)}
