"""Classe de interação com a OpenAI"""
import base64
import json
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from . import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------- transcrição
def transcribe(audio_path: Path) -> str:
    """Manda o arquivo de áudio como está e recebe o texto falado."""
    with open(audio_path, "rb") as fh:
        result = _client.audio.transcriptions.create(
            model=config.OPENAI_TRANSCRIBE_MODEL,
            file=fh,
            response_format="json",
        )
    return result.text.strip()


# ------------------------------------------------------- análise da transcrição
class TranscriptAnalysis(BaseModel):
    """Formato da resposta que pedimos à OpenAI."""
    critical_terms: list[str] = Field(description="Termos clínicos críticos citados (dor, falta de ar, tontura, sangramento...)")
    sentiment: Literal["positivo", "neutro", "negativo"]
    symptoms: list[str] = Field(description="Indícios percebidos no conteúdo da fala: fadiga, dispneia, fala arrastada, confusão, ansiedade...")
    risk_level: Literal["baixo", "medio", "alto"]
    summary: str = Field(description="Resumo clínico em uma ou duas frases, em português")


_TRANSCRIPT_INSTRUCTIONS = (
    "Você é um assistente clínico que apoia a equipe médica de um hospital. "
    "Analise a transcrição de uma gravação de voz de um paciente. Identifique termos críticos, o sentimento "
    "predominante e indícios de condições como fadiga, dispneia (dificuldade respiratória), fala arrastada, dor "
    "ou confusão, relatados pelo paciente ou perceptíveis no texto. Se o texto não parecer a fala de um paciente "
    "(por exemplo, uma leitura, uma música ou um texto sem sentido), classifique o risco como baixo. "
    "Seja objetivo e responda em português."
)


def analyze_transcript(transcript: str) -> TranscriptAnalysis:
    """Manda a transcrição e recebe termos críticos, sentimento, sintomas e risco."""
    response = _client.responses.parse(
        model=config.OPENAI_CHAT_MODEL,
        instructions=_TRANSCRIPT_INSTRUCTIONS,
        input=f"TRANSCRIÇÃO:\n{transcript or '(vazia)'}",
        text_format=TranscriptAnalysis,
        reasoning={"effort": "low"},
    )
    return response.output_parsed


# ------------------------------------------------------------ frames de vídeo
class FrameDescription(BaseModel):
    confirmed: bool = Field(description="True se a imagem é compatível com o evento detectado")
    description: str = Field(description="Descrição objetiva do que se vê, em português")
    risk_level: Literal["baixo", "medio", "alto"]


_FRAME_INSTRUCTIONS = (
    "Você é um assistente clínico que revisa frames de vídeos de pacientes (fisioterapia, internação, cirurgia). "
    "O sistema local detectou um evento usando YOLOv8 e estimativa de pose; a imagem contém as caixas e o esqueleto "
    "desenhados e foi capturada no exato instante da detecção, portanto pode haver borrão de movimento e a ação "
    "pode estar em curso (por exemplo, uma queda ainda não concluída). "
    "Avalie se a cena é compatível com o evento informado, descreva o que vê e classifique o risco. "
    "Seja crítico: confirme apenas quando a imagem mostrar claramente o evento. Posturas normais de exercício "
    "(agachamento, flexão de tronco, apoio nas mãos) e artefatos do esqueleto estimado NÃO confirmam o evento. "
    "Responda em português."
)


def describe_event_frame(frame_path: Path, event: dict) -> FrameDescription:
    """Mostra o frame-chave de um evento à OpenAI e pergunta se a cena confirma o que a regra detectou."""
    image_b64 = base64.b64encode(frame_path.read_bytes()).decode()
    text = (
        f"Evento detectado: {event['type']} ({event['description']}). "
        f"Instante: {event['start']:.1f}s a {event['end']:.1f}s."
    )
    response = _client.responses.parse(
        model=config.OPENAI_CHAT_MODEL,
        instructions=_FRAME_INSTRUCTIONS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"},
            ],
        }],
        text_format=FrameDescription,
        reasoning={"effort": "low"},
    )
    return response.output_parsed


# ------------------------------------------------------- conclusão do relatório
_SUMMARY_INSTRUCTIONS = (
    "Você redige a seção de conclusão de um relatório automático de monitoramento de paciente para a equipe médica. "
    "Com base nos achados estruturados (JSON), escreva de 1 a 3 parágrafos em português: o que foi analisado, "
    "quais eventos ou sinais foram detectados (com instantes de tempo, quando houver), o nível de risco e a ação recomendada. "
    "Use linguagem natural: não cite nomes de campos ou chaves do JSON (por exemplo, escreva 'ângulo do tronco' em vez "
    "de 'torso_angle_deg' e 'severidade ATENÇÃO' em vez de 'severity'). Não invente achados que não estejam no JSON."
)


def write_summary(findings: dict) -> str:
    """Recebe todos os achados e devolve a conclusão do relatório em texto corrido."""
    response = _client.responses.create(
        model=config.OPENAI_CHAT_MODEL,
        instructions=_SUMMARY_INSTRUCTIONS,
        input=json.dumps(findings, ensure_ascii=False, default=str),
        reasoning={"effort": "low"},
    )
    return response.output_text.strip()
