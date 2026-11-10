"""Junta todas as evidências em um único risco de 0 a 100 e traduz esse número em severidade."""
from . import config

# Pontuação das evidências
WEIGHTS = {
    "video_rule.fall": 70,
    "video_rule.immobility": 35,
    "video_rule.asymmetry": 20,
    "video_rule.person_lost": 15,
    "audio_terms": 8,                     # por termo crítico encontrado na fala
    "audio_symptom": 12,                  # por indício clínico percebido na fala
    "audio_sentiment": 10,                # sentimento predominante negativo
    "audio_risk.alto": 20,
    "audio_risk.medio": 10,
    "llm_vision_confirmed": 15,           # bônus quando a OpenAI olha o frame e confirma o evento
}
# Tetos por categoria, para que muitas evidências pequenas do mesmo tipo não inflem o risco sozinhas.
CAPS = {"audio_terms": 32, "audio_symptom": 36, "video_rule.person_lost": 15, "llm_vision_confirmed": 30}


def score(evidences: list[dict]) -> dict:
    totals: dict[str, float] = {}
    for ev in evidences:
        key = _weight_key(ev)
        if key is None:
            continue
        totals[key] = totals.get(key, 0.0) + WEIGHTS[key] * ev.get("score", 1.0)

    for key, cap in CAPS.items():
        if key in totals:
            totals[key] = min(totals[key], cap)

    risk = int(round(min(100.0, sum(totals.values()))))
    if risk >= config.SEVERITY_CRITICAL:
        severity = "CRÍTICO"
    elif risk >= config.SEVERITY_ATTENTION:
        severity = "ATENÇÃO"
    else:
        severity = "INFO"
    return {
        "risk": risk,
        "severity": severity,
        "alert": severity != "INFO",
        "breakdown": {k: round(v, 1) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])},
    }


def _weight_key(ev: dict):
    """Descobre qual linha da tabela WEIGHTS se aplica à evidência. None significa que ela não pontua."""
    source, kind = ev["source"], ev.get("type", "")
    if source == "video_rule":
        return f"video_rule.{kind}"
    if source == "llm_vision":
        return "llm_vision_confirmed" if ev.get("score", 0) >= 1 else None
    if source == "audio_risk":
        return f"audio_risk.{kind}" if kind in ("alto", "medio") else None
    return source if source in WEIGHTS else None
