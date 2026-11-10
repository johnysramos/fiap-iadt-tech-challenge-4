"""Ponto de entrada: python -m apolo <arquivo>

Recebe um único arquivo de áudio ou vídeo, roda a análise que cabe a ele e deixa em outputs/
um relatório em Markdown (mais o vídeo anotado, no caso de vídeo). Se o risco chegar a ATENÇÃO
ou CRÍTICO, um alerta para a equipe médica é impresso no console.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from . import config, media, openai_client, report, scoring, video_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apolo",
        description="Analisa um arquivo de áudio ou vídeo de paciente, procura sinais de risco e gera relatório com alerta.",
    )
    parser.add_argument("arquivo", type=Path, help="caminho do arquivo de áudio ou vídeo")
    path = parser.parse_args().arquivo

    try:
        kind = media.classify(path)
    except ValueError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    if not config.OPENAI_API_KEY:
        print("ERRO: defina OPENAI_API_KEY (variável de ambiente ou arquivo .env).", file=sys.stderr)
        return 2

    # Cada execução ganha a própria pasta, com data e hora no nome.
    started = datetime.now()
    out_dir = config.OUTPUTS_DIR / f"{started:%Y-%m-%d_%H%M%S}_{path.stem}"
    out_dir.mkdir(parents=True)
    print(f"[apolo] Arquivo: {path.name} ({kind})")
    print(f"[apolo] Saída:   {out_dir}")

    # Dicionário com as análises
    ctx = {
        "file": {"name": path.name, "path": str(path.resolve()), "kind": kind, "metadata": {}},
        "analyzed_at": started.isoformat(timespec="seconds"),
        "evidences": [],
    }

    if kind == "video":
        _run_video(path, out_dir, ctx)
    else:
        _run_audio(path, out_dir, ctx)

    ctx["result"] = scoring.score(ctx["evidences"])
    print("[apolo] Pedindo à OpenAI a conclusão do relatório...")
    ctx["summary"] = openai_client.write_summary({
        "file": ctx["file"],
        "evidences": ctx["evidences"],
        "result": ctx["result"],
        "audio_analysis": ctx.get("audio", {}).get("analysis"),
        "video_stats": ctx.get("video", {}).get("stats"),
    })
    report.write(out_dir, ctx)

    _print_alert(ctx)
    print(f"[apolo] Concluído em {(datetime.now() - started).seconds}s. Relatório: {out_dir / 'report.md'}")
    return 0


def _run_video(path: Path, out_dir: Path, ctx: dict) -> None:
    """Fluxo de vídeo."""
    ctx["file"]["metadata"] = media.video_metadata(path)

    print("[apolo] Analisando o vídeo com YOLOv8 + MediaPipe Pose...")
    video = video_analysis.analyze(path, out_dir)
    events = video["events"]
    print(f"[apolo] {video['stats']['frames_analyzed']} frames analisados, {len(events)} evento(s) detectado(s) pelas regras.")

    # Frame-chave dos eventos
    for ev in events:
        if ev["type"] == "person_lost":
            continue   # sem ninguém no frame não há o que confirmar
        print(f"[apolo] Pedindo à OpenAI (visão) que avalie o evento '{ev['type']}' em {ev['start']:.1f}s...")
        desc = openai_client.describe_event_frame(Path(ev["frame"]), ev)
        ev["llm_validation"] = f"{'confirmado' if desc.confirmed else 'não confirmado'} (risco {desc.risk_level}): {desc.description}"
        ctx["evidences"].append({
            "source": "llm_vision", "type": ev["type"], "start": ev["start"], "end": ev["end"],
            "score": 1.0 if desc.confirmed else 0.0, "description": ev["llm_validation"],
        })

    ctx["evidences"] += events
    ctx["video"] = {"stats": video["stats"], "annotated_video": video["annotated_video"]}


def _run_audio(path: Path, out_dir: Path, ctx: dict) -> None:
    """Fluxo de áudio: a OpenAI transcreve a gravação e depois classifica o texto."""
    ctx["file"]["metadata"] = media.audio_metadata(path)

    print("[apolo] Transcrevendo o áudio com a OpenAI...")
    transcript = openai_client.transcribe(path)
    (out_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    print("[apolo] Classificando a transcrição com a OpenAI (termos críticos, sentimento, sintomas, risco)...")
    analysis = openai_client.analyze_transcript(transcript)

    # Evidências da OpenAI
    def evidence(source, kind, description):
        return {"source": source, "type": kind, "start": None, "end": None, "score": 1.0, "description": description}

    for term in analysis.critical_terms:
        ctx["evidences"].append(evidence("audio_terms", "critical_term", f"Termo crítico na fala: '{term}'"))
    for symptom in analysis.symptoms:
        ctx["evidences"].append(evidence("audio_symptom", "symptom", f"Indício na fala: {symptom}"))
    if analysis.sentiment == "negativo":
        ctx["evidences"].append(evidence("audio_sentiment", "negativo", "Sentimento predominante negativo"))
    if analysis.risk_level in ("medio", "alto"):
        ctx["evidences"].append(evidence("audio_risk", analysis.risk_level, f"Risco clínico percebido na fala: {analysis.risk_level}"))

    ctx["audio"] = {"transcript": transcript, "analysis": analysis.model_dump()}


def _print_alert(ctx: dict) -> None:
    """Imprime o alerta no console quando a severidade é ATENÇÃO ou CRÍTICO."""
    r = ctx["result"]
    if not r["alert"]:
        print(f"[apolo] Risco {r['risk']}/100 (INFO). Nenhum alerta emitido.")
        return
    width = 70
    print("\n" + "=" * width)
    print(f"  ALERTA À EQUIPE MÉDICA — {r['severity']} (risco {r['risk']}/100)")
    print(f"  Arquivo: {ctx['file']['name']}")
    print("-" * width)
    for ev in ctx["evidences"]:
        if ev["source"] == "llm_vision":
            continue   # a validação visual já aparece junto do evento no relatório
        when = f"{ev['start']:.1f}s" if ev.get("start") is not None else "geral"
        print(f"  [{ev['source']:<15}] {when:>8}  {ev['description']}")
    print("=" * width + "\n")


if __name__ == "__main__":
    sys.exit(main())
