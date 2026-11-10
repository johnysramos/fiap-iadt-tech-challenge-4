"""Monta o report.md da execução a partir de tudo o que a análise acumulou."""
from pathlib import Path


def write(out_dir: Path, ctx: dict) -> None:
    (out_dir / "report.md").write_text(_markdown(ctx), encoding="utf-8")


def _fmt_t(seconds) -> str:
    """Segundos para mm:ss.d"""
    seconds = float(seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def _markdown(ctx: dict) -> str:
    f, r = ctx["file"], ctx["result"]
    lines = [
        f"# Relatório de monitoramento — {f['name']}",
        "",
        f"- **Data da análise:** {ctx['analyzed_at']}",
        f"- **Tipo:** {f['kind']}",
        "- **Metadados:** " + ", ".join(f"{k}={v}" for k, v in f["metadata"].items()),
        f"- **Risco:** {r['risk']}/100 — **{r['severity']}**",
        f"- **Alerta à equipe médica:** {'SIM' if r['alert'] else 'não'}",
        "",
        "## Conclusão",
        "",
        ctx.get("summary", ""),
        "",
    ]

    video = ctx.get("video")
    if video:
        stats = video["stats"]
        objects = ", ".join(f"{k}={v}" for k, v in stats["objects"].items()) or "nenhum"
        lines += [
            "## Análise de vídeo (YOLOv8 + MediaPipe Pose)",
            "",
            "- Vídeo anotado: [annotated.mp4](annotated.mp4)",
            f"- Frames analisados: {stats['frames_analyzed']} de {stats['frames_total']}",
            f"- Pose detectada em {stats['pose_detected_ratio']:.0%} dos frames analisados",
            f"- Máximo de pessoas em cena: {stats['max_persons']}",
            f"- Objetos detectados (contagem de frames): {objects}",
            "",
            "### Eventos detectados pelas regras",
            "",
        ]
        events = [e for e in ctx["evidences"] if e["source"] == "video_rule"]
        if events:
            lines += ["| Início | Fim | Evento | Descrição | Frame | Validação OpenAI |", "|---|---|---|---|---|---|"]
            for e in events:
                frame = f"[{Path(e['frame']).name}](frames/{Path(e['frame']).name})" if e.get("frame") else ""
                lines.append(f"| {_fmt_t(e['start'])} | {_fmt_t(e['end'])} | {e['type']} | {e['description']} | {frame} | {e.get('llm_validation', '')} |")
        else:
            lines.append("Nenhum evento fora do padrão foi detectado pelas regras.")
        lines.append("")

    audio = ctx.get("audio")
    if audio:
        a = audio["analysis"]
        lines += [
            "## Análise de áudio (OpenAI)",
            "",
            f"- Sentimento: **{a['sentiment']}** — risco clínico percebido: **{a['risk_level']}**",
            f"- Termos críticos: {', '.join(a['critical_terms']) or 'nenhum'}",
            f"- Indícios na fala: {', '.join(a['symptoms']) or 'nenhum'}",
            f"- Resumo: {a['summary']}",
            "",
            "### Transcrição",
            "",
            "> " + (audio["transcript"].replace("\n", " ") or "(vazia)"),
            "",
        ]

    lines += ["## Composição do risco", "", "| Componente | Pontos |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in r["breakdown"].items()]
    lines += ["", "Severidade: INFO abaixo de 30, ATENÇÃO de 30 a 69, CRÍTICO a partir de 70. O alerta é emitido a partir de ATENÇÃO.", ""]
    return "\n".join(lines)
