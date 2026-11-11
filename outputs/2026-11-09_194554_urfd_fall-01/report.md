# Relatório de monitoramento — urfd_fall-01.mp4

- **Data da análise:** 2026-11-09T19:45:54
- **Tipo:** video
- **Metadados:** fps=30.0, frames=160, width=640, height=480, duration_s=5.33
- **Risco:** 85/100 — **CRÍTICO**
- **Alerta à equipe médica:** SIM

## Conclusão

Foi analisado um vídeo de 5,33 segundos com acompanhamento visual do paciente ao longo de parte da cena, com detecção de postura na maior parte dos quadros analisados. O principal achado foi um evento compatível com queda entre 2,5 e 3,5 segundos, quando foi observada uma pessoa caída no chão, parcialmente deitada de lado/prona, próxima à cama e fora de postura ereta. Os sinais associados também indicam uma descida abrupta do quadril e inclinação importante do tronco, reforçando a compatibilidade com queda.

O nível de risco foi classificado como **CRÍTICO**, com alerta ativo. Recomenda-se **intervenção imediata da equipe assistencial**, com verificação presencial do paciente, avaliação de possíveis lesões, checagem de consciência e sinais vitais, além de registro do evento e implementação de medidas de segurança para prevenção de nova queda.

## Análise de vídeo (YOLOv8 + MediaPipe Pose)

- Vídeo anotado: [annotated.mp4](annotated.mp4)
- Frames analisados: 54 de 160
- Pose detectada em 93% dos frames analisados
- Máximo de pessoas em cena: 1
- Objetos detectados (contagem de frames): chair=109, person=42, tv=5, bed=3, couch=2

### Eventos detectados pelas regras

| Início | Fim | Evento | Descrição | Frame | Validação OpenAI |
|---|---|---|---|---|---|
| 00:02.5 | 00:03.5 | fall | Queda: quadril caiu 58% da altura do corpo em 1.0s (velocidade de pico 0.94) e tronco a 33° da vertical | [evento_01_fall.jpg](frames/evento_01_fall.jpg) | confirmado (risco alto): Há uma pessoa caída no chão, parcialmente deitada de lado/prona próxima à cama, com o corpo fora de postura ereta. A cena é compatível com uma queda. |

## Composição do risco

| Componente | Pontos |
|---|---|
| video_rule.fall | 70.0 |
| llm_vision_confirmed | 15.0 |

Severidade: INFO abaixo de 30, ATENÇÃO de 30 a 69, CRÍTICO a partir de 70. O alerta é emitido a partir de ATENÇÃO.
