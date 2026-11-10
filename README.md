# apolo — Tech Challenge Fase 4 (FIAP IADT)

O apolo recebe **um arquivo de áudio ou de vídeo** de um paciente, analisa o conteúdo, procura sinais de risco e escreve um relatório. Quando o risco é alto, ele também emite um alerta para a equipe médica.

Cada execução cuida de um arquivo só, do começo ao fim, e deixa tudo o que encontrou em uma pasta própria.

## O que ele faz

**Com um vídeo**, o YOLOv8 localiza pessoas e objetos em cena e o MediaPipe Pose, que faz o papel do OpenPose citado no enunciado, estima a postura da pessoa frame a frame. A partir da postura, quatro regras detectam queda, imobilidade prolongada, assimetria postural e pessoa fora de quadro. Cada evento ganha um frame-chave, que é enviado à OpenAI para ela confirmar se a cena mostra mesmo o que a regra apontou. O resultado inclui o **vídeo anotado**, com as caixas e o esqueleto desenhados.

**Com um áudio**, a OpenAI transcreve a gravação e depois classifica o texto: termos críticos, sentimento, indícios de fadiga, dispneia ou fala arrastada e um nível de risco.

**No fim**, as evidências são somadas em um risco de 0 a 100, que vira uma severidade: INFO, ATENÇÃO ou CRÍTICO. A partir de ATENÇÃO, o alerta aparece no console. A OpenAI também escreve a conclusão do relatório.

## Requisitos

- Windows, Linux ou macOS com **Python 3.11**. Versões mais novas ainda não têm pacotes estáveis de torch, ultralytics e mediapipe.
- Uma chave da OpenAI com créditos.

## Instalação

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Baixe os dois modelos locais para a pasta `models/`:

```powershell
curl.exe -L -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
curl.exe -L -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Por último, copie `.env.example` para `.env` e preencha `OPENAI_API_KEY`. Se a chave já estiver como variável de ambiente do sistema, não precisa colocá-la no `.env`.

## Como usar

```powershell
python -m apolo data\samples\urfd_fall-01.mp4
python -m apolo data\samples\sintetico_paciente_dispneia.mp3
```

O comando recebe um único argumento: o caminho do arquivo. São aceitos vídeos `.mp4 .mov .avi .mkv .webm` e áudios `.wav .mp3 .m4a .flac .ogg` de até 25 MB. Qualquer outro tipo é recusado com uma mensagem de erro.

Cada execução grava os resultados em `outputs/<data_hora>_<nome do arquivo>/`:

| Arquivo | O que é |
|---|---|
| `report.md` | O relatório: risco, eventos ou sinais detectados, transcrição, composição do risco e a conclusão escrita pela OpenAI |
| `annotated.mp4` | O vídeo com as caixas do YOLOv8, o esqueleto do MediaPipe Pose e uma faixa indicando o evento ativo (só para vídeo) |
| `frames/` | Um frame-chave por evento detectado (só para vídeo) |
| `transcript.txt` | A transcrição da fala (só para áudio) |

Quando a severidade chega a ATENÇÃO ou CRÍTICO, o console mostra algo assim:

```
======================================================================
  ALERTA À EQUIPE MÉDICA — CRÍTICO (risco 85/100)
  Arquivo: urfd_fall-01.mp4
----------------------------------------------------------------------
  [video_rule     ]     2.5s  Queda: quadril caiu 58% da altura do corpo em 1.0s (velocidade de pico 0.94) e tronco a 33° da vertical
======================================================================
```

## Como o código está organizado

```
apolo/
├── __main__.py        ponto de entrada: decide se é áudio ou vídeo e roda o pipeline
├── config.py          .env, caminhos e os números fixos do pipeline
├── media.py           confere o tipo do arquivo e lê os metadados
├── openai_client.py   classe de interação com a API da OpenAI
├── video_analysis.py  YOLOv8 + MediaPipe Pose, vídeo anotado, medidas de postura e regras de evento
├── scoring.py         soma as evidências em um risco de 0 a 100
└── report.py          escreve o report.md
data/samples/          amostras públicas usadas nos testes (fontes e licenças em SOURCES.md)
docs/                  relatório técnico
models/                yolov8n.pt e pose_landmarker_lite.task
```

## Limitações

- O vídeo precisa vir de **câmera fixa, com uma pessoa principal**, como em um leito ou uma sala de fisioterapia. Vídeos editados, com cortes e close-ups, geram eventos falsos de "pessoa fora de quadro".
- A pose é estimada em 2D.
- As respostas da OpenAI não são determinísticas. A mesma gravação pode receber termos críticos ou sentimento um pouco diferentes em execuções distintas.
