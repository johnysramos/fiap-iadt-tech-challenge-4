# Amostras de áudio e vídeo

Todos os arquivos desta pasta são públicos e podem ser usados livremente para fins acadêmicos. Nenhum deles contém dados reais de pacientes.

## Vídeos

Só as imagens dos vídeos são analisadas; a trilha de áudio, quando existe, é ignorada.

| Arquivo | O que mostra | De onde veio | Licença |
|---|---|---|---|
| `urfd_fall-01.mp4` | Uma pessoa em pé que cai no chão (sequência `fall-01`, câmera 0, 160 frames a 30 fps, 640x480). Montado a partir dos PNGs do dataset com o ffmpeg. | [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html), Universidade de Rzeszów | CC BY-NC-SA 4.0 (uso acadêmico, não comercial) |
| `urfd_adl-01.mp4` | Uma atividade do dia a dia sem queda: a pessoa agacha para pegar algo (sequência `adl-01`, câmera 0, 150 frames a 30 fps, 640x480). | Idem | Idem |
| `commons_squat_demo.webm` | Demonstração de agachamento (7 s, 1280x720). | [Wikimedia Commons: Squat - exercise demonstration video](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) | CC BY 3.0 |
| `commons_squat_frontal_raise.webm` | Agachamento com elevação frontal dos braços (28 s, 640x480). | [Wikimedia Commons: Squat and Frontal Raise](https://commons.wikimedia.org/wiki/File:Squat_and_Frontal_Raise.webm) | CC BY-SA 4.0 |
| `commons_propiocepcion_fisioterapia.webm` | Sessão de fisioterapia com exercícios de propriocepção de membro inferior (98 s, 1920x1080). É um vídeo editado, com cortes e close-ups. | [Wikimedia Commons: Trabajo propioceptivo en miembro inferior](https://commons.wikimedia.org/wiki/File:Trabajo_propioceptivo_en_miembro_inferior.webm) | CC BY-SA 3.0 |

O UR Fall Detection Dataset pede que seja citado assim: Bogdan Kwolek, Michal Kepski, "Human fall detection on embedded platform using depth maps and wireless accelerometer", Computer Methods and Programs in Biomedicine, vol. 117, n. 3, 2014, pp. 489-501.

## Áudios

| Arquivo | O que contém | De onde veio | Licença |
|---|---|---|---|
| `librivox_fim_carneiro.mp3` | Leitura em português do poema "Fim", de Mário de Sá-Carneiro (31 s). Caso de controle: fala fluente, sem queixa clínica. | [LibriVox / archive.org: Coleção de Poemas em Português](https://archive.org/details/poemas_portugues_001_librivox) | Domínio público |
| `sintetico_paciente_dispneia.mp3` | **Áudio sintético**, gerado com o TTS da OpenAI (`gpt-4o-mini-tts`, voz `onyx`, com instrução de fala cansada e ofegante). Um paciente fictício relata falta de ar, aperto no peito, tontura e fala arrastada (27 s). Como não existe amostra pública de fala clínica em português, o caso positivo foi sintetizado. | Gerado para o projeto | Uso livre |