"""Análise de vídeo: percorre os frames com YOLOv8 e MediaPipe Pose, grava o vídeo anotado e aplica as regras de evento.
"""
import math
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from . import config

# Índices dos pontos do MediaPipe Pose que usamos
_NOSE, _L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE = 0, 11, 12, 23, 24, 27, 28
_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

# Limiares das regras de evento. Foram calibrados com as sequências de queda e de agachamento do UR Fall Detection Dataset.
_FALL_DROP_RATIO = 0.40        # o quadril precisa descer mais de 40% da altura do corpo em até 1 s
_FALL_TORSO_DEG = 30.0         # e o tronco sair mais de 30° da vertical (em 2D, cair rumo à câmera encurta o tronco)
_FALL_MIN_VELOCITY = 0.6       # com pico de velocidade alto; agachar ou sentar fica abaixo de 0,4
_MIN_BODY_HEIGHT = 0.25        # só avaliamos queda se a pessoa ocupa ao menos 1/4 da altura do quadro (corpo inteiro visível)
_FRONTAL_SHOULDER_RATIO = 0.15 # ombros com ao menos 15% da altura do corpo indicam vista frontal, única em que medimos inclinação
_IMMOBILITY_VELOCITY = 0.04    # abaixo disso (unidades normalizadas por segundo) consideramos a pessoa parada
_IMMOBILITY_SECONDS = 10.0
_TILT_DEG = 20.0               # inclinação dos ombros/quadril que conta como assimetria
_TILT_SECONDS = 3.0
_LOST_SECONDS = 3.0            # tempo sem ninguém em cena para disparar "pessoa fora de quadro"
_COOLDOWN_SECONDS = 5.0        # intervalo mínimo entre dois eventos do mesmo tipo
_KEYFRAME_DELAY = 1.0          # o frame-chave é salvo 1 s depois da detecção, quando o movimento já terminou

_BOX_COLOR = (0, 200, 255)
_SKELETON_COLOR = (0, 255, 120)
_JOINT_COLOR = (255, 80, 80)
_ALERT_COLOR = (0, 0, 255)


def analyze(video_path: Path, out_dir: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps / config.ANALYSIS_FPS))   # analisamos 1 a cada `stride` frames
    dt = stride / fps
    aspect = width / height

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    yolo = YOLO(str(config.YOLO_MODEL_PATH))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
    ))

    events, pending = [], []
    objects = Counter()
    max_persons, pose_hits, analyzed = 0, 0, 0
    state = _RuleState()
    last_boxes, last_landmarks, prev_points = [], None, None

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps

        if idx % stride == 0:
            analyzed += 1
            det = yolo.predict(frame, conf=config.YOLO_CONFIDENCE, verbose=False, device="cpu")[0]
            last_boxes = [(tuple(int(v) for v in b.xyxy[0].tolist()), det.names[int(b.cls[0])], float(b.conf[0])) for b in det.boxes]
            persons = sum(1 for _, name, _ in last_boxes if name == "person")
            objects.update(name for _, name, _ in last_boxes)
            max_persons = max(max_persons, persons)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = pose.detect_for_video(mp_image, int(t * 1000))
            last_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

            feats, points = _pose_features(last_landmarks, prev_points, dt, aspect)
            if last_landmarks is not None:
                pose_hits += 1
            prev_points = points
            feats["persons"] = persons

            for ev in state.update(t, feats, dt):
                ev["frame"] = str(frames_dir / f"evento_{len(events) + 1:02d}_{ev['type']}.jpg")
                events.append(ev)
                pending.append((t + _KEYFRAME_DELAY, ev))   # o frame-chave fica agendado para daqui a 1 s

        # Todos os frames vão para o vídeo anotado, repetindo a última detecção nos que não foram analisados.
        active = [e["type"] for e in events if t - e["end"] <= 3.0]
        _draw(frame, last_boxes, last_landmarks, width, height, active)
        for due, ev in [p for p in pending if t >= p[0]]:
            cv2.imwrite(ev["frame"], frame)
            pending.remove((due, ev))
        writer.write(frame)
        last_frame = frame
        idx += 1

    for _, ev in pending:   # se o vídeo acabou antes do frame-chave, fica o último frame
        cv2.imwrite(ev["frame"], last_frame)

    cap.release()
    writer.release()
    pose.close()

    return {
        "events": events,
        "annotated_video": str(out_dir / "annotated.mp4"),
        "stats": {
            "frames_total": total,
            "frames_analyzed": analyzed,
            "analysis_fps": round(fps / stride, 2),
            "pose_detected_ratio": pose_hits / analyzed if analyzed else 0.0,
            "max_persons": max_persons,
            "objects": dict(objects.most_common(10)),
        },
    }


# ------------------------------------------------------------------ features
def _pose_features(landmarks, prev_points, dt: float, aspect: float):
    """Transforma os 33 pontos da pose nos números que as regras usam."""
    nan = float("nan")
    if landmarks is None:
        return {"hip_height": nan, "torso_angle_deg": nan, "velocity": nan, "tilt_deg": nan,
                "body_height": nan, "framed": False}, None

    # As coordenadas vêm normalizadas de 0 a 1; multiplicamos x pela proporção da imagem para os ângulos saírem certos.
    pts = np.array([[lm.x * aspect, lm.y] for lm in landmarks])
    vis = np.array([getattr(lm, "visibility", 1.0) or 1.0 for lm in landmarks])

    shoulder_mid = (pts[_L_SHOULDER] + pts[_R_SHOULDER]) / 2
    hip_mid = (pts[_L_HIP] + pts[_R_HIP]) / 2
    ankle_mid = (pts[_L_ANKLE] + pts[_R_ANKLE]) / 2

    torso = shoulder_mid - hip_mid
    torso_angle = math.degrees(math.atan2(abs(torso[0]), abs(torso[1]) + 1e-9))   # 0° = em pé, 90° = deitado

    def line_tilt(a, b):
        d = pts[a] - pts[b]
        return math.degrees(math.atan2(abs(d[1]), abs(d[0]) + 1e-9))

    body_height = float(abs(ankle_mid[1] - pts[_NOSE][1]))
    # "framed" diz se o corpo inteiro está visível; só nesses frames a regra de queda confia na medida.
    framed = bool(body_height >= _MIN_BODY_HEIGHT and min(vis[_NOSE], vis[_L_ANKLE], vis[_R_ANKLE]) > 0.5)
    shoulder_width = abs(pts[_L_SHOULDER][0] - pts[_R_SHOULDER][0])
    frontal = body_height > 0 and shoulder_width / body_height >= _FRONTAL_SHOULDER_RATIO
    tilt = (line_tilt(_L_SHOULDER, _R_SHOULDER) + line_tilt(_L_HIP, _R_HIP)) / 2 if frontal else nan

    # Velocidade: quanto os pontos visíveis se deslocaram desde o frame analisado anterior.
    velocity = nan
    if prev_points is not None:
        mask = vis > 0.5
        if mask.any():
            velocity = float(np.linalg.norm(pts[mask] - prev_points[mask], axis=1).mean() / dt)

    return {
        "hip_height": float(1.0 - hip_mid[1]),   # 1 = topo da imagem, 0 = base
        "torso_angle_deg": float(torso_angle),
        "velocity": velocity,
        "tilt_deg": float(tilt),
        "body_height": body_height,
        "framed": framed,
    }, pts


# -------------------------------------------------------------------- regras
class _RuleState:
    """Guarda o que as regras precisam lembrar entre um frame e o próximo."""

    def __init__(self):
        self.recent = deque()            # (t, altura do quadril, altura do corpo, velocidade, enquadrado) do último 1 s
        self.still_since = None
        self.tilt_since = None
        self.lost_since = None
        self.lost_fired = False
        self.seen_person = False
        self.last_fired = {}

    def update(self, t: float, f: dict, dt: float) -> list[dict]:
        fired = []

        def fire(kind, start, desc):
            # Respeita o intervalo mínimo entre eventos do mesmo tipo, senão uma queda viraria dez eventos seguidos.
            if t - self.last_fired.get(kind, -1e9) >= _COOLDOWN_SECONDS:
                self.last_fired[kind] = t
                fired.append({"source": "video_rule", "type": kind, "start": round(max(0.0, start), 2),
                              "end": round(t, 2), "score": 1.0, "description": desc})

        has_pose = not math.isnan(f["hip_height"])

        # Queda. Comparamos a altura atual do quadril com o ponto mais alto do último segundo. Para contar como
        # queda, o quadril precisa ter descido bastante e rápido, e o tronco precisa ter saído da vertical.
        # A referência só vale se o corpo inteiro estava enquadrado, o que descarta close-ups e cortes de câmera,
        # e a descida não pode ser maior que a própria altura do corpo (um salto de edição faria isso).
        if has_pose:
            vel = 0.0 if math.isnan(f["velocity"]) else f["velocity"]
            self.recent.append((t, f["hip_height"], f["body_height"], vel, f["framed"]))
            while self.recent and t - self.recent[0][0] > 1.0:
                self.recent.popleft()
            refs = [r for r in self.recent if r[4]]
            t_max, hip_max, body_h, _, _ = max(refs, key=lambda r: r[1]) if refs else (t, f["hip_height"], 0.0, 0.0, False)
            drop = hip_max - f["hip_height"]
            peak_vel = max(r[3] for r in self.recent if r[0] >= t_max)
            if (body_h > 0 and _FALL_DROP_RATIO * body_h < drop <= body_h and peak_vel >= _FALL_MIN_VELOCITY
                    and f["torso_angle_deg"] > _FALL_TORSO_DEG):
                fire("fall", t_max, f"Queda: quadril caiu {drop / body_h:.0%} da altura do corpo em "
                                    f"{t - t_max:.1f}s (velocidade de pico {peak_vel:.2f}) e tronco a "
                                    f"{f['torso_angle_deg']:.0f}° da vertical")
        else:
            self.recent.clear()   # sem pose não dá para comparar; recomeça a janela quando a pessoa reaparecer

        # Imobilidade prolongada: a pessoa está em cena, mas praticamente não se mexe há muito tempo.
        if has_pose and not math.isnan(f["velocity"]) and f["velocity"] < _IMMOBILITY_VELOCITY:
            self.still_since = t if self.still_since is None else self.still_since
            if t - self.still_since >= _IMMOBILITY_SECONDS:
                fire("immobility", self.still_since, f"Paciente imóvel há {t - self.still_since:.0f}s")
        else:
            self.still_since = None

        # Assimetria postural: ombros ou quadril inclinados por alguns segundos. Só faz sentido de frente para a câmera.
        if has_pose and not math.isnan(f["tilt_deg"]) and f["tilt_deg"] > _TILT_DEG:
            self.tilt_since = t if self.tilt_since is None else self.tilt_since
            if t - self.tilt_since >= _TILT_SECONDS:
                fire("asymmetry", self.tilt_since, f"Inclinação de ombros/quadril de {f['tilt_deg']:.0f}° por "
                                                   f"{t - self.tilt_since:.0f}s")
        else:
            self.tilt_since = None

        # Pessoa fora de quadro: havia alguém e agora não há mais ninguém. Dispara uma vez por sumiço.
        if f["persons"] > 0:
            self.seen_person = True
            self.lost_since = None
            self.lost_fired = False
        elif self.seen_person:
            self.lost_since = t if self.lost_since is None else self.lost_since
            if t - self.lost_since >= _LOST_SECONDS and not self.lost_fired:
                self.lost_fired = True
                fire("person_lost", self.lost_since, f"Nenhuma pessoa em cena há {t - self.lost_since:.0f}s")

        return fired


# ------------------------------------------------------------------- desenho
def _draw(frame, boxes, landmarks, width, height, active_events):
    """Desenha as caixas do YOLO, o esqueleto do Pose e, se houver evento recente, uma faixa preta no topo."""
    for (x1, y1, x2, y2), name, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BOX_COLOR, 2)
    if landmarks is not None:
        pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for a, b in _CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], _SKELETON_COLOR, 2)
        for p in pts:
            cv2.circle(frame, p, 3, _JOINT_COLOR, -1)
    if active_events:
        label = "EVENTO: " + ", ".join(dict.fromkeys(active_events))
        cv2.rectangle(frame, (0, 0), (width, 36), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _ALERT_COLOR, 2)
