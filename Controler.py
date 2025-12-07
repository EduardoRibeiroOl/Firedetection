import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ----------------------------------------------------------------------
# CONFIGURAÇÕES DA REDE
# ----------------------------------------------------------------------
RASPBERRY_IP = "192.168.137.160"   # ajuste se mudar
VIDEO_URL = f"http://{RASPBERRY_IP}:5000/video"
CONTROL_URL = f"http://{RASPBERRY_IP}:5000/control?cmd="


# ----------------------------------------------------------------------
# FUNÇÃO DE CONTROLE DO ROBÔ (SEM ATRASO)
# ----------------------------------------------------------------------
def robot(cmd):
    try:
        requests.get(CONTROL_URL + cmd, timeout=0.05)
    except:
        pass


# ----------------------------------------------------------------------
# MODELO YOLO
# ----------------------------------------------------------------------
model = YOLO("bo.pt")  # seu modelo

# Acha o índice da classe "fire"
fire_class_idx = None
for idx, name in model.names.items():
    if name.lower() == 'fire':
        fire_class_idx = idx
        break

assert fire_class_idx is not None, "Modelo não contém a classe 'fire'!"


# ----------------------------------------------------------------------
# CAPTURA DE VÍDEO (STREAM DO RASPBERRY)
# ----------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_URL)

CONF_THRESH = 0.1
IOU_THRESH = 0.2

# Filtro de cor
FIRE_COLOR_RANGE = {
    'lower': np.array([0, 100, 100]),
    'upper': np.array([30, 255, 255])
}


def is_fire_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_COLOR_RANGE['lower'], FIRE_COLOR_RANGE['upper'])
    return np.sum(mask) > (roi.size * 0.15)


fire_detected = False


# ----------------------------------------------------------------------
# LOOP PRINCIPAL
# ----------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Não conseguiu pegar frame do Raspberry...")
        continue

    current_frame_fire = False

    # Predição YOLO
    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=1280,
        augment=True,
        device='cpu',
        verbose=False
    )

    # Pós-processamento
    for result in results:
        for box in result.boxes:
            if box.cls != fire_class_idx:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            area = (x2 - x1) * (y2 - y1)

            if not (300 < area < 50000):
                continue
            if not is_fire_color(roi):
                continue

            conf = float(box.conf.item())
            if area < 500:
                conf = min(1.0, conf * 1.5)

            if conf > 0.15:
                current_frame_fire = True

                # Desenho
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Fire {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ------------------------------------------------------------------
    # CONTROLE DO ROBÔ COM BASE NA IA
    # ------------------------------------------------------------------

    if current_frame_fire:
        if not fire_detected:
            print("🔥 FOGO DETECTADO! MANDANDO AVANÇAR")
        fire_detected = True
        robot("up")      # AVANÇA AUTOMATICAMENTE

    else:
        if fire_detected:
            print("Fogo sumiu! Parando robô.")
        fire_detected = False
        robot("stop")    # PARA SE NÃO VÊ FOGO


    # ------------------------------------------------------------------
    # MOSTRAR NA TELA
    # ------------------------------------------------------------------
    cv2.imshow("Detector de Fogo - IA + Raspberry Stream", frame)
    if cv2.waitKey(1) == ord("q"):
        robot("stop")
        break


cap.release()
cv2.destroyAllWindows()
