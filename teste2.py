import cv2
import numpy as np
from ultralytics import YOLO

# 1. Carregar modelo
model = YOLO(r'C:\ex\neurose\models\bo.pt')

# 2. Verificar índice da classe 'Fire'
fire_class_idx = None
for idx, name in model.names.items():
    if name.lower() == 'fire':
        fire_class_idx = idx
        break
assert fire_class_idx is not None, "Modelo não contém a classe 'Fire'!"

# 3. Configurações de detecção
CONF_THRESH = 0.1
IOU_THRESH = 0.2
FIRE_COLOR_RANGE = {
    'lower': np.array([0, 100, 100]),
    'upper': np.array([30, 255, 255])
}

# 4. Funções de filtro
def is_fire_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_COLOR_RANGE['lower'], FIRE_COLOR_RANGE['upper'])
    return np.sum(mask) > (roi.size * 0.15)

# 5. Captura de vídeo
cap = cv2.VideoCapture(0)
fire_detected = False  # Variável para controle de estado

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame_fire = False  # Flag para fogo no frame atual
    
    # Detecção
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
            if box.cls == fire_class_idx:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                area = (x2 - x1) * (y2 - y1)
                
                if not (300 < area < 50000): continue
                if not is_fire_color(roi): continue
                
                conf = box.conf.item()
                if area < 500:
                    conf = min(1.0, conf * 1.5)
                
                if conf > 0.15:
                    current_frame_fire = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 
                               f"Fire {conf:.2f}", 
                               (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
    
    # Lógica de impressão no terminal
    if current_frame_fire and not fire_detected:
        print(1)  # Imprime 1 quando o fogo é detectado (apenas na primeira ocorrência)
        fire_detected = True
    elif not current_frame_fire:
        fire_detected = False
    
    # Exibição
    cv2.imshow('Detector de Fogo - Tempo Real', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()