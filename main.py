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
CONF_THRESH = 0.1  # Limiar baixo para sensibilidade
IOU_THRESH = 0.2   # Overlap reduzido
FIRE_COLOR_RANGE = {
    'lower': np.array([0, 100, 100]),  # HSV: Tons de vermelho/laranja
    'upper': np.array([30, 255, 255])
}

# 4. Funções de filtro
def is_fire_color(roi):
    """Verifica se a região tem pixels na faixa de cor do fogo"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_COLOR_RANGE['lower'], FIRE_COLOR_RANGE['upper'])
    return np.sum(mask) > (roi.size * 0.15)  # Pelo menos 15% de pixels "de fogo"

# 5. Captura de vídeo
cap = cv2.VideoCapture(0)  # Webcam (ou use um vídeo)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detecção
    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=1280,
        augment=True,
        device='cpu',
        verbose=False  # Desativa logs para tempo real
    )
    
    # Pós-processamento
    for result in results:
        for box in result.boxes:
            if box.cls == fire_class_idx:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                area = (x2 - x1) * (y2 - y1)
                
                # Filtros consecutivos
                if not (300 < area < 50000): continue  # Tamanho
                if not is_fire_color(roi): continue    # Cor
                
                # Aumento de confiança para pequenos fogos
                conf = box.conf.item()
                if area < 500:
                    conf = min(1.0, conf * 1.5)  # Limita a 100%
                
                # Desenho
                if conf > 0.15:  # Confiança mínima para exibir
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 
                               f"Fire {conf:.2f}", 
                               (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
    
    # Exibição
    cv2.imshow('Detector de Fogo - Tempo Real', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()