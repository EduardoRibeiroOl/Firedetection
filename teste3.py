import cv2
import numpy as np
from ultralytics import YOLO

# 1. Carregar modelo treinado
model = YOLO('C:/ex/neurose/models/bo.pt')  # Substitua pelo seu caminho

# 2. Configurações
IMG_PATH = 'C:/ex/neurose/models/image.png'  # Substitua pelo caminho da sua imagem
OUTPUT_PATH = 'C:/ex/neurose/models/resultado_deteccao.jpg'
FIRE_COLOR_RANGE = {
    'lower': np.array([0, 100, 100]),  # Tons de vermelho/laranja (HSV)
    'upper': np.array([30, 255, 255])
}

# 3. Funções de filtro
def is_fire_color(roi):
    """Verifica se a região tem cor de fogo"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_COLOR_RANGE['lower'], FIRE_COLOR_RANGE['upper'])
    return np.sum(mask) > (roi.size * 0.15)  # Pelo menos 15% de pixels de fogo

def is_valid_fire(roi, area):
    """Combina múltiplos critérios para validar fogo"""
    # Filtro por tamanho (ajuste conforme necessidade)
    if not (300 < area < 50000):  # Em pixels
        return False
    
    # Filtro por textura (fogo tem alta variação)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
        return False
    
    return is_fire_color(roi)

# 4. Carregar imagem
image = cv2.imread(IMG_PATH)