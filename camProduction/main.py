from ultralytics import YOLO
from flask import Flask, render_template
import cv2
import threading
import time

from calculos import ponto_medio_box
from calculos import optica

FOV_cam = 70 # graus, campo de visão da camera(fov)
model = YOLO("Firedetection/models/bo.pt")
#print(model.names) 

def camera_thread():
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        results = model(frame)
        
        for box in results[0].boxes:
            class_id = int(box.cls[0]) # Tem duas classes no BO.pt, 0 para "fire" e 1 para "smoke", 
            label = model.names[class_id] # indice zero bate aqui indicando fogo 

            if label != "fire":
                continue  # sla do porque disso, não era pra ser o contrário?

            x1, y1, x2, y2 = box.xyxy[0]

            centro_x, centro_y = ponto_medio_box(x1, y1, x2, y2)
            angulo = optica(centro_x, frame.shape[1], FOV_cam)
            
            print(centro_x, centro_y)
            print(frame.shape)
            print(f"{angulo:.2f}°")


        frame_anotado = results[0].plot()

        cv2.imshow("Deteccao de Fogo", frame_anotado)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

camera_thread()