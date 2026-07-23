from ultralytics import YOLO
from flask import Flask, render_template
import cv2
import threading
from time import sleep
from flask import Flask, Response, request, jsonify
from gpiozero import Motor
from calculos import ponto_medio_box
from calculos import optica

app = Flask(__name__)

MotorA = Motor(forward=17, backward=27)
MotorB = Motor(forward=22, backward=23)

current_cmd = "stop"
#state_lock = threading.Lock() # Posso dar mil motivos pra isso aqui, mas joga no Claude que ele explica melhor que eu

FOV_cam = 70 # graus, campo de visão da camera(fov)
model = YOLO("Firedetection/models/bo.pt")
#print(model.names) 

data_metrics = [[], []] # armazenar estado do angulo, e dos centros [angulo1, centro_x1, centro_y1] [angulo2, centro_x2, centro_y2]
clock_state = 0  # 0 primeira medição, 1 para estado de movimento, 2 para estado de parada, 3 para segunda medição

def camera_thread():
    global clock_state
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
                continue  
            
            x1, y1, x2, y2 = box.xyxy[0]
            center_x, center_y = ponto_medio_box(x1, y1, x2, y2)            
            
            #Aqui vai veririficar se realmente está tendo fogo na localidade
            # ou se é apenas algo sendo confundido e VAI PARAR DE SE MOVER


            #Faz a medição do ângulo da caixa de detecção em relação ao centro da imagem  
            angle_catching = optica(center_x, frame.shape[1], FOV_cam)
            
            if clock_state == 0:
                data_metrics[0] = [angle_catching, center_x, center_y]
                clock_state = 1
            
            elif clock_state == 1:
                # vai se mover aproximadamente um pouco a direita, uns 40 cm
                
                MotorA.forward()
                MotorB.stop()
                sleep (2) # se movendo para a direita
                MotorA.forward()
                MotorB.forward()
                sleep (4) # tempo de movimento para A FRENTE, pode ser ajustado conforme a necessidade
                MotorA.stop()
                MotorB.stop()
                sleep (1) # tempo de espera para parar o movimento
                MotorB.forward ()
                sleep (2) # se movendo para a esquerda
                MotorA.stop()
                MotorB.stop()
                
                # coloca um sensor para ver se parou e depois volta para a vista normal 
                #if movimento acabou:
                clock_state = 2

            elif clock_state == 2:
                data_metrics[1] = [angle_catching, center_x, center_y]
                #fazer tudo oque precisa com lançador
                clock_state = 1 


            # Se quiser, pode descomentar essas linhas para ver os valores no terminal
            #print(center_x, center_y)
            #print(frame.shape)
            #print(f"{angle_catching:.2f}°")


        frame_anotado = results[0].plot()

        cv2.imshow("Deteccao de Fogo", frame_anotado)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

camera_thread()