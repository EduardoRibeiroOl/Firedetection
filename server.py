from ultralytics import YOLO
from flask import Flask, render_template
import cv2
import threading
import lgpio
import time

# GPIO

chip = lgpio.gpiochip_open(0)

# Motor esquerdo
M1_A = 17
M1_B = 27

# Motor direito
M2_A = 22
M2_B = 23

for pin in [M1_A, M1_B, M2_A, M2_B]:
    lgpio.gpio_claim_output(chip, pin)

# Funções dos motores

def parar():
    lgpio.gpio_write(chip, M1_A, 0)
    lgpio.gpio_write(chip, M1_B, 0)
    lgpio.gpio_write(chip, M2_A, 0)
    lgpio.gpio_write(chip, M2_B, 0)

def frente():
    lgpio.gpio_write(chip, M1_A, 1)
    lgpio.gpio_write(chip, M1_B, 0)

    lgpio.gpio_write(chip, M2_A, 1)
    lgpio.gpio_write(chip, M2_B, 0)

def tras():
    lgpio.gpio_write(chip, M1_A, 0)
    lgpio.gpio_write(chip, M1_B, 1)

    lgpio.gpio_write(chip, M2_A, 0)
    lgpio.gpio_write(chip, M2_B, 1)

def esquerda():
    lgpio.gpio_write(chip, M1_A, 0)
    lgpio.gpio_write(chip, M1_B, 1)

    lgpio.gpio_write(chip, M2_A, 1)
    lgpio.gpio_write(chip, M2_B, 0)

def direita():
    lgpio.gpio_write(chip, M1_A, 1)
    lgpio.gpio_write(chip, M1_B, 0)

    lgpio.gpio_write(chip, M2_A, 0)
    lgpio.gpio_write(chip, M2_B, 1)

# IA

modelo = YOLO("fogo.pt") # caminho do modelo

def camera_thread():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        resultados = modelo(frame)

        frame_anotado = resultados[0].plot()

        cv2.imshow("Deteccao de Fogo", frame_anotado)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/frente")
def mover_frente():
    frente()
    return "ok"

@app.route("/tras")
def mover_tras():
    tras()
    return "ok"

@app.route("/esquerda")
def mover_esquerda():
    esquerda()
    return "ok"

@app.route("/direita")
def mover_direita():
    direita()
    return "ok"

@app.route("/parar")
def motor_parar():
    parar()
    return "ok"

# MAIN

if __name__ == "__main__":

    t = threading.Thread(target=camera_thread)
    t.daemon = True
    t.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )