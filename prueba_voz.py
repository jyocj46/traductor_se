import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from collections import deque
import pyttsx3
import time
from sqlalchemy import create_engine
import pandas as pd

# Configuración de voz
voz = pyttsx3.init()
voz.setProperty('rate', 140)
voz.setProperty('voice', voz.getProperty('voices')[0].id)

# Conexión a base de datos para obtener palabras válidas
DB_USER = "usr_traductor"
DB_PASSWORD = "oUxJEZ59sw6bPZdBBPfSvJNUBzTlkQ5f"
DB_HOST = "dpg-d0mjg0e3jp1c738dep3g-a.oregon-postgres.render.com"
DB_PORT = "5432"
DB_NAME = "lenguaje_senias_g7z9"
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Leer palabras válidas
palabras_validas = set(pd.read_sql_table('palabras', engine)['palabra'].str.upper())

# Cargar modelo y codificador
interpreter = tf.lite.Interpreter(model_path="modelo_senas.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
encoder = joblib.load("label_encoder.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables
buffer_predicciones = deque(maxlen=10)
palabra_actual = ""
letra_mostrada = ""
umbral_consistencia = 7
letra_anterior = ""
tiempo_inicio = None
tiempo_ultima_letra_confirmada = time.time()
esperando_decir = False
contador_espera_activo = False

# Cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

cv2.namedWindow("Traductor IA", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Traductor IA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🖐️ Traductor iniciado. Presiona ESPACIO para confirmar letra, 'v' para decir la palabra, 'ESC' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alto, ancho, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    puntos = []

    if result and result.multi_hand_landmarks:
        contador_espera_activo = False
        cantidad_manos = len(result.multi_hand_landmarks)
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

        if cantidad_manos == 1:
            puntos.extend([0.0] * 63)
        elif cantidad_manos != 2:
            continue

        if len(puntos) == 126:
            entrada = np.array(puntos, dtype=np.float32).reshape(1, -1)
            interpreter.set_tensor(input_details[0]['index'], entrada)
            interpreter.invoke()
            salida = interpreter.get_tensor(output_details[0]['index'])
            letra = encoder.inverse_transform([np.argmax(salida)])[0].strip()
            buffer_predicciones.append(letra)

            if buffer_predicciones.count(letra) >= umbral_consistencia:
                if letra == letra_anterior:
                    if tiempo_inicio and (time.time() - tiempo_inicio >= 2.0):
                        if letra == "DEL":
                            palabra_actual = palabra_actual[:-1]
                        elif letra == "ESP":
                            palabra_actual += " "
                        else:
                            palabra_actual += letra

                        print(f"✅ Letra confirmada automáticamente: {letra}")
                        buffer_predicciones.clear()
                        tiempo_inicio = None
                        letra_anterior = ""
                        tiempo_ultima_letra_confirmada = time.time()
                        esperando_decir = True

                        if palabra_actual.strip().upper() in palabras_validas:
                            print(f"🗣️ Palabra reconocida: {palabra_actual.strip()}")
                            voz.say(palabra_actual.strip())
                            voz.runAndWait()
                            palabra_actual = ""
                            letra_mostrada = ""
                            esperando_decir = False
                else:
                    letra_anterior = letra
                    tiempo_inicio = time.time()

            letra_mostrada = letra

        # for hand_landmarks in result.multi_hand_landmarks:
        #     mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        letra_anterior = ""
        tiempo_inicio = None
        if palabra_actual and not contador_espera_activo:
            tiempo_ultima_letra_confirmada = time.time()
            esperando_decir = True
            contador_espera_activo = True

    # UI lateral
    panel_ancho = 300
    panel = np.ones((alto, panel_ancho, 3), dtype=np.uint8) * 255
    cv2.rectangle(panel, (0, 0), (panel_ancho-1, alto-1), (100, 100, 100), 2)
    cv2.putText(panel, "Letra detectada", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    letra_grande = letra_mostrada if letra_mostrada else "-"
    (tw, th), _ = cv2.getTextSize(letra_grande, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)
    x_text = int((panel_ancho - tw) / 2)
    y_text = int((alto + th) / 2)
    cv2.putText(panel, letra_grande, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 102, 204), 8)

    duracion_objetivo = 1.0
    barra_ancho = 200
    barra_altura = 20
    barra_x = int((panel_ancho - barra_ancho) / 2)
    barra_y = y_text + 60
    if letra_mostrada == letra_anterior and tiempo_inicio:
        tiempo_pasado = time.time() - tiempo_inicio
        progreso = min(tiempo_pasado / duracion_objetivo, 1.0)
        barra_color = (0, 200, 0) if progreso == 1.0 else (0, 150, 255)
        cv2.rectangle(panel, (barra_x, barra_y), (barra_x + barra_ancho, barra_y + barra_altura), (180, 180, 180), 2)
        cv2.rectangle(panel, (barra_x, barra_y), (barra_x + int(barra_ancho * progreso), barra_y + barra_altura), barra_color, -1)

    frame = np.hstack((frame, panel))

    cv2.rectangle(frame, (0, alto - 80), (ancho + panel_ancho, alto), (255, 255, 255), -1)
    cv2.line(frame, (0, alto - 80), (ancho + panel_ancho, alto - 80), (100, 100, 100), 2)
    (tw, th), _ = cv2.getTextSize(palabra_actual, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    x_text = int((ancho + panel_ancho - tw) / 2)
    cv2.putText(frame, palabra_actual, (x_text, alto - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    if esperando_decir and palabra_actual and contador_espera_activo:
        progreso_habla = min((time.time() - tiempo_ultima_letra_confirmada) / 3.0, 1.0)
        barra_azul_x = 50
        barra_azul_y = alto - 20
        barra_azul_ancho = ancho + panel_ancho - 100
        barra_azul_alto = 10
        cv2.rectangle(frame, (barra_azul_x, barra_azul_y), (barra_azul_x + barra_azul_ancho, barra_azul_y + barra_azul_alto), (200, 200, 255), 2)
        cv2.rectangle(frame, (barra_azul_x, barra_azul_y), (barra_azul_x + int(barra_azul_ancho * progreso_habla), barra_azul_y + barra_azul_alto), (255, 120, 0), -1)

    cv2.imshow("Traductor IA", frame)

    if esperando_decir and palabra_actual and contador_espera_activo and (time.time() - tiempo_ultima_letra_confirmada >= 3):
        print(f"🗣️ Hablando automáticamente: {palabra_actual}")
        voz.say(palabra_actual)
        voz.runAndWait()
        palabra_actual = ""
        letra_mostrada = ""
        esperando_decir = False

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32 and letra_mostrada:
        if letra_mostrada == "DEL":
            palabra_actual = palabra_actual[:-1]
        elif letra_mostrada == "ESP":
            palabra_actual += " "
        else:
            palabra_actual += letra_mostrada
            tiempo_ultima_letra_confirmada = time.time()
            esperando_decir = True

        print(f"✅ Letra confirmada: {letra_mostrada}")
        buffer_predicciones.clear()

        if palabra_actual.strip().upper() in palabras_validas:
            print(f"🗣️ Palabra reconocida: {palabra_actual.strip()}")
            voz.say(palabra_actual.strip())
            voz.runAndWait()
            palabra_actual = ""
            letra_mostrada = ""
            esperando_decir = False

    elif key == ord('v') and palabra_actual:
        print(f"🗣️ Hablando: {palabra_actual}")
        voz.say(palabra_actual)
        voz.runAndWait()
        palabra_actual = ""
        letra_mostrada = ""

cap.release()
cv2.destroyAllWindows()