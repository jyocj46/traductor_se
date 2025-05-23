import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import deque
import pyttsx3
import time

# Configuración de voz
voz = pyttsx3.init()
voz.setProperty('rate', 140)
voz.setProperty('voice', voz.getProperty('voices')[0].id)

# Cargar modelo y codificador
model = load_model("modelo_senas.h5")
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

# Captura webcam
cap = cv2.VideoCapture(0)
print("🖐️ Inicia detección. Presiona ESPACIO para confirmar letra, 'v' para decir palabra, 'ESC' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    letra_predicha = ""
    puntos = []

    if result.multi_hand_landmarks:
        cantidad_manos = len(result.multi_hand_landmarks)

        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

        if cantidad_manos == 1:
            puntos.extend([0.0] * 63)
        elif cantidad_manos != 2:
            continue  # ignorar si no es 1 o 2 manos

        if len(puntos) == 126:
            entrada = np.array(puntos).reshape(1, -1)
            pred = model.predict(entrada)
            letra = encoder.inverse_transform([np.argmax(pred)])[0].strip()
            buffer_predicciones.append(letra)

            if buffer_predicciones.count(letra) >= umbral_consistencia:
                letra_predicha = letra

                if letra == letra_anterior:
                    if tiempo_inicio and (time.time() - tiempo_inicio >= 2.0):
                        palabra_actual += letra
                        print(f"✅ Letra confirmada automáticamente: {letra}")
                        print(f"Letra cruda detectada: '{letra}'")
                        buffer_predicciones.clear()
                        tiempo_inicio = None
                        letra_anterior = ""
                else:
                    letra_anterior = letra
                    tiempo_inicio = time.time()

            letra_mostrada = letra

        # Dibujar manos
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar interfaz
    alto, ancho, _ = frame.shape
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
        cv2.rectangle(panel, (barra_x, barra_y),
                      (barra_x + int(barra_ancho * progreso), barra_y + barra_altura), barra_color, -1)

    frame = np.hstack((frame, panel))

    # Panel inferior
    cv2.rectangle(frame, (0, alto - 80), (ancho + panel_ancho, alto), (255, 255, 255), -1)
    cv2.line(frame, (0, alto - 80), (ancho + panel_ancho, alto - 80), (100, 100, 100), 2)

    (tw, th), _ = cv2.getTextSize(palabra_actual, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    x_text = int((ancho + panel_ancho - tw) / 2)
    cv2.putText(frame, palabra_actual, (x_text, alto - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Mostrar
    cv2.namedWindow("Traductor IA", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Traductor IA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Traductor IA", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32 and letra_mostrada:
        palabra_actual += letra_mostrada
        print(f"✅ Letra confirmada: {letra_mostrada}")
        buffer_predicciones.clear()
    elif key == ord('v') and palabra_actual:
        print(f"🗣️ Hablando: {palabra_actual}")
        voz.say(palabra_actual)
        voz.runAndWait()
        palabra_actual = ""
        letra_mostrada = ""

cap.release()
cv2.destroyAllWindows()