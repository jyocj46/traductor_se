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
voz.setProperty('rate', 140)  # velocidad
voz.setProperty('voice', voz.getProperty('voices')[0].id)  # puedes ajustar el idioma si es necesario

# Cargar modelo y codificador
model = load_model("modelo_senas.h5")
encoder = joblib.load("label_encoder.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
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

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            puntos = []
            for lm in hand_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            if len(puntos) == 63:
                entrada = np.array(puntos).reshape(1, -1)
                pred = model.predict(entrada)
                letra = encoder.inverse_transform([np.argmax(pred)])[0]
                buffer_predicciones.append(letra)

                # Si hay consistencia suficiente
                if buffer_predicciones.count(letra) >= umbral_consistencia:
                    letra_predicha = letra

                    if letra == letra_anterior:
                        if tiempo_inicio and (time.time() - tiempo_inicio >= 2.0):
                            palabra_actual += letra
                            print(f"✅ Letra confirmada automáticamente: {letra}")
                            buffer_predicciones.clear()
                            tiempo_inicio = None  # Reiniciar
                            letra_anterior = ""   # Evitar duplicados
                    else:
                        letra_anterior = letra
                        tiempo_inicio = time.time()
    
                    letra_mostrada = letra


    # Mostrar texto
    # Tamaño del frame
    alto, ancho, _ = frame.shape

    # Crear panel lateral blanco con borde
    panel_ancho = 300
    panel = np.ones((alto, panel_ancho, 3), dtype=np.uint8) * 255
    cv2.rectangle(panel, (0, 0), (panel_ancho-1, alto-1), (100, 100, 100), 2)  # borde gris

    # Título
    cv2.putText(panel, "Letra detectada", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    # Letra detectada centrada
    letra_grande = letra_mostrada if letra_mostrada else "-"
    (tw, th), _ = cv2.getTextSize(letra_grande, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)
    x_text = int((panel_ancho - tw) / 2)
    y_text = int((alto + th) / 2)
    cv2.putText(panel, letra_grande, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 102, 204), 8)

    # Unir el panel al frame
    frame = np.hstack((frame, panel))

    # Dibujar panel inferior para la palabra formada
    cv2.rectangle(frame, (0, alto - 80), (ancho + panel_ancho, alto), (255, 255, 255), -1)
    cv2.line(frame, (0, alto - 80), (ancho + panel_ancho, alto - 80), (100, 100, 100), 2)

    # Texto centrado
    (tw, th), _ = cv2.getTextSize(palabra_actual, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    x_text = int((ancho + panel_ancho - tw) / 2)
    cv2.putText(frame, palabra_actual, (x_text, alto - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Mostrar en pantalla completa
    cv2.namedWindow("Traductor IA", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Traductor IA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Traductor IA", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32 and letra_mostrada:  # Espacio: confirmar letra
        palabra_actual += letra_mostrada
        print(f"✅ Letra confirmada: {letra_mostrada}")
        buffer_predicciones.clear()
    elif key == ord('v') and palabra_actual:  # 'v': decir palabra
        print(f"🗣️ Hablando: {palabra_actual}")
        voz.say(palabra_actual)
        voz.runAndWait()
        palabra_actual = ""
        letra_mostrada = ""

cap.release()
cv2.destroyAllWindows()