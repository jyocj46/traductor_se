import cv2
import mediapipe as mp
import numpy as np
import time
import sys
from sqlalchemy import create_engine, Table, MetaData, insert

# Conexión a PostgreSQL
DB_USER = "usr_traductor"
DB_PASSWORD = "oUxJEZ59sw6bPZdBBPfSvJNUBzTlkQ5f"
DB_HOST = "dpg-d0mjg0e3jp1c738dep3g-a.oregon-postgres.render.com"
DB_PORT = "5432"
DB_NAME = "lenguaje_senias_g7z9"

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
metadata = MetaData()
metadata.reflect(bind=engine)

tabla_numeros = metadata.tables['datos_numeros']

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("\n[CAPTURA NÚMEROS] Usa DOS MANOS para hacer la seña. Presiona ESC para salir.\n")

label = input("👉 ¿Qué deseas capturar? (ej. 6, 7, 8, 9, UWU): ").strip().upper()
if not label:
    print("❌ Entrada vacía. Terminando programa.")
    sys.exit()

captura_id = 1
manos_detectadas = False
inicio_tiempo = None
duracion_objetivo = 2  # segundos

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al acceder a la cámara.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(img_rgb)

    frame_display = frame.copy()

    cantidad_manos = len(resultados.multi_hand_landmarks) if resultados.multi_hand_landmarks else 0

    if cantidad_manos == 2:
        if not manos_detectadas:
            manos_detectadas = True
            inicio_tiempo = time.time()
        else:
            tiempo_transcurrido = time.time() - inicio_tiempo
            progreso = int((tiempo_transcurrido / duracion_objetivo) * 300)
            cv2.rectangle(frame_display, (10, 10), (310, 40), (200, 200, 200), 2)
            cv2.rectangle(frame_display, (10, 10), (10 + min(progreso, 300), 40), (0, 255, 0), -1)
            cv2.putText(frame_display, f"Captura en {max(0, round(duracion_objetivo - tiempo_transcurrido, 1))}s", 
                        (320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if tiempo_transcurrido >= duracion_objetivo:
                coordenadas = []
                for mano in resultados.multi_hand_landmarks:
                    for punto in mano.landmark:
                        coordenadas.extend([punto.x, punto.y, punto.z])

                if len(coordenadas) == 126:
                    valores = {f'x{i//3}': coordenadas[i] for i in range(0, 126, 3)}
                    valores.update({f'y{i//3}': coordenadas[i] for i in range(1, 126, 3)})
                    valores.update({f'z{i//3}': coordenadas[i] for i in range(2, 126, 3)})
                    valores['label'] = label
                    with engine.begin() as conn:
                        conn.execute(insert(tabla_numeros).values(valores))
                    print(f"✅ Captura {captura_id} guardada con label '{label}'.\n")
                    captura_id += 1
                else:
                    print("❌ Error: cantidad incorrecta de puntos.\n")

                manos_detectadas = False
                inicio_tiempo = None

    else:
        manos_detectadas = False
        inicio_tiempo = None

    # Dibujar manos
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Captura NUMEROS", frame_display)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
