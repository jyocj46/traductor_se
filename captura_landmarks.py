import cv2
import mediapipe as mp
import numpy as np
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

# Reflejar la tabla de señas (una mano)
tabla_senas = metadata.tables['datos_senas']

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("\n[CAPTURA LETRAS] Usa UNA MANO para hacer la seña. Presiona una tecla (A-Z, 0-9), ESPACIO o BACKSPACE para capturar. ESC para salir.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(img_rgb)

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Captura LETRAS (una mano)", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:  # ESC para salir
        break
    elif tecla == 8:  # BACKSPACE
        letra = "DEL"
    elif tecla == 32:  # ESPACIO
        letra = "ESP"
    elif (48 <= tecla <= 57) or (65 <= tecla <= 90) or (97 <= tecla <= 122):  # Letras y números
        letra = chr(tecla).upper()
    else:
        continue  # Ignorar otras teclas

    if resultados.multi_hand_landmarks:
        cantidad_manos = len(resultados.multi_hand_landmarks)

        if cantidad_manos == 1:
            mano = resultados.multi_hand_landmarks[0]
            coordenadas = []
            for punto in mano.landmark:
                coordenadas.extend([punto.x, punto.y, punto.z])

            if len(coordenadas) == 63:
                valores = {f'x{i//3}': coordenadas[i] for i in range(0, 63, 3)}
                valores.update({f'y{i//3}': coordenadas[i] for i in range(1, 63, 3)})
                valores.update({f'z{i//3}': coordenadas[i] for i in range(2, 63, 3)})
                valores['label'] = letra
                with engine.begin() as conn:
                    conn.execute(insert(tabla_senas).values(valores))
                print(f"✅ Captura guardada en datos_senas para '{letra}'.")
            else:
                print("❌ Error: cantidad incorrecta de puntos en una mano.")
        else:
            print("❌ Solo se permite UNA mano. Intenta nuevamente.")
    else:
        print("❌ No se detectaron manos, intenta de nuevo.")

cap.release()
cv2.destroyAllWindows()
