import cv2
import mediapipe as mp
import numpy as np
from sqlalchemy import create_engine, Table, MetaData, insert

# Conexión a PostgreSQL
DB_USER = "usr_traductor"
DB_PASSWORD = "NVILfbkCxpEqMNWLAJBY9JxnDSSaUNXy"
DB_HOST = "dpg-d044scruibrs73aoldfg-a.oregon-postgres.render.com"
DB_PORT = "5432"
DB_NAME = "lenguaje_senias"

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
metadata = MetaData()
metadata.reflect(bind=engine)
tabla = metadata.tables['datos_senas']

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("\nHaz la seña y presiona la tecla correspondiente (A-Z) para capturarla. Presiona 'ESC' para salir.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(img_rgb)

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Captura de Manos", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:  # ESC para salir
        break

    # Verificar si se presionó una letra A-Z o a-z
    if (48 <= tecla <= 57) or (65 <= tecla <= 90) or (97 <= tecla <= 122):

        letra = chr(tecla).upper()  # Convertimos cualquier tecla a MAYÚSCULA

        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                coordenadas = []
                for punto in hand_landmarks.landmark:
                    coordenadas.extend([punto.x, punto.y, punto.z])

                if len(coordenadas) == 63:
                    valores = {f'x{i//3}': coordenadas[i] for i in range(0, 63, 3)}
                    valores.update({f'y{i//3}': coordenadas[i] for i in range(1, 63, 3)})
                    valores.update({f'z{i//3}': coordenadas[i] for i in range(2, 63, 3)})
                    valores['label'] = letra
                    with engine.begin() as conn:
                        conn.execute(insert(tabla).values(valores))
                    print(f"✅ Captura guardada para la letra '{letra}'.")
                else:
                    print("❌ Error: cantidad incorrecta de puntos.")
        else:
            print("❌ No se detectaron manos, intenta de nuevo.")

cap.release()
cv2.destroyAllWindows()
