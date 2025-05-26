import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Cargar modelo entrenado y codificador
model = load_model("modelo_senas.h5")
encoder = joblib.load("label_encoder.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


print("🚀 Iniciando predicción. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    prediccion = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            datos = []
            for lm in hand_landmarks.landmark:
                datos.extend([lm.x, lm.y, lm.z])

            if len(datos) == 63:
                X_input = np.array(datos).reshape(1, -1)
                prediction = model.predict(X_input)
                index = np.argmax(prediction)
                prediccion = encoder.inverse_transform([index])[0]

    # Mostrar resultado
    cv2.putText(frame, f"Prediccion: {prediccion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Traductor de Senas IA", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()