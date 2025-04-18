import cv2
import mediapipe as mp
import pandas as pd
import os
from pynput import keyboard

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables
data = []
current_label = None
output_file = "datos_letras.csv"

def on_press(key):
    global current_label
    try:
        k = key.char.upper()
        if k.isalpha():
            current_label = k
            print(f"Etiqueta actual: {current_label}")
    except:
        pass

# Escuchar teclas
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Captura desde la cámara
cap = cv2.VideoCapture(0)

print("Presiona una letra (A-Z) para guardar la seña.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_label:
                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                row.append(current_label)
                data.append(row)
                print(f"Guardado: {current_label}")
                current_label = None

    cv2.imshow("Captura de Señales", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

# Guardar en CSV
columns = [f'{e}{i}' for i in range(21) for e in ['x', 'y', 'z']] + ['label']
df = pd.DataFrame(data, columns=columns)

if os.path.exists(output_file):
    df.to_csv(output_file, mode='a', header=False, index=False)
else:
    df.to_csv(output_file, index=False)

cap.release()
cv2.destroyAllWindows()
print(f"\nDataset guardado en: {output_file}")
