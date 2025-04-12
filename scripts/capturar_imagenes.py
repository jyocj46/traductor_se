import cv2
import mediapipe as mp
import os

letra = "A"  # Cambia según la letra que vayas a capturar
directorio = f"dataset/{letra}"
os.makedirs(directorio, exist_ok=True)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
contador = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(image_rgb)

    if resultados.multi_hand_landmarks:
        # Guardar imagen si hay manos
        cv2.imwrite(f"{directorio}/{contador}.jpg", frame)
        contador += 1
        print(f"Imagen {contador} guardada.")

    cv2.imshow("Captura - Presiona Q para salir", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
