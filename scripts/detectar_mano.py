import cv2
import mediapipe as mp

# Inicializar la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Iniciar la cámara
cap = cv2.VideoCapture(0)

# Establecer resolución HD (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Crear ventana en modo pantalla completa
cv2.namedWindow("Deteccion de Mano", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Deteccion de Mano", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Si detecta mano, dibujar los puntos clave
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar imagen en pantalla completa
    cv2.imshow("Deteccion de Mano", img)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
