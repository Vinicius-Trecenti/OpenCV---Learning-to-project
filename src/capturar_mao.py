import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Captura de vídeo da câmera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Lê o frame da câmera
    if not success:
        break

    # Converte a imagem para RGB (MediaPipe usa imagens RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar as mãos
    results = hands.process(img_rgb)

    # Se mãos forem detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os pontos de referência nas mãos
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Opcional: Mostrar as coordenadas dos pontos de referência
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"ID: {id}, Pos: ({cx}, {cy})")

    # Exibe o frame com os pontos mapeados
    cv2.imshow("Image", img)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
