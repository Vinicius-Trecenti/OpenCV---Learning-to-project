import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Função que será chamada quando a mão for detectada
def on_hand_detected():
    print("Mão detectada!")

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

    # Verifica se uma mão foi detectada
    if results.multi_hand_landmarks:
        on_hand_detected()  # Chama a função personalizada

        # Para cada mão detectada, desenha os pontos de referência
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Exibe o frame com os pontos mapeados
    cv2.imshow("Image", img)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
