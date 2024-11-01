import cv2
import mediapipe as mp
import math

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Função para detectar o gesto "OK"
def is_ok_gesture(hand_landmarks):
    # Pega as coordenadas dos landmarks do polegar e indicador
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Coordenadas dos outros dedos
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calcula a distância entre o polegar e o indicador
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

    # Condições para o gesto "OK":
    # 1. O polegar e o indicador estão próximos.
    # 2. Os outros dedos estão estendidos (acima de certa posição).
    if distance < 0.05 and middle_tip.y < index_tip.y and ring_tip.y < index_tip.y and pinky_tip.y < index_tip.y:
        return True
    return False

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
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os pontos de referência nas mãos
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verifica se o gesto de "OK" foi feito
            if is_ok_gesture(hand_landmarks):
                cv2.putText(img, "OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Exibe o frame com os pontos mapeados e a detecção do gesto
    cv2.imshow("Image", img)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
