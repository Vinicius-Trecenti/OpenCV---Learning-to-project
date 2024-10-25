import cv2
import mediapipe as mp
import math

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Permite a detecção de duas mãos
mp_draw = mp.solutions.drawing_utils

# Função para calcular a distância entre dois pontos (landmarks)
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Função para detectar o gesto de "Picture" com as duas mãos
def is_picture_gesture(hand_landmarks1, hand_landmarks2):
    # Pega as coordenadas dos polegares e indicadores de ambas as mãos
    thumb_tip_1 = hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip_1 = hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_tip_2 = hand_landmarks2.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip_2 = hand_landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Calcula as distâncias entre polegares e indicadores de cada mão
    thumb_to_index_dist_1 = calculate_distance(thumb_tip_1, index_tip_1)
    thumb_to_index_dist_2 = calculate_distance(thumb_tip_2, index_tip_2)

    # Calcula a distância entre o polegar da mão esquerda e o indicador da mão direita (e vice-versa)
    thumb1_to_index2_dist = calculate_distance(thumb_tip_1, index_tip_2)
    thumb2_to_index1_dist = calculate_distance(thumb_tip_2, index_tip_1)

    # Verifica se os polegares e indicadores de ambas as mãos estão formando uma moldura retangular
    if (thumb_to_index_dist_1 > 0.1 and thumb_to_index_dist_2 > 0.1) and (thumb1_to_index2_dist < 0.3 and thumb2_to_index1_dist < 0.3):
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

    # Verifica se há duas mãos detectadas
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand_landmarks1 = results.multi_hand_landmarks[0]
        hand_landmarks2 = results.multi_hand_landmarks[1]

        # Desenha os pontos de referência nas duas mãos
        mp_draw.draw_landmarks(img, hand_landmarks1, mp_hands.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(img, hand_landmarks2, mp_hands.HAND_CONNECTIONS)

        # Verifica se o gesto de "Picture" foi feito
        if is_picture_gesture(hand_landmarks1, hand_landmarks2):
            cv2.putText(img, "Picture Gesture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Exibe o frame com os pontos mapeados e a detecção do gesto
    cv2.imshow("Image", img)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
