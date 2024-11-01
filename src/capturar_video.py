import cv2

# Inicializa a captura de vídeo (0 indica a câmera padrão)
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Loop para capturar os frames
while True:
    ret, frame = cap.read()  # Lê um frame da câmera
    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Exibe o frame capturado
    cv2.imshow('Camera', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
