import cv2 as cv
import numpy as np

# Carrega a imagem em escala de cinza
image = cv.imread("img/lousa_with_teste.jpg", 0)

# Verifica se a imagem foi carregada
if image is None:
    print("Erro: Imagem não encontrada.")
else:
    # Redimensiona a imagem para 40% do tamanho original
    scale_percent = 30
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    
    # Aplica um desfoque para reduzir ruídos
    blurred_image = cv.GaussianBlur(resized_image, (5, 5), 0)
    
    # Detecta bordas com Canny
    edges = cv.Canny(blurred_image, 50, 150)
    
    # Encontra os contornos
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Converte a imagem para colorida para desenhar os contornos
    image_contours = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)
    
    # Percorre os contornos e procura retângulos
    for contour in contours:
        # Aproxima o contorno para detectar polígonos
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        # Se o contorno tem 4 pontos, pode ser um retângulo
        if len(approx) == 4:
            # Desenha o contorno do retângulo na imagem
            cv.drawContours(image_contours, [approx], 0, (0, 255, 0), 2)
            # Exibe a imagem com o contorno da lousa
            cv.imshow("Lousa Detectada", image_contours)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
