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
    edges = cv.Canny(blurred_image, 100, 200)
    
    # Encontra os contornos
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Converte a imagem para colorida para desenhar os contornos
    image_contours = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)
    
    # Quantidade de pixels extras na parte inferior
    extra_pixels = 10  # Valor em pixels para adicionar ao recorte inferior
    
    # Encontra o maior contorno (assumindo que é a borda externa)
    main_contour = max(contours, key=cv.contourArea)
    
    # Desenha apenas o contorno externo principal
    cv.drawContours(image_contours, [main_contour], 0, (0, 255, 0), 2)
    
    # Obtém o retângulo delimitador do contorno principal e ajusta a altura
    x, y, w, h = cv.boundingRect(main_contour)
    h += extra_pixels  # Adiciona pixels extras na parte inferior
    
    # Recorta a imagem usando o contorno principal com a borda inferior expandida
    cropped_image = resized_image[y:y+h, x:x+w]
    
    # Salva a imagem recortada
    cv.imwrite("img/lousa_recortada.jpg", cropped_image)
    
    # Exibe a imagem com o contorno principal
    cv.imshow("Lousa Detectada (Borda Externa)", image_contours)
    cv.imshow("Imagem Recortada", cropped_image)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
