import cv2
import numpy as np
import glob

# Caminho para as imagens
imagens = glob.glob("*.jpg")  # ajuste o caminho conforme necessário

for img_path in imagens:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # Detecção de bordas
    edges = cv2.Canny(blurred, 50, 150)

    # Detectar linhas com HoughLinesP
    linhas = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Detectar círculos com HoughCircles
    circulos = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=50, param2=60, minRadius=5, maxRadius=100)
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for c in circulos[0, :]:
            cv2.circle(img, (c[0], c[1]), c[2], (255, 0, 0), 2)
            cv2.circle(img, (c[0], c[1]), 2, (0, 0, 255), 3)

    # Mostrar resultado
    cv2.imshow(f"Resultado - {img_path}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
