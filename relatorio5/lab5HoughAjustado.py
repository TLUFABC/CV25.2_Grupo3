import cv2
import numpy as np

cams = []
for i in range(10):
    camera = cv2.VideoCapture(i)
    if camera.isOpened():
        cams.append(i)
    camera.release()

# Inicializar duas webcams (ajuste os índices conforme necessário)
cam_esq = cv2.VideoCapture(cams[0])
cam_dir = cv2.VideoCapture(cams[1])

while True:
    ret1, frame1 = cam_esq.read()
    ret2, frame2 = cam_dir.read()

    if not ret1 or not ret2:
        print("Erro ao capturar vídeo")
        break

    def processar(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(blurred, 50, 150)

        # Linhas
        linhas = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 50, 10)
        if linhas is not None:
            for linha in linhas:
                x1, y1, x2, y2 = linha[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Círculos
        circulos = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 40,
                                    param1=50, param2=40, minRadius=70, maxRadius=150)
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for c in circulos[0, :]:
                cv2.circle(frame, (c[0], c[1]), c[2], (255, 0, 0), 2)
                cv2.circle(frame, (c[0], c[1]), 2, (0, 0, 255), 3)

        return frame

    # Processar e mostrar
    frame1_proc = processar(frame1)
    frame2_proc = processar(frame2)

    cv2.imshow("Camera Esquerda", frame1_proc)
    cv2.imshow("Camera Direita", frame2_proc)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_esq.release()
cam_dir.release()
cv2.destroyAllWindows()
