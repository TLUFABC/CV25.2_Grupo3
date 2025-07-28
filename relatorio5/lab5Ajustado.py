import cv2 as cv
import numpy as np

MIN_MATCH_COUNT = 10

cams = []
for i in range(10):
    camera = cv.VideoCapture(i)
    if camera.isOpened():
        cams.append(i)
    camera.release()


# Inicia captura das duas câmeras (ajuste os índices conforme necessário)
cap_left = cv.VideoCapture(cams[0])
cap_right = cv.VideoCapture(cams[1])

# Verifica se as câmeras foram abertas corretamente
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Erro ao abrir as câmeras")
    exit()

# Inicia o detector SIFT
sift = cv.SIFT_create()

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

while True:
    # Captura os quadros das duas câmeras
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()
    
    if not ret1 or not ret2:
        print("Erro ao capturar quadros")
        break

    # Converte os quadros para escala de cinza
    gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

    # Detecta os keypoints e os descritores
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)

    if des1 is not None and des2 is not None:
        matches = flann.knnMatch(des1, des2, k=2)

        # Filtra os bons matches com o teste de razão de Lowe
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        matchesMask = None
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = gray_left.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            frame_right = cv.polylines(frame_right, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)

        else:
            print("Matches insuficientes - {}/{}".format(len(good), MIN_MATCH_COUNT))

        # Desenha os matches
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        img_matches = cv.drawMatches(frame_left, kp1, frame_right, kp2, good, None, **draw_params)

        cv.imshow("Matches", img_matches)

    # Encerra com 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap_left.release()
cap_right.release()
cv.destroyAllWindows()
