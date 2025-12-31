import cv2
import numpy as np

# ============================
# Kalman oluşturma fonksiyonu
# ============================
def create_kalman(x, y, vx, vy):
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    kf.statePost = np.array([[x], [y], [vx], [vy]], np.float32)

    return kf


# ============================
# 2 Kalman Filter
# ============================
kalman1 = create_kalman(100, 200, 2, 1)
kalman2 = create_kalman(700, 400, -2, -1)

# Gerçek pozisyonlar
obj1 = [100, 200]
obj2 = [700, 400]

vel1 = [2, 1]
vel2 = [-2, -1]

canvas = np.zeros((600, 900, 3), dtype=np.uint8)
frame = 0

while True:
    canvas[:] = 0
    frame += 1

    # ============================
    # GERÇEK HAREKET
    # ============================
    obj1[0] += vel1[0]
    obj1[1] += vel1[1]

    obj2[0] += vel2[0]
    obj2[1] += vel2[1]

    # ============================
    # ÖLÇÜMLER (occlusion simülasyonu)
    # ============================
    measure1 = True
    measure2 = True

    # Nesne 1 için ölçüm kaybı
    if 120 < frame < 180:
        measure1 = False

    # Nesne 2 için ölçüm kaybı
    if 220 < frame < 270:
        measure2 = False

    # ============================
    # KALMAN 1
    # ============================
    pred1 = kalman1.predict()
    p1x, p1y = int(pred1[0]), int(pred1[1])

    if measure1:
        meas1 = np.array([
            [np.float32(obj1[0] + np.random.randint(-10, 10))],
            [np.float32(obj1[1] + np.random.randint(-10, 10))]
        ])
        kalman1.correct(meas1)
        cv2.circle(canvas, (int(meas1[0]), int(meas1[1])), 5, (0, 0, 255), -1)

    # ============================
    # KALMAN 2
    # ============================
    pred2 = kalman2.predict()
    p2x, p2y = int(pred2[0]), int(pred2[1])

    if measure2:
        meas2 = np.array([
            [np.float32(obj2[0] + np.random.randint(-10, 10))],
            [np.float32(obj2[1] + np.random.randint(-10, 10))]
        ])
        kalman2.correct(meas2)
        cv2.circle(canvas, (int(meas2[0]), int(meas2[1])), 5, (0, 0, 255), -1)

    # ============================
    # ÇİZİMLER
    # ============================
    cv2.circle(canvas, (obj1[0], obj1[1]), 6, (0, 255, 0), -1)
    cv2.circle(canvas, (obj2[0], obj2[1]), 6, (0, 255, 0), -1)

    cv2.circle(canvas, (p1x, p1y), 6, (255, 0, 0), -1)
    cv2.circle(canvas, (p2x, p2y), 6, (255, 0, 0), -1)

    cv2.putText(canvas, "Green: True | Red: Measurement | Blue: Kalman",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(canvas, f"Frame: {frame}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Multi Kalman Tracking Demo", canvas)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
