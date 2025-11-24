import numpy as np
import cv2

def main():
    cam = cv2.VideoCapture("furkan/video.mp4")
    if not cam.isOpened():
        print("Video açılamadı!")
        return

    orig_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MOG2 arka plan çıkarıcı
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=50,
        varThreshold=16,
        detectShadows=True
    )

    # Morfolojik filtreler
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    out = cv2.VideoWriter(
        'result_mog2_only.avi',
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        30,
        (orig_w, orig_h)
    )

    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Video bitti!")
            break

        frame_count += 1

        # ========== 1. MOG2 Uygula ==========
        fg_mask = backSub.apply(frame)

        # Gölge temizleme (gölge = 127 civarı)
        fg_mask_no_shadow = cv2.threshold(
            fg_mask, 200, 255, cv2.THRESH_BINARY
        )[1]

        # ========== 2. Gürültü Temizleme ==========
        fg_mask_clean = cv2.morphologyEx(
            fg_mask_no_shadow,
            cv2.MORPH_OPEN,
            kernel_open
        )
        fg_mask_clean = cv2.morphologyEx(
            fg_mask_clean,
            cv2.MORPH_CLOSE,
            kernel_close
        )

        # ========== 3. Kontur Bul ==========
        contours, _ = cv2.findContours(
            fg_mask_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        img_result = frame.copy()

        # ========== 4. Tespit — Bounding Box ==========
        for c in contours:
            area = cv2.contourArea(c)

            # Çok küçük veya çok büyük objeleri alma
            if area < 500 or area > 50000:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Boyut filtresi
            if w < 400 and h < 600:
                cv2.rectangle(img_result, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                cv2.putText(img_result, f"{int(area)}",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        # ========== 5. Görselleştirme ==========
        cv2.putText(img_result, f"MOG2 ONLY | Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        cv2.imshow("MOG2 Mask Clean", fg_mask_clean)
        cv2.imshow("Final Detection", img_result)

        out.write(img_result)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if frame_count % 30 == 0:
            print(f"Frame {frame_count} işlendi.")

    cam.release()
    out.release()
    cv2.destroyAllWindows()
    print("Tamamlandı.")

if __name__ == "__main__":
    main()
