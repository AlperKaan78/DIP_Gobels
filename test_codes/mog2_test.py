import cv2
import numpy as np

def test_mog2():
    # Video aç
    # path_of_video = "Data/atrium.avi"
    path_of_video = "Data/mobese_3.mp4"
    cam = cv2.VideoCapture(path_of_video)
    
    if not cam.isOpened():
        print("HATA: Video açılamadı!")
        print("'data/atrium.avi' dosyasının var olduğundan emin olun.")
        return
    
    # Video bilgilerini al
    width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cam.get(cv2.CAP_PROP_FPS))
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 50)
    print("MOG2 BACKGROUND SUBTRACTION TEST")
    print("=" * 50)
    print(f"Video boyutu: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Toplam frame: {total_frames}")
    print("\nKontroller:")
    print("  ESC - Çıkış")
    print("  SPACE - Duraklat/Devam")
    print("=" * 50)
    
    # MOG2 arka plan çıkarıcı oluştur
    mog2_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,           # Kaç frame'lik geçmiş
        varThreshold=24,       # Hassasiyet (düşük=daha hassas)
        detectShadows=True     # Gölge tespiti
    )
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cam.read()
            
            if not ret:
                print("\nVideo bitti!")
                break
            
            frame_count += 1
            
            # MOG2 uygula
            fg_mask = mog2_subtractor.apply(frame)
            
            # Gölgeleri kaldır (200 den küçük olanları 0 yapıyor yani gölgeleri siliyor)
            fg_mask_no_shadow = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
            
            # Gürültü temizleme
            kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            
            fg_mask_clean = cv2.morphologyEx(fg_mask_no_shadow, cv2.MORPH_OPEN,  kernel_open)
            fg_mask_clean = cv2.morphologyEx(fg_mask_clean,     cv2.MORPH_CLOSE, kernel_close)
            
            # Arka plan modelini al (opsiyonel)
            bg_image = mog2_subtractor.getBackgroundImage()
            
            # Frame numarasını göster
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Görselleştirme
            cv2.imshow('1. Original Frame', frame)
            cv2.imshow('2. MOG2 Raw (Gray=Shadow)', fg_mask)
            cv2.imshow('3. MOG2 Without Shadow', fg_mask_no_shadow)
            cv2.imshow('4. MOG2 Cleaned', fg_mask_clean)
            
            if bg_image is not None:
                cv2.imshow('5. Background Model', bg_image)
            
            # Her 30 frame'de konsola bilgi yazdır
            if frame_count % 30 == 0:
                print(f"Frame processed: {frame_count}/{total_frames} "
                      f"(%{int(frame_count/total_frames*100)})")
        
        # Klavye kontrolü
        key = cv2.waitKey(30) & 0xff
        
        if key == 27:  # ESC
            print("\nHalted by user!")
            break
        elif key == 32:  # SPACE
            paused = not paused
            print("HALTED" if paused else "DEVAM EDİYOR")
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"\nTest is done!")
    print(f"Total {frame_count} frame was processed.")


if __name__ == "__main__":
    test_mog2()