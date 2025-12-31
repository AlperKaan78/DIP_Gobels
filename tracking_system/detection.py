"""
detection.py - MOG2 Background Subtraction + Blob Detection
İlk versiyon: Basit test için
"""

import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, 
                 mog2_history=500,
                 mog2_var_threshold=16,
                 detect_shadows=True):
        """
        Args:
            mog2_history: MOG2 geçmiş frame sayısı
            mog2_var_threshold: Hassasiyet (düşük = daha hassas)
            detect_shadows: Gölge tespiti
        """
        # MOG2 arka plan çıkarıcı
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_var_threshold,
            detectShadows=detect_shadows
        )
        
        # Morfolojik kerneller
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        print(f"✅ Detector initialized (history={mog2_history}, varThresh={mog2_var_threshold})")
    
    def detect(self, frame):
        """
        Bir frame'de nesne tespiti yapar
        
        Returns:
            blobs: List of detected blobs
                   Her blob: {'bbox': (x,y,w,h), 'center': (cx,cy), 'area': area}
            fg_mask: Temizlenmiş foreground mask (görselleştirme için)
        """
        # 1. MOG2 uygula
        fg_mask = self.backSub.apply(frame)
        
        # 2. Gölge temizleme (gölge=127, foreground=255)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # 3. Morphology: noise + hole filling
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        # 4. Kontur bul
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Blob'ları filtrele ve ölçümlerini çıkar
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Alan filtresi
            if area < 500 or area > 50000:
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Boyut filtresi
            if w > 400 or h > 600:
                continue
            
            # Merkez noktası
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            blobs.append({
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': int(area),
                'contour': cnt  # Opsiyonel: daha sonra kullanılabilir
            })
        
        return blobs, fg_mask
    
    def draw_detections(self, frame, blobs):
        """
        Tespit edilen blobs'ları frame üzerine çizer
        """
        result = frame.copy()
        
        for blob in blobs:
            x, y, w, h = blob['bbox']
            cx, cy = blob['center']
            area = blob['area']
            
            # Bounding box
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Merkez noktası
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            
            # Alan bilgisi
            cv2.putText(result, f"{area}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Tespit sayısı
        cv2.putText(result, f"Detections: {len(blobs)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result


# ============================================================
# TEST KODU
# ============================================================
def test_detector():
    """
    Basit test: detection.py'nin çalışıp çalışmadığını kontrol et
    """
    import os
    
    print("="*60)
    print("DETECTION MODULE TEST")
    print("="*60)
    
    # Video yolu
    video_path = "Data/rouen_video.avi"
    
    if not os.path.exists(video_path):
        print(f"❌ Video bulunamadı: {video_path}")
        print("💡 Video yolunu düzenle ve tekrar dene")
        return
    
    # Video aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Video açılamadı!")
        return
    
    # Detector oluştur
    detector = ObjectDetector()
    
    frame_count = 0
    
    print("\n🎬 Video işleniyor...")
    print("   ESC: Çıkış")
    print("   SPACE: Duraklat/Devam")
    print("="*60 + "\n")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video bitti!")
                break
            
            frame_count += 1
            
            # 🔍 DETECTION
            blobs, fg_mask = detector.detect(frame)
            
            # 🎨 VISUALIZATION
            result = detector.draw_detections(frame, blobs)
            
            # Frame sayısı ekle
            cv2.putText(result, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Göster
            cv2.imshow("1. Original", frame)
            cv2.imshow("2. FG Mask (cleaned)", fg_mask)
            cv2.imshow("3. Detection Result", result)
            
            # Her 30 frame'de konsola bilgi
            if frame_count % 30 == 0:
                print(f"Frame {frame_count:4d} | Detections: {len(blobs):2d}")
        
        # Klavye kontrolü
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == 27:  # ESC
            print("\n⛔ Kullanıcı tarafından durduruldu")
            break
        elif key == 32:  # SPACE
            paused = not paused
            print("⏸️  DURAKLAT" if paused else "▶️  DEVAM")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Toplam {frame_count} frame işlendi")
    print("="*60)


if __name__ == "__main__":
    test_detector()