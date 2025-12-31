"""
optical_flow.py - Dense Optical Flow + Blob Refinement
Paper: Beaupre et al. 2018 - Improving Multiple Object Tracking with Optical Flow
"""

import cv2
import numpy as np

class OpticalFlowProcessor:
    def __init__(self):
        """
        Farneback Dense Optical Flow parametreleri
        """
        # Farneback parametreleri (ofd2.py'den optimize edilmiş)
        self.flow_params = {
            'pyr_scale': 0.5,      # Pyramid scale
            'levels': 3,           # Pyramid levels
            'winsize': 7,          # Averaging window size
            'iterations': 3,       # Iterations at each pyramid level
            'poly_n': 5,           # Pixel neighborhood size
            'poly_sigma': 1.1,     # Gaussian std for polynomial expansion
            'flags': 0
        }
        
        self.prev_gray = None
        
        print("✅ OpticalFlowProcessor initialized")
    
    def compute_flow(self, frame):
        """
        İki frame arası dense optical flow hesaplar
        
        Args:
            frame: Güncel frame (BGR)
        
        Returns:
            flow: Optical flow field (H, W, 2) - (dx, dy) for each pixel
            flow_magnitude: Flow magnitude (speed) image
            None if this is first frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None
        
        # Dense optical flow hesapla
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, 
            gray, 
            None,
            **self.flow_params
        )
        
        # Flow magnitude (hız) hesapla
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        flow_magnitude = np.sqrt(fx**2 + fy**2)
        
        self.prev_gray = gray
        
        return flow, flow_magnitude
    
    def get_blob_motion(self, blob, flow):
        """
        Bir blob'un ortalama motion vektörünü hesaplar
        
        Args:
            blob: {'bbox': (x,y,w,h), 'center': (cx,cy), ...}
            flow: Optical flow field
        
        Returns:
            motion: {'vx': mean_vx, 'vy': mean_vy, 'speed': mean_speed, 'angle': mean_angle}
        """
        if flow is None:
            return {'vx': 0, 'vy': 0, 'speed': 0, 'angle': 0}
        
        x, y, w, h = blob['bbox']
        
        # Blob bölgesindeki flow'u al
        roi_flow = flow[y:y+h, x:x+w]
        
        # Ortalama motion vektörü
        mean_vx = np.mean(roi_flow[:, :, 0])
        mean_vy = np.mean(roi_flow[:, :, 1])
        
        # Hız (magnitude)
        mean_speed = np.sqrt(mean_vx**2 + mean_vy**2)
        
        # Yön (angle) - radyan cinsinden
        mean_angle = np.arctan2(mean_vy, mean_vx)
        
        return {
            'vx': float(mean_vx),
            'vy': float(mean_vy),
            'speed': float(mean_speed),
            'angle': float(mean_angle)
        }
    
    def add_motion_to_blobs(self, blobs, flow):
        """
        Her blob'a motion bilgisi ekler
        
        Args:
            blobs: Detection'dan gelen blob listesi
            flow: Optical flow field
        
        Returns:
            blobs_with_motion: Her blob'a 'motion' key'i eklenmiş hali
        """
        if flow is None:
            # İlk frame - motion yok
            for blob in blobs:
                blob['motion'] = {'vx': 0, 'vy': 0, 'speed': 0, 'angle': 0}
            return blobs
        
        for blob in blobs:
            blob['motion'] = self.get_blob_motion(blob, flow)
        
        return blobs
    
    def should_merge(self, blob1, blob2, distance_threshold=80, angle_threshold=np.pi/3):
        """
        İki blob'un merge edilip edilmeyeceğine karar verir
        Paper: Section III-B (Merging foreground blobs)
        
        GÜNCEL THRESHOLDS:
        - distance_threshold: 50 → 80 piksel (daha uzak blob'ları merge et)
        - angle_threshold: π/4 → π/3 (45° → 60°, daha geniş açı toleransı)
        - speed_ratio: 0.5 → 0.4 (hız farkını daha fazla tolere et)
        """
        # 1. Yakınlık kontrolü
        cx1, cy1 = blob1['center']
        cx2, cy2 = blob2['center']
        distance = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        
        if distance > distance_threshold:
            return False
        
        # 2. Motion bilgisi var mı?
        speed1 = blob1['motion']['speed']
        speed2 = blob2['motion']['speed']
        
        # İkisi de neredeyse statik → merge etme
        if speed1 < 0.5 and speed2 < 0.5:
            return False
        
        # En az biri hızlıysa devam et
        if speed1 < 0.5 or speed2 < 0.5:
            # Biri statik biri hareketli → merge etme
            return False
        
        # 3. Benzer hız kontrolü (±60% tolerance) ← GÜNCELLENDİ
        speed_ratio = min(speed1, speed2) / (max(speed1, speed2) + 1e-6)
        
        if speed_ratio < 0.4:  # 0.5 → 0.4 (daha rahat) ← GÜNCELLENDİ
            return False
        
        # 4. Benzer yön kontrolü
        angle1 = blob1['motion']['angle']
        angle2 = blob2['motion']['angle']
        angle_diff = abs(angle1 - angle2)
        
        # Angle wrap-around kontrolü (180° → -180° geçişi)
        if angle_diff > np.pi:
            angle_diff = 2*np.pi - angle_diff
        
        if angle_diff > angle_threshold:
            return False
        
        # Tüm koşullar sağlandı → MERGE!
        return True
    
    def should_split(self, blob, flow, min_blob_size=800):
        """
        Bir blob'un split edilip edilmeyeceğine karar verir
        Paper: Section III-C (Flow separation)
        
        Mantık: Blob içinde birbirine ters yönde hareket eden bölgeler var mı?
        
        Args:
            blob: Blob dict
            flow: Optical flow field
            min_blob_size: Split için minimum blob boyutu
        
        Returns:
            True if should split, False otherwise
        """
        # Küçük blob'ları split etme
        if blob['area'] < min_blob_size:
            return False
        
        if flow is None:
            return False
        
        x, y, w, h = blob['bbox']
        roi_flow = flow[y:y+h, x:x+w]
        
        # Blob içindeki flow vektörlerinin angle'larını hesapla
        fx = roi_flow[:, :, 0].flatten()
        fy = roi_flow[:, :, 1].flatten()
        angles = np.arctan2(fy, fx)
        
        # Angle dağılımını kontrol et
        # Eğer blob içinde hem pozitif hem negatif yönde güçlü motion varsa → SPLIT
        
        # Basit yöntem: angle variance'ı yüksekse split et
        angle_std = np.std(angles)
        
        # Threshold: π/2 (90°) std'den fazlaysa muhtemelen ters yönde nesneler var
        if angle_std > np.pi/2:
            return True
        
        return False
    
    def _merge_two_blobs(self, blob1, blob2):
        """
        İki blob'u birleştirir (union)
        
        Args:
            blob1, blob2: Merge edilecek blob'lar
        
        Returns:
            merged_blob: Yeni birleştirilmiş blob
        """
        x1, y1, w1, h1 = blob1['bbox']
        x2, y2, w2, h2 = blob2['bbox']
        
        # Yeni bbox: İki bbox'ı kapsayan en küçük dikdörtgen
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)
        
        new_w = x_max - x_min
        new_h = y_max - y_min
        
        # Yeni merkez
        new_cx = (x_min + x_max) // 2
        new_cy = (y_min + y_max) // 2
        
        # Yeni alan (yaklaşık - iki blob'un toplamı)
        new_area = blob1['area'] + blob2['area']
        
        # Ortalama motion (weighted average by area)
        total_area = blob1['area'] + blob2['area']
        avg_vx = (blob1['motion']['vx'] * blob1['area'] + 
                  blob2['motion']['vx'] * blob2['area']) / total_area
        avg_vy = (blob1['motion']['vy'] * blob1['area'] + 
                  blob2['motion']['vy'] * blob2['area']) / total_area
        avg_speed = np.sqrt(avg_vx**2 + avg_vy**2)
        avg_angle = np.arctan2(avg_vy, avg_vx)
        
        merged_blob = {
            'bbox': (x_min, y_min, new_w, new_h),
            'center': (new_cx, new_cy),
            'area': new_area,
            'motion': {
                'vx': avg_vx,
                'vy': avg_vy,
                'speed': avg_speed,
                'angle': avg_angle
            },
            'merged': True  # Merge bayrağı
        }
        
        return merged_blob
    
    def refine_blobs(self, blobs):
        """
        Blob'ları merge/split ederek refine eder
        Paper: Section III-B (Merging) + III-C (Splitting)
        
        Args:
            blobs: Motion bilgisi eklenmiş blob listesi
        
        Returns:
            refined_blobs: Merge/split yapılmış blob listesi
        """
        if len(blobs) == 0:
            return blobs
        
        # ========== 1. MERGE LOGIC ==========
        # İki blob'u karşılaştır, merge edilebilirse birleştir
        merged_blobs = []
        merged_indices = set()
        
        for i in range(len(blobs)):
            if i in merged_indices:
                continue
            
            current_blob = blobs[i].copy()  # Copy to avoid modifying original
            
            for j in range(i+1, len(blobs)):
                if j in merged_indices:
                    continue
                
                # Merge kontrolü
                if self.should_merge(current_blob, blobs[j], 
                                    distance_threshold=75,      # 70 piksel
                                    angle_threshold=np.pi/3):   # 60 derece
                    # MERGE!
                    current_blob = self._merge_two_blobs(current_blob, blobs[j])
                    merged_indices.add(j)
            
            merged_blobs.append(current_blob)
            merged_indices.add(i)
        
        # ========== 2. SPLIT LOGIC ==========
        # (Şimdilik basit - gelecekte geliştirilebilir)
        final_blobs = []
        for blob in merged_blobs:
            # Split gerekli mi kontrol et
            # Şimdilik skip - merge daha kritik
            final_blobs.append(blob)
        
        return final_blobs
    
    def visualize_flow(self, flow, frame_shape):
        """
        Optical flow'u HSV renk uzayında görselleştirir
        Hue (H) = yön, Value (V) = hız
        
        Args:
            flow: Optical flow field
            frame_shape: Frame boyutu
        
        Returns:
            flow_bgr: Görselleştirme için BGR image
        """
        if flow is None:
            return np.zeros((*frame_shape[:2], 3), dtype=np.uint8)
        
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        
        # Angle ve magnitude hesapla
        angle = np.arctan2(fy, fx) + np.pi  # [0, 2π]
        magnitude = np.sqrt(fx**2 + fy**2)
        
        # HSV image oluştur
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = angle * (90 / np.pi / 2)  # Hue: yön
        hsv[..., 1] = 255                        # Saturation: tam
        hsv[..., 2] = np.minimum(magnitude * 4, 255).astype(np.uint8)  # Value: hız
        
        # BGR'ye çevir
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_bgr


# ============================================================
# TEST KODU
# ============================================================
def test_optical_flow():
    """
    Optical flow modülünü detection ile birlikte test et
    + Blob refinement (merge/split) testi
    """
    import os
    import sys
    
    # detection.py'yi import et
    from detection import ObjectDetector
    
    print("="*60)
    print("OPTICAL FLOW + REFINEMENT TEST")
    print("="*60)
    
    video_path = "Data/rouen_video.avi"
    
    if not os.path.exists(video_path):
        print(f"❌ Video bulunamadı: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Video açılamadı!")
        return
    
    # Modülleri başlat
    detector = ObjectDetector()
    flow_processor = OpticalFlowProcessor()
    
    frame_count = 0
    
    print("\n🎬 Video işleniyor...")
    print("   ESC: Çıkış")
    print("   SPACE: Duraklat/Devam")
    print("\n   🟢 Yeşil bbox: Normal detection")
    print("   🟡 Sarı bbox: MERGED detection")
    print("="*60 + "\n")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video bitti!")
                break
            
            frame_count += 1
            
            # ========== 1. DETECTION ==========
            blobs, fg_mask = detector.detect(frame)
            
            # ========== 2. OPTICAL FLOW ==========
            flow, flow_mag = flow_processor.compute_flow(frame)
            
            # ========== 3. ADD MOTION TO BLOBS ==========
            blobs_with_motion = flow_processor.add_motion_to_blobs(blobs, flow)
            
            # ========== 4. REFINE BLOBS (MERGE/SPLIT) ==========
            refined_blobs = flow_processor.refine_blobs(blobs_with_motion)
            
            # ========== 5. VISUALIZATION ==========
            result = frame.copy()
            
            for blob in refined_blobs:
                x, y, w, h = blob['bbox']
                cx, cy = blob['center']
                motion = blob['motion']
                
                # Merge edilmiş blob'ları SARI renkte göster
                is_merged = blob.get('merged', False)
                color = (0, 255, 255) if is_merged else (0, 255, 0)  # CYAN if merged, GREEN otherwise
                thickness = 3 if is_merged else 2
                
                # Bounding box
                cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
                
                # Merkez noktası
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                
                # Motion vektörü (ok)
                vx, vy = motion['vx'], motion['vy']
                scale = 5  # Görselleştirme için büyüt
                end_x = int(cx + vx * scale)
                end_y = int(cy + vy * scale)
                cv2.arrowedLine(result, (cx, cy), (end_x, end_y), 
                               (255, 0, 255), 2, tipLength=0.3)
                
                # Motion bilgileri
                speed = motion['speed']
                label = f"v={speed:.1f}"
                if is_merged:
                    label += " [M]"  # Merged bayrağı
                
                cv2.putText(result, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_merged else 1)
            
            # ========== 6. FRAME BİLGİLERİ ==========
            num_merged = sum(1 for b in refined_blobs if b.get('merged', False))
            
            cv2.putText(result, f"Original: {len(blobs)} | Refined: {len(refined_blobs)} (Merged: {num_merged})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(result, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # ========== 7. OPTICAL FLOW GÖRSELLEŞTIRMESI ==========
            flow_vis = flow_processor.visualize_flow(flow, frame.shape)
            
            # ========== 8. GÖSTER ==========
            cv2.imshow("1. Original", frame)
            cv2.imshow("2. FG Mask", fg_mask)
            cv2.imshow("3. Optical Flow (HSV)", flow_vis)
            cv2.imshow("4. Detection + Motion + Refinement", result)
            
            # ========== 9. KONSOL ÇIKTISI ==========
            if frame_count % 30 == 0:
                avg_speed = np.mean([b['motion']['speed'] for b in refined_blobs]) if refined_blobs else 0
                print(f"Frame {frame_count:4d} | Det: {len(blobs):2d} → Refined: {len(refined_blobs):2d} "
                      f"(merged: {num_merged}) | Avg Speed: {avg_speed:.2f}")
        
        # ========== 10. KLAVYE KONTROLÜ ==========
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == 27:  # ESC
            print("\n⛔ Kullanıcı tarafından durduruldu")
            break
        elif key == 32:  # SPACE
            paused = not paused
            status = "⏸️  DURAKLAT" if paused else "▶️  DEVAM"
            print(status)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Toplam {frame_count} frame işlendi")
    print("="*60)


if __name__ == "__main__":
    test_optical_flow()