"""
tracker.py - Kalman Filter Based Multi-Object Tracker
Paper: Shantaiya et al. 2015 - Multiple Object Tracking using Kalman Filter and Optical Flow
"""

import cv2
import numpy as np

class Track:
    """
    Tek bir nesne için tracking bilgisi
    Her track kendi Kalman filter'ına sahip
    """
    
    # Class variable: track ID counter
    _next_id = 1
    
    def __init__(self, detection, frame_id):
        """
        Args:
            detection: İlk detection {'bbox', 'center', 'area', 'motion'}
            frame_id: Track'in başladığı frame ID
        """
        self.id = Track._next_id
        Track._next_id += 1
        
        # Track state
        self.age = 0                    # Kaç frame'dir yaşıyor
        self.hits = 1                   # Kaç kez tespit edildi
        self.misses = 0                 # Kaç frame'dir kayıp
        self.confidence = 1.0           # Track güvenilirliği [0, 1]
        
        # Detection bilgileri
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.area = detection['area']
        
        # Motion bilgisi (optical flow'dan)
        motion = detection.get('motion', {'vx': 0, 'vy': 0, 'speed': 0})
        self.vx = motion['vx']
        self.vy = motion['vy']
        
        # Kalman Filter oluştur
        self.kf = self._create_kalman_filter(detection['center'], self.vx, self.vy)
        
        # Track history (debugging için)
        self.history = [detection['center']]
        
        print(f"  ✨ Track {self.id} created at frame {frame_id}")
    
    def _create_kalman_filter(self, center, vx, vy):
        """
        Kalman filter oluşturur
        State: [x, y, vx, vy] - pozisyon ve hız
        Measurement: [x, y] - sadece pozisyon ölçülür
        
        Paper: test_kalman.py'deki implementasyon
        """
        kf = cv2.KalmanFilter(4, 2)  # 4 state, 2 measurement
        
        # Transition Matrix (F): State geçiş matrisi
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Measurement Matrix (H): [x, y] ölçülür
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Process Noise Covariance (Q): Model belirsizliği
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        
        # Measurement Noise Covariance (R): Ölçüm belirsizliği
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8
        
        # Error Covariance (P): Başlangıç belirsizliği
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initial State: [x, y, vx, vy]
        cx, cy = center
        kf.statePost = np.array([[cx], [cy], [vx], [vy]], np.float32)
        
        return kf
    
    def predict(self):
        """
        Kalman prediction: Nesnenin bir sonraki frame'deki konumunu tahmin et
        
        Returns:
            predicted_center: (px, py) - tahmin edilen merkez
        """
        prediction = self.kf.predict()
        
        # ========== NUMPY FIX ==========
        # prediction[0] bir array → scalar çıkar
        px = int(prediction[0][0])
        py = int(prediction[1][0])
        
        return (px, py)
    
    def update(self, detection):
        """
        Track'i yeni detection ile güncelle (Kalman correct)
        
        Args:
            detection: Yeni detection dict
        """
        # Kalman correct
        cx, cy = detection['center']
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)
        
        # Track bilgilerini güncelle
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.area = detection['area']
        
        # Motion bilgisini güncelle
        motion = detection.get('motion', {'vx': 0, 'vy': 0})
        self.vx = motion['vx']
        self.vy = motion['vy']
        
        # İstatistikleri güncelle
        self.hits += 1
        self.misses = 0
        self.confidence = min(1.0, self.confidence + 0.1)
        
        # History'ye ekle
        self.history.append(detection['center'])
        if len(self.history) > 30:  # Son 30 frame
            self.history.pop(0)
    
    def handle_miss(self, predicted_center):
        """
        Detection yokken track'i güncelle (predict-only mode)
        Occlusion handling
        
        Args:
            predicted_center: Kalman'dan tahmin edilen pozisyon
        """
        self.misses += 1
        self.confidence = max(0, self.confidence - 0.15)  # 0.2 → 0.15 (daha yavaş düşsün)
        
        # Predicted pozisyonu kullan
        px, py = predicted_center
        x, y, w, h = self.bbox
        
        # Bbox'u tahmin edilen merkeze kaydır
        dx = px - (x + w//2)
        dy = py - (y + h//2)
        self.bbox = (x + dx, y + dy, w, h)
        self.center = predicted_center
    
    def is_lost(self, max_misses=20):  # 10 → 20 (daha toleranslı)
        """
        Track kayıp mı? (silinmeli mi?)
        
        Args:
            max_misses: Maksimum kayıp frame sayısı
        
        Returns:
            True if track should be deleted
        """
        # Çok uzun süre kayıp
        if self.misses > max_misses:
            return True
        
        # Confidence çok düştü
        if self.confidence < 0.1:  # 0.2 → 0.1 (daha toleranslı)
            return True
        
        return False
    
    def get_state(self):
        """
        Track'in güncel durumunu döndür (visualization için)
        
        Returns:
            state dict
        """
        return {
            'id': self.id,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'confidence': self.confidence,
            'hits': self.hits,
            'misses': self.misses,
            'age': self.age,
            'velocity': (self.vx, self.vy)
        }


class TrackerManager:
    """
    Tüm track'leri yöneten manager class
    """
    
    def __init__(self, max_misses=20):  # 10 → 20
        """
        Args:
            max_misses: Track silinmeden önce maksimum kayıp frame sayısı
        """
        self.tracks = []
        self.max_misses = max_misses
        self.frame_count = 0
        
        print("✅ TrackerManager initialized")
    
    def predict_all(self):
        """
        Tüm track'ler için Kalman prediction yap
        
        Returns:
            predictions: List of (track_id, predicted_center)
        """
        predictions = []
        for track in self.tracks:
            pred_center = track.predict()
            predictions.append((track.id, pred_center))
            track.age += 1
        
        return predictions
    
    def create_track(self, detection):
        """
        Yeni track oluştur
        
        Args:
            detection: Detection dict
        """
        new_track = Track(detection, self.frame_count)
        self.tracks.append(new_track)
    
    def update_track(self, track_id, detection):
        """
        Mevcut track'i güncelle
        
        Args:
            track_id: Track ID
            detection: Yeni detection
        """
        for track in self.tracks:
            if track.id == track_id:
                track.update(detection)
                break
    
    def handle_miss(self, track_id, predicted_center):
        """
        Track için detection bulunamadı (occlusion)
        
        Args:
            track_id: Track ID
            predicted_center: Kalman prediction
        """
        for track in self.tracks:
            if track.id == track_id:
                track.handle_miss(predicted_center)
                break
    
    def remove_lost_tracks(self):
        """
        Kayıp track'leri sil
        
        Returns:
            num_removed: Silinen track sayısı
        """
        initial_count = len(self.tracks)
        
        self.tracks = [t for t in self.tracks if not t.is_lost(self.max_misses)]
        
        removed = initial_count - len(self.tracks)
        if removed > 0:
            print(f"  🗑️  {removed} track(s) removed (lost)")
        
        return removed
    
    def get_active_tracks(self):
        """
        Aktif track'leri döndür
        
        Returns:
            List of track states
        """
        return [track.get_state() for track in self.tracks]
    
    def update_frame_count(self):
        """
        Frame counter'ı artır
        """
        self.frame_count += 1


# ============================================================
# TEST KODU - HUNGARIAN ALGORITHM İLE
# ============================================================
def test_tracker():
    """
    Tracker modülünü test et
    Data association: Hungarian Algorithm (IoU-based)
    """
    import os
    from detection import ObjectDetector
    from optical_flow import OpticalFlowProcessor
    from data_association import associate_detections_to_tracks, associate_hybrid

    
    print("="*60)
    print("TRACKER MODULE TEST (with Hungarian Algorithm)")
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
    tracker_manager = TrackerManager(max_misses=20)  # 10 → 20
    
    frame_count = 0
    
    print("\n🎬 Video işleniyor...")
    print("   ESC: Çıkış | SPACE: Duraklat")
    print("\n   Renk kodları:")
    print("   🟢 Yeşil: Normal track (high confidence)")
    print("   🟡 Sarı: Low confidence track")
    print("   🔴 Kırmızı: Lost track (predict-only)")
    print("\n   🔷 Data Association: Hungarian (IoU-based)")
    print("="*60 + "\n")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video bitti!")
                break
            
            frame_count += 1
            tracker_manager.update_frame_count()
            
            # 1. DETECTION
            blobs, fg_mask = detector.detect(frame)
            
            # 2. OPTICAL FLOW
            flow, flow_mag = flow_processor.compute_flow(frame)
            blobs_with_motion = flow_processor.add_motion_to_blobs(blobs, flow)
            
            # 3. REFINEMENT
            refined_blobs = flow_processor.refine_blobs(blobs_with_motion)
            
            # 4. TRACKING - PREDICT
            predictions = tracker_manager.predict_all()
            
            # 5. DATA ASSOCIATION - HUNGARIAN ALGORITHM
            active_tracks = tracker_manager.get_active_tracks()
            
            # matches, unmatched_detections, unmatched_tracks = associate_detections_to_tracks(
            #     active_tracks,
            #     refined_blobs,
            #     metric='iou',           # IoU-based
            #     iou_threshold=0.3       # Min 0.3 overlap
            # )
            matches, unmatched_detections, unmatched_tracks = associate_hybrid(
                active_tracks,
                refined_blobs,
                iou_threshold=0.25,      # 0.3 → 0.25 (biraz daha düşük)
                distance_threshold=120   # 100 → 120 (biraz daha yüksek)
            )
            
            # 6. UPDATE TRACKS
            # Matched tracks
            for track_idx, det_idx in matches:
                track_id = active_tracks[track_idx]['id']
                tracker_manager.update_track(track_id, refined_blobs[det_idx])
            
            # Unmatched tracks (lost/occluded)
            for track_idx in unmatched_tracks:
                track_id = active_tracks[track_idx]['id']
                # Kalman prediction'ı kullan
                for tid, pred_center in predictions:
                    if tid == track_id:
                        tracker_manager.handle_miss(track_id, pred_center)
                        break
            
            # Unmatched detections (new tracks)
            for det_idx in unmatched_detections:
                tracker_manager.create_track(refined_blobs[det_idx])
            
            # 7. REMOVE LOST TRACKS
            tracker_manager.remove_lost_tracks()
            
            # 8. VISUALIZATION
            result = frame.copy()
            active_tracks_vis = tracker_manager.get_active_tracks()
            
            for track_state in active_tracks_vis:
                x, y, w, h = track_state['bbox']
                cx, cy = track_state['center']
                track_id = track_state['id']
                confidence = track_state['confidence']
                hits = track_state['hits']
                misses = track_state['misses']
                
                # Renk: confidence'a göre
                if misses > 5:
                    color = (0, 0, 255)  # Kırmızı: Lost
                elif confidence < 0.5:
                    color = (0, 255, 255)  # Sarı: Low confidence
                else:
                    color = (0, 255, 0)  # Yeşil: Good
                
                # Bounding box
                thickness = 3 if misses == 0 else 1
                cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
                
                # Track ID + Info
                label = f"ID:{track_id} (conf:{confidence:.2f})"
                if misses > 0:
                    label = f"ID:{track_id} [LOST:{misses}]"
                
                cv2.putText(result, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Merkez
                cv2.circle(result, (cx, cy), 5, (255, 0, 255), -1)
            
            # Frame bilgileri
            cv2.putText(result, f"Frame: {frame_count} | Tracks: {len(active_tracks_vis)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(result, f"Detections: {len(refined_blobs)} | Matches: {len(matches)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Göster
            cv2.imshow("Tracking Result (Hungarian)", result)
            
            # Konsol
            if frame_count % 30 == 0:
                total_tracks_created = Track._next_id - 1
                print(f"Frame {frame_count:4d} | Active: {len(active_tracks_vis):2d} | "
                      f"Det: {len(refined_blobs):2d} | Match: {len(matches):2d} | "
                      f"Total Created: {total_tracks_created}")
        
        # Klavye
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
    print(f"🎯 Toplam {Track._next_id - 1} track oluşturuldu")
    print("="*60)


if __name__ == "__main__":
    test_tracker()