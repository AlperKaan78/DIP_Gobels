"""
main.py - Complete Multi-Object Tracking Pipeline
Combines: Detection (MOG2) + Optical Flow + Tracking (Kalman) + Data Association (Hungarian)

Papers:
- Beaupre et al. 2018: Optical Flow + Edge Preprocessing
- Shantaiya et al. 2015: Kalman Filter + Optical Flow
"""

import cv2
import numpy as np
import os
import time
from detection import ObjectDetector
from optical_flow import OpticalFlowProcessor
from tracker import TrackerManager, Track
from data_association import associate_hybrid

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """
    Sistem parametreleri
    Tüm threshold ve ayarlar burada - kolay tuning!
    """
    # Video
    VIDEO_PATH = "Data/rouen_video.avi"
    OUTPUT_PATH = "outputs/tracking_result.avi"
    
    # Detection (MOG2)
    MOG2_HISTORY = 500
    MOG2_VAR_THRESHOLD = 16
    MOG2_DETECT_SHADOWS = True
    
    # Optical Flow (Blob Refinement)
    OF_MERGE_DISTANCE = 80          # Blob merge için max mesafe
    OF_MERGE_ANGLE = np.pi / 3      # Blob merge için max açı farkı (60°)
    
    # Tracking (Kalman)
    MAX_MISSES = 30                 # Track silinmeden önce max kayıp frame
    
    # Data Association (Hungarian)
    ASSOCIATION_METRIC = 'hybrid'   # 'iou', 'euclidean', 'hybrid'
    IOU_THRESHOLD = 0.25
    DISTANCE_THRESHOLD = 120
    
    # Visualization
    SHOW_WINDOWS = True
    SHOW_FG_MASK = True
    SHOW_OPTICAL_FLOW = True
    SHOW_TRACKING_RESULT = True
    
    # Colors (BGR)
    COLOR_GOOD_TRACK = (0, 255, 0)      # Yeşil
    COLOR_LOW_CONF = (0, 255, 255)      # Sarı
    COLOR_LOST_TRACK = (0, 0, 255)      # Kırmızı
    COLOR_CENTER = (255, 0, 255)        # Mor
    
    # Stats
    PRINT_EVERY_N_FRAMES = 30
    
    @classmethod
    def print_config(cls):
        """Kullanılan parametreleri yazdır"""
        print("="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Video: {cls.VIDEO_PATH}")
        print(f"Output: {cls.OUTPUT_PATH}")
        print(f"\nDetection:")
        print(f"  MOG2 History: {cls.MOG2_HISTORY}")
        print(f"  Var Threshold: {cls.MOG2_VAR_THRESHOLD}")
        print(f"\nOptical Flow:")
        print(f"  Merge Distance: {cls.OF_MERGE_DISTANCE}px")
        print(f"  Merge Angle: {cls.OF_MERGE_ANGLE:.2f} rad ({np.degrees(cls.OF_MERGE_ANGLE):.0f}°)")
        print(f"\nTracking:")
        print(f"  Max Misses: {cls.MAX_MISSES}")
        print(f"\nData Association:")
        print(f"  Metric: {cls.ASSOCIATION_METRIC}")
        print(f"  IoU Threshold: {cls.IOU_THRESHOLD}")
        print(f"  Distance Threshold: {cls.DISTANCE_THRESHOLD}px")
        print("="*60)


# ============================================================
# STATISTICS COLLECTOR
# ============================================================
class Statistics:
    """
    Tracking performans istatistikleri
    """
    def __init__(self):
        self.total_frames = 0
        self.total_detections = 0
        self.total_matches = 0
        self.total_tracks_created = 0
        self.total_tracks_removed = 0
        self.processing_times = []
        
    def update(self, num_detections, num_matches, processing_time):
        self.total_frames += 1
        self.total_detections += num_detections
        self.total_matches += num_matches
        self.processing_times.append(processing_time)
        
    def print_summary(self):
        avg_fps = 1.0 / (np.mean(self.processing_times) + 1e-6)
        match_rate = (self.total_matches / max(self.total_detections, 1)) * 100
        
        print("\n" + "="*60)
        print("TRACKING STATISTICS")
        print("="*60)
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Total Detections: {self.total_detections}")
        print(f"Total Matches: {self.total_matches}")
        print(f"Match Rate: {match_rate:.1f}%")
        print(f"Tracks Created: {self.total_tracks_created}")
        print(f"Tracks Removed: {self.total_tracks_removed}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average Processing Time: {np.mean(self.processing_times)*1000:.1f}ms per frame")
        print("="*60)


# ============================================================
# VISUALIZATION
# ============================================================
def draw_tracking_result(frame, tracks, frame_count, num_detections, num_matches):
    """
    Tracking sonuçlarını frame üzerine çiz
    """
    result = frame.copy()
    
    for track_state in tracks:
        x, y, w, h = track_state['bbox']
        cx, cy = track_state['center']
        track_id = track_state['id']
        confidence = track_state['confidence']
        misses = track_state['misses']
        
        # Renk seçimi
        if misses > 5:
            color = Config.COLOR_LOST_TRACK
        elif confidence < 0.5:
            color = Config.COLOR_LOW_CONF
        else:
            color = Config.COLOR_GOOD_TRACK
        
        # Bounding box
        thickness = 3 if misses == 0 else 1
        cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        
        # Label
        if misses > 0:
            label = f"ID:{track_id} [LOST:{misses}]"
        else:
            label = f"ID:{track_id} (c:{confidence:.2f})"
        
        # Text background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x, y-label_h-8), (x+label_w, y), color, -1)
        cv2.putText(result, label, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Center point
        cv2.circle(result, (cx, cy), 5, Config.COLOR_CENTER, -1)
    
    # Info panel
    info_bg_color = (0, 0, 0)
    cv2.rectangle(result, (0, 0), (500, 90), info_bg_color, -1)
    
    cv2.putText(result, f"Frame: {frame_count}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(result, f"Active Tracks: {len(tracks)}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(result, f"Detections: {num_detections} | Matches: {num_matches}", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return result


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_tracking_pipeline(config=Config):
    """
    Ana tracking pipeline
    """
    # Config'i yazdır
    config.print_config()
    
    # Video aç
    if not os.path.exists(config.VIDEO_PATH):
        print(f"❌ Video bulunamadı: {config.VIDEO_PATH}")
        return
    
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Video açılamadı!")
        return
    
    # Video bilgileri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    
    # Output video writer
    os.makedirs(os.path.dirname(config.OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (width, height))
    
    # Modülleri başlat
    print("\n🔧 Initializing modules...")
    detector = ObjectDetector(
        mog2_history=config.MOG2_HISTORY,
        mog2_var_threshold=config.MOG2_VAR_THRESHOLD,
        detect_shadows=config.MOG2_DETECT_SHADOWS
    )
    flow_processor = OpticalFlowProcessor()
    tracker_manager = TrackerManager(max_misses=config.MAX_MISSES)
    stats = Statistics()
    
    print("\n🎬 Processing video...")
    print("   Controls:")
    print("   - ESC: Exit")
    print("   - SPACE: Pause/Resume")
    print("="*60 + "\n")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video completed!")
                break
            
            frame_count += 1
            tracker_manager.update_frame_count()
            
            # ========== PIPELINE ==========
            
            # 1. Detection (MOG2 + Morphology + Contours)
            blobs, fg_mask = detector.detect(frame)
            
            # 2. Optical Flow (Dense flow + Motion info)
            flow, flow_mag = flow_processor.compute_flow(frame)
            blobs_with_motion = flow_processor.add_motion_to_blobs(blobs, flow)
            
            # 3. Refinement (Merge/Split blobs)
            refined_blobs = flow_processor.refine_blobs(blobs_with_motion)
            
            # 4. Tracking - Predict
            predictions = tracker_manager.predict_all()
            
            # 5. Data Association (Hungarian)
            active_tracks = tracker_manager.get_active_tracks()
            
            matches, unmatched_detections, unmatched_tracks = associate_hybrid(
                active_tracks,
                refined_blobs,
                iou_threshold=config.IOU_THRESHOLD,
                distance_threshold=config.DISTANCE_THRESHOLD
            )
            
            # 6. Update Tracks
            # Matched
            for track_idx, det_idx in matches:
                track_id = active_tracks[track_idx]['id']
                tracker_manager.update_track(track_id, refined_blobs[det_idx])
            
            # Unmatched tracks (lost)
            for track_idx in unmatched_tracks:
                track_id = active_tracks[track_idx]['id']
                for tid, pred_center in predictions:
                    if tid == track_id:
                        tracker_manager.handle_miss(track_id, pred_center)
                        break
            
            # Unmatched detections (new tracks)
            for det_idx in unmatched_detections:
                tracker_manager.create_track(refined_blobs[det_idx])
            
            # 7. Remove lost tracks
            num_removed = tracker_manager.remove_lost_tracks()
            
            # 8. Visualization
            active_tracks_vis = tracker_manager.get_active_tracks()
            result = draw_tracking_result(
                frame, active_tracks_vis, frame_count, 
                len(refined_blobs), len(matches)
            )
            
            # 9. Statistics
            processing_time = time.time() - start_time
            stats.update(len(refined_blobs), len(matches), processing_time)
            stats.total_tracks_created = Track._next_id - 1
            stats.total_tracks_removed += num_removed
            
            # 10. Display & Save
            if config.SHOW_WINDOWS:
                if config.SHOW_TRACKING_RESULT:
                    cv2.imshow("Tracking Result", result)
                
                if config.SHOW_FG_MASK:
                    cv2.imshow("Foreground Mask", fg_mask)
                
                if config.SHOW_OPTICAL_FLOW and flow is not None:
                    flow_vis = flow_processor.visualize_flow(flow, frame.shape)
                    cv2.imshow("Optical Flow", flow_vis)
            
            out.write(result)
            
            # Console output
            if frame_count % config.PRINT_EVERY_N_FRAMES == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Frame {frame_count:4d}/{total_frames} ({progress:.1f}%) | "
                      f"Active: {len(active_tracks_vis):2d} | "
                      f"Det: {len(refined_blobs):2d} | "
                      f"Match: {len(matches):2d}/{len(refined_blobs):2d} | "
                      f"FPS: {1/processing_time:.1f}")
        
        # Keyboard control
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == 27:  # ESC
            print("\n⛔ Stopped by user")
            break
        elif key == 32:  # SPACE
            paused = not paused
            print("⏸️  PAUSED" if paused else "▶️  RESUMED")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Final stats
    stats.print_summary()
    
    print(f"\n💾 Output saved to: {config.OUTPUT_PATH}")
    print("\n✅ Pipeline completed successfully!")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================
def main():
    """
    Ana fonksiyon - komut satırından çalıştırılabilir
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Object Tracking System')
    parser.add_argument('--video', type=str, default=Config.VIDEO_PATH,
                       help='Input video path')
    parser.add_argument('--output', type=str, default=Config.OUTPUT_PATH,
                       help='Output video path')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display windows (headless mode)')
    parser.add_argument('--max-misses', type=int, default=Config.MAX_MISSES,
                       help='Max frames before removing lost track')
    
    args = parser.parse_args()
    
    # Config'i güncelle
    Config.VIDEO_PATH = args.video
    Config.OUTPUT_PATH = args.output
    Config.SHOW_WINDOWS = not args.no_display
    Config.MAX_MISSES = args.max_misses
    
    # Pipeline'ı çalıştır
    run_tracking_pipeline(Config)


if __name__ == "__main__":
    main()