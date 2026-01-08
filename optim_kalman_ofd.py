import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import time

class KalmanTracker:
    """Kalman Filtresi - nesne pozisyon tahmini"""
    def __init__(self, initial_position):
        self.kf = cv2.KalmanFilter(4, 2)
        
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.kf.statePost = np.array([
            [initial_position[0]],
            [initial_position[1]],
            [0],
            [0]
        ], dtype=np.float32)
        
    def predict(self):
        prediction = self.kf.predict()
        # .item() kullanarak dizinin içindeki skaler değeri aliyoruz
        return (int(prediction[0].item()), int(prediction[1].item()))

    def update(self, measurement):
        measurement_matrix = np.array([[np.float32(measurement[0])],
                                    [np.float32(measurement[1])]])
        self.kf.correct(measurement_matrix)
        # self.kf.statePost[0] yerine .item() ekliyoruz
        return (int(self.kf.statePost[0].item()), int(self.kf.statePost[1].item()))


class ObjectTracker:
    """Çoklu nesne takibi"""
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.positions = OrderedDict()
        self.trajectories = OrderedDict()
        self.bounding_boxes = OrderedDict()  # YENİ: Bounding box sakla
        
    def register(self, centroid, bbox=None):
        self.objects[self.next_object_id] = KalmanTracker(centroid)
        self.disappeared[self.next_object_id] = 0
        self.positions[self.next_object_id] = centroid
        self.trajectories[self.next_object_id] = [centroid]
        self.bounding_boxes[self.next_object_id] = bbox  # YENİ
        self.next_object_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.positions[object_id]
        del self.trajectories[object_id]
        if object_id in self.bounding_boxes:
            del self.bounding_boxes[object_id]
        
    def update(self, detections):
        """
        detections: [(cx, cy, x, y, w, h), ...]
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                predicted = self.objects[object_id].predict()
                self.positions[object_id] = predicted
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            return self.positions
        
        if len(self.objects) == 0:
            for detection in detections:
                cx, cy, x, y, w, h = detection
                self.register((cx, cy), (x, y, w, h))
        else:
            object_ids = list(self.objects.keys())
            predictions = []
            for object_id in object_ids:
                pred = self.objects[object_id].predict()
                predictions.append(pred)
            
            D = np.zeros((len(predictions), len(detections)))
            for i, pred in enumerate(predictions):
                for j, det in enumerate(detections):
                    cx, cy = det[0], det[1]
                    D[i, j] = np.linalg.norm(np.array(pred) - np.array([cx, cy]))
            
            row_idx, col_idx = linear_sum_assignment(D)
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(row_idx, col_idx):
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                cx, cy, x, y, w, h = detections[col]
                
                updated_pos = self.objects[object_id].update((cx, cy))
                self.positions[object_id] = updated_pos
                self.disappeared[object_id] = 0
                self.bounding_boxes[object_id] = (x, y, w, h)  # YENİ: bbox güncelle
                
                self.trajectories[object_id].append(updated_pos)
                if len(self.trajectories[object_id]) > 30:
                    self.trajectories[object_id].pop(0)
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(len(predictions))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                predicted = self.objects[object_id].predict()
                self.positions[object_id] = predicted
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            unused_cols = set(range(len(detections))) - used_cols
            for col in unused_cols:
                cx, cy, x, y, w, h = detections[col]
                self.register((cx, cy), (x, y, w, h))
        
        return self.positions


def get_flow_direction(flow, x, y, w, h):
    """
    Bounding box içindeki optical flow yönünü hesapla
    Returns: (mean_vx, mean_vy, magnitude)
    """
    # Bounding box içindeki flow'u al
    roi_flow = flow[y:y+h, x:x+w]
    
    if roi_flow.size == 0:
        return (0, 0, 0)
    
    # Ortalama flow vektörü
    mean_vx = np.mean(roi_flow[..., 0])
    mean_vy = np.mean(roi_flow[..., 1])
    magnitude = np.sqrt(mean_vx**2 + mean_vy**2)
    
    return (mean_vx, mean_vy, magnitude)


def get_flow_angle(vx, vy):
    """Flow yönünü açi olarak döndür (derece)"""
    angle = np.arctan2(vy, vx) * 180 / np.pi
    return angle


def group_objects_by_flow(objects, bboxes, flow, distance_threshold=100, angle_threshold=30):
    """
    Optical flow yönüne ve mesafeye göre nesneleri grupla
    
    Args:
        objects: {id: (cx, cy), ...}
        bboxes: {id: (x, y, w, h), ...}
        flow: Optical flow array
        distance_threshold: Maksimum mesafe (piksel)
        angle_threshold: Maksimum açi farki (derece)
    
    Returns:
        groups: [[id1, id2], [id3], ...] şeklinde gruplar
    """
    if len(objects) == 0:
        return []
    
    # Her nesne için flow bilgisini hesapla
    flow_data = {}
    for obj_id, (cx, cy) in objects.items():
        if obj_id not in bboxes:
            continue
        
        x, y, w, h = bboxes[obj_id]
        vx, vy, mag = get_flow_direction(flow, x, y, w, h)
        angle = get_flow_angle(vx, vy)
        
        flow_data[obj_id] = {
            'pos': (cx, cy),
            'vx': vx,
            'vy': vy,
            'mag': mag,
            'angle': angle
        }
    
    # Gruplama
    groups = []
    used = set()
    
    for obj_id in flow_data.keys():
        if obj_id in used:
            continue
        
        group = [obj_id]
        used.add(obj_id)
        
        data1 = flow_data[obj_id]
        cx1, cy1 = data1['pos']
        angle1 = data1['angle']
        mag1 = data1['mag']
        
        # Minimum hareket eşiği
        if mag1 < 0.5:
            groups.append(group)
            continue
        
        # Diğer nesnelerle karşilaştir
        for other_id in flow_data.keys():
            if other_id == obj_id or other_id in used:
                continue
            
            data2 = flow_data[other_id]
            cx2, cy2 = data2['pos']
            angle2 = data2['angle']
            mag2 = data2['mag']
            
            # Minimum hareket eşiği
            if mag2 < 0.5:
                continue
            
            # Mesafe kontrolü
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if distance > distance_threshold:
                continue
            
            # Açi farki kontrolü
            angle_diff = abs(angle1 - angle2)
            # 180 derece wrap around (örn: 10° ve 350° aslinda 20° fark)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff <= angle_threshold:
                group.append(other_id)
                used.add(other_id)
        
        groups.append(group)
    
    return groups


def draw_grouped_bbox(img, groups, objects, bboxes, colors):
    """
    Gruplara göre tek bir büyük dikdörtgen çiz
    Tekli nesneler (grup boyutu=1) için de kendi çerçevelerini çiz
    """
    for group in groups:
        if len(group) == 0:
            continue
        
        # Grubun rengini ilk nesnenin rengine göre belirle
        color = colors[group[0] % len(colors)]
        
        # Tekli nesne - sadece kendi çerçevesi
        if len(group) == 1:
            obj_id = group[0]
            if obj_id not in bboxes or obj_id not in objects:
                continue
            
            x, y, w, h = bboxes[obj_id]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            continue
        
        # Çoklu nesne - birleşik dikdörtgen
        all_points = []
        for obj_id in group:
            if obj_id not in bboxes or obj_id not in objects:
                continue
            
            x, y, w, h = bboxes[obj_id]
            
            # Box'in 4 köşesini ekle
            all_points.extend([
                [x, y],
                [x + w, y],
                [x, y + h],
                [x + w, y + h]
            ])
        
        if len(all_points) == 0:
            continue
        
        # Tüm noktalari kapsayan minimum dikdörtgeni bul
        all_points = np.array(all_points)
        min_x = np.min(all_points[:, 0])
        min_y = np.min(all_points[:, 1])
        max_x = np.max(all_points[:, 0])
        max_y = np.max(all_points[:, 1])
        
        # Biraz padding ekle
        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img.shape[1], max_x + padding)
        max_y = min(img.shape[0], max_y + padding)
        
        # Grup dikdörtgenini çiz (kalin çizgi)
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color, 3)


def main():
    # ========== PERFORMANS AYARLARI ==========
    RESIZE_SCALE = 1
    SKIP_FRAMES = 2
    SHOW_DEBUG = True
    
    # YENİ: Gruplama parametreleri
    GROUP_DISTANCE = 120    # Maksimum mesafe (piksel)
    GROUP_ANGLE = 35        # Maksimum açi farki (derece)
    
    video_path = "/home/yunus/projects/DIP_Gobels/Data/rouen_video.avi"
    cam = cv2.VideoCapture(video_path)
    
    original_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    
    print(f"Video: {original_width}x{original_height} @ {fps} FPS")
    print(f"İşlem boyutu: {int(original_width*RESIZE_SCALE)}x{int(original_height*RESIZE_SCALE)}")
    
    ret, prev = cam.read()
    if not ret:
        print("Video okunamadi!")
        return
    
    prev = cv2.resize(prev, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=20,
        detectShadows=False
    )
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    tracker = ObjectTracker(max_disappeared=20, max_distance=100)
    
    print("\nBirleştime yöntemi seçin:")
    print("1: OR  (Birleşim) - En fazla tespit")
    print("2: AND (Kesişim) - En kesin tespit")
    print("3: WEIGHTED (Ağirlikli) - Dengeli")
    
    choice = input("Seçim (1-3): ")
    
    if choice == "1":
        merge_method = "or"
        method_name = "OR"
    elif choice == "2":
        merge_method = "and"
        method_name = "AND"
    elif choice == "3":
        merge_method = "weighted"
        method_name = "WEIGHTED"
    else:
        merge_method = "weighted"
        method_name = "WEIGHTED"
        print("Varsayilan: WEIGHTED")
    
    print(f"\n🚀 Başliyor... (ESC ile çikiş)")
    print(f"📦 Gruplama: Mesafe<{GROUP_DISTANCE}px, Açi<{GROUP_ANGLE}°\n")
    
    frame_count = 0
    process_count = 0
    
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
        (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0)
    ]
    
    # YENİ: Flow'u sakla (gruplama için gerekli)
    current_flow = None
    
    # Sahne değişimi sayaci
    scene_change_count = 0
    
    # Kontur güncelleme sistemi
    bbox_update_counter = 0
    BBOX_UPDATE_INTERVAL = 5  # Her 5 frame'de bir güncelle
    stable_groups = []  # Stabil grup bilgisi
    
    while True:
        ret, img = cam.read()
        
        if not ret:
            print("\n✓ Video bitti!")
            break
        
        frame_count += 1
        
        if frame_count % SKIP_FRAMES != 0:
            continue
        
        process_count += 1
        
        img = cv2.resize(img, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ===== SAHNE DEĞİŞİMİ TESPİTİ =====
        
        # 1. Frame fark kontrolü (hizli)
        frame_diff = cv2.absdiff(prevgray, gray)
        diff_ratio = np.sum(frame_diff > 30) / frame_diff.size
        
        if diff_ratio > 0.7:
            print(f"⚠️  Frame {frame_count}: Sahne değişimi tespit edildi (diff_ratio: {diff_ratio:.2f})")
            scene_change_count += 1
            # BackgroundSubtractor'i resetle
            backSub = cv2.createBackgroundSubtractorMOG2(
                history=300,
                varThreshold=20,
                detectShadows=True
            )
            # Konturlari da resetle
            bbox_update_counter = 0
            stable_groups = []
            prevgray = gray
            continue  # Bu frame'i atla
        
        # ===== İŞLEME BAŞLA =====
        
        # MOG2
        fg_mask = backSub.apply(img)
        fg_mask_no_shadow = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        fg_mask_clean = cv2.morphologyEx(fg_mask_no_shadow, cv2.MORPH_OPEN, kernel_open)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel_close)
        
        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prevgray, gray, None,
            0.5, 2, 10, 2, 5, 1.1, 0
        )
        current_flow = flow  # YENİ: Sakla
        prevgray = gray
        
        # ===== OPTICAL FLOW GÖRSELLEŞTİRME =====
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 2. Ortalama magnitude kontrolü (daha hassas sahne değişimi tespiti)
        avg_mag = np.mean(mag)
        if avg_mag > 8.0:
            print(f"⚠️  Frame {frame_count}: Aşiri hareket tespit edildi (avg_mag: {avg_mag:.2f})")
            scene_change_count += 1
            bbox_update_counter = 0
            stable_groups = []
            prevgray = gray
            continue  # Bu frame'i atla
        
        # 1. Magnitude görüntüsü (hareket hizi)
        mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        mag_img = mag_normalized.astype(np.uint8)
        mag_colored = cv2.applyColorMap(mag_img, cv2.COLORMAP_JET)
        
        # 2. Direction görüntüsü (HSV - renk = yön, parlaklik = hiz)
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = yön
        hsv[..., 1] = 255  # Saturation = maksimum
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = hiz
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Flow arrows (seyrek olarak - her 20 piksel)
        step = 20
        flow_arrows = img.copy()
        for y in range(0, gray.shape[0], step):
            for x in range(0, gray.shape[1], step):
                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx*fx + fy*fy)
                if magnitude > 1.0:  # Sadece anlamli flow
                    end_x = int(x + fx * 3)
                    end_y = int(y + fy * 3)
                    cv2.arrowedLine(flow_arrows, (x, y), (end_x, end_y), 
                                   (0, 255, 0), 1, tipLength=0.3)
        
        # Threshold flow
        thresh_flow = (mag > 1.8).astype(np.uint8) * 255
        thresh_flow = cv2.dilate(thresh_flow, None, iterations=1)
        
        # Maskeleri birleştir
        if merge_method == "or":
            merged_mask = cv2.bitwise_or(fg_mask_clean, thresh_flow)
        elif merge_method == "and":
            merged_mask = cv2.bitwise_and(fg_mask_clean, thresh_flow)
        else:
            merged_mask = cv2.addWeighted(fg_mask_clean, 0.6, thresh_flow, 0.4, 0)
            _, merged_mask = cv2.threshold(merged_mask, 127, 255, cv2.THRESH_BINARY)
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Kontur tespiti
        contours, _ = cv2.findContours(
            merged_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # YENİ: Tespit bilgilerini topla (bbox dahil)
        detections = []
        min_area = int(400 * RESIZE_SCALE * RESIZE_SCALE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            (x, y, w, h) = cv2.boundingRect(c)
            
            max_w = int(400 * RESIZE_SCALE)
            max_h = int(600 * RESIZE_SCALE)
            
            if w < max_w and h < max_h:
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:
                    # YENİ: bbox bilgisini de ekle
                    detections.append((cx, cy, x, y, w, h))
        
        # Tracker'i güncelle
        objects = tracker.update(detections)
        
        # YENİ: Optical flow bazli gruplama - ama sadece belirli araliklarla
        bbox_update_counter += 1
        if bbox_update_counter >= BBOX_UPDATE_INTERVAL:
            bbox_update_counter = 0
            stable_groups = group_objects_by_flow(
                objects, 
                tracker.bounding_boxes, 
                current_flow,
                distance_threshold=GROUP_DISTANCE,
                angle_threshold=GROUP_ANGLE
            )
        
        # ===== GÖRSELLEŞTIRME =====
        
        img_result = img.copy()
        
        # Hangi nesneler 2+ kişilik grupta olduğunu bul
        objects_in_multi_groups = set()
        for group in stable_groups:
            if len(group) > 1:  # Sadece 2+ üyeli GERÇEK gruplar
                objects_in_multi_groups.update(group)
        
        # ===== SADECE GRUP ÇERÇEVELERİNİ ÇİZ =====
        # Bu fonksiyon:
        # - Tekli nesneler (grup=1) için → kendi çerçevesi
        # - Çoklu gruplar (grup>1) için → büyük ortak çerçeve
        draw_grouped_bbox(img_result, stable_groups, objects, tracker.bounding_boxes, colors)
        
        # ===== SADECE FLOW OKLARI =====
        # Trajectory (iz) çizimi KALDIRILDI
        for object_id, (cx, cy) in objects.items():
            color = colors[object_id % len(colors)]
            
            # Sadece flow oku (hareket yönü) - merkeze yakin, küçük
            if object_id in tracker.bounding_boxes:
                x, y, w, h = tracker.bounding_boxes[object_id]
                vx, vy, mag = get_flow_direction(current_flow, x, y, w, h)
                
                if mag > 0.5:
                    scale = 10  # 15'ten küçültüldü - daha küçük oklar
                    end_x = int(cx + vx * scale)
                    end_y = int(cy + vy * scale)
                    cv2.arrowedLine(img_result, (cx, cy), (end_x, end_y), 
                                   color, 2, tipLength=0.4)
        
        # FPS hesapla
        fps_counter += 1
        if time.time() - fps_start_time > 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Bilgi paneli
        info_panel = np.zeros((165, img.shape[1], 3), dtype=np.uint8)
        
        cv2.putText(info_panel, f"FPS: {current_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_panel, f"Frame: {frame_count}/{process_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Method: {method_name}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(info_panel, f"Objects: {len(objects)}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(info_panel, f"Groups: {len(stable_groups)}", (10, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(info_panel, f"Scene Changes: {scene_change_count}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        display = np.vstack([info_panel, img_result])
        
        cv2.imshow('Flow-Based Grouping [OPTIMIZED]', display)
        
        if SHOW_DEBUG:
            # # 1. Background Subtraction sonuçlari
            # cv2.imshow("1. MOG2 Raw", fg_mask)
            # cv2.imshow("2. MOG2 Clean", fg_mask_clean)
            
            # 2. Optical Flow görselleştirmeleri
            # cv2.imshow("3. Flow Magnitude (Speed)", mag_colored)
            # cv2.imshow("4. Flow Direction (HSV)", flow_rgb)
            # cv2.imshow("5. Flow Arrows", flow_arrows)
            # cv2.imshow("6. Flow Threshold", thresh_flow)
            
            # # 3. Birleştirilmiş maske
            cv2.imshow("7. Merged Mask", merged_mask)

        if process_count % 100 == 0:
            print(f"⚡ Frame {frame_count} | İşlenen: {process_count} | "
                  f"FPS: {current_fps:.1f} | Nesneler: {len(objects)} | Gruplar: {len(stable_groups)}")
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            print("\n⏹️  Kullanici durdurdu.")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("📊 İŞLEM RAPORU")
    print("="*50)
    print(f"Toplam Frame: {frame_count}")
    print(f"İşlenen Frame: {process_count}")
    print(f"Atlanan Frame (Sahne Değişimi): {scene_change_count}")
    print(f"Tespit Edilen Nesne: {tracker.next_object_id}")
    print(f"Ortalama FPS: {current_fps:.1f}")
    print("="*50)


if __name__ == "__main__":
    main()