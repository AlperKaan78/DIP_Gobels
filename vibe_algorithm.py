import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class ViBeBackgroundSubtractor:
    """
    ViBe (Visual Background Extractor) implementasyonu
    Makaledeki [4] numaralı referans
    """
    def __init__(self, n_samples=20, min_match=2, radius=20, subsample_factor=16):
        self.n_samples = n_samples
        self.min_match = min_match
        self.radius = radius
        self.subsample_factor = subsample_factor
        self.samples = None
        self.height = 0
        self.width = 0
        
    def initialize(self, first_frame):
        """İlk frame ile model başlat"""
        if len(first_frame.shape) == 3:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        self.height, self.width = first_frame.shape
        self.samples = np.zeros((self.height, self.width, self.n_samples), dtype=np.uint8)
        
        # Her piksel için rastgele komşu değerlerle başlat
        for i in range(self.n_samples):
            # Rastgele offset
            offset_y = np.random.randint(-1, 2, (self.height, self.width))
            offset_x = np.random.randint(-1, 2, (self.height, self.width))
            
            # Sınırları kontrol et
            y_coords = np.clip(np.arange(self.height)[:, None] + offset_y, 0, self.height - 1)
            x_coords = np.clip(np.arange(self.width)[None, :] + offset_x, 0, self.width - 1)
            
            self.samples[:, :, i] = first_frame[y_coords, x_coords]
    
    def apply(self, frame):
        """Frame'e background subtraction uygula"""
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        if self.samples is None:
            self.initialize(frame_gray)
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Foreground mask oluştur
        fg_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Her piksel için eşleşme sayısını hesapla
        matches = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for i in range(self.n_samples):
            dist = np.abs(frame_gray.astype(np.int16) - self.samples[:, :, i].astype(np.int16))
            matches += (dist <= self.radius).astype(np.uint8)
        
        # Yeterli eşleşme yoksa foreground
        fg_mask = (matches < self.min_match).astype(np.uint8) * 255
        
        # Model güncelle (background pikselleri için)
        bg_mask = fg_mask == 0
        update_mask = np.random.rand(self.height, self.width) < (1.0 / self.subsample_factor)
        update_mask = update_mask & bg_mask
        
        if np.any(update_mask):
            # Rastgele bir sample index seç
            sample_idx = np.random.randint(0, self.n_samples, (self.height, self.width))
            
            # Güncelle
            for i in range(self.n_samples):
                mask = update_mask & (sample_idx == i)
                self.samples[mask, i] = frame_gray[mask]
            
            # Komşuları da güncelle
            y_indices, x_indices = np.where(update_mask)
            if len(y_indices) > 0:
                # Rastgele komşu seç
                offset_y = np.random.randint(-1, 2, len(y_indices))
                offset_x = np.random.randint(-1, 2, len(x_indices))
                
                neighbor_y = np.clip(y_indices + offset_y, 0, self.height - 1)
                neighbor_x = np.clip(x_indices + offset_x, 0, self.width - 1)
                
                neighbor_sample = np.random.randint(0, self.n_samples, len(y_indices))
                
                for idx in range(len(y_indices)):
                    ny, nx, ns = neighbor_y[idx], neighbor_x[idx], neighbor_sample[idx]
                    self.samples[ny, nx, ns] = frame_gray[y_indices[idx], x_indices[idx]]
        
        return fg_mask


class ObjectDetectionImprover:
    def __init__(self, alpha=0.01, merge_distance=7, angle_threshold=np.pi/2, 
                 intersection_threshold=0.4, area_ratio_threshold=0.65, sigma=1/3,
                 min_blob_area=500, max_blob_area=50000):  # Yeni parametreler
        """
        Makaledeki yöntemi uygulayan sınıf (ViBe ile) - OPTIMIZE EDİLMİŞ
        
        Args:
            alpha: Arka plan biriktirme oranı
            merge_distance: Blob birleştirme için maksimum mesafe
            angle_threshold: Açı eşiği (π/2)
            intersection_threshold: Kesişim oranı eşiği
            area_ratio_threshold: Alan oranı eşiği
            sigma: Canny edge detector için parametre
            min_blob_area: Minimum blob alanı (gürültüyü filtreler)
            max_blob_area: Maximum blob alanı (çok büyük blobları filtreler)
        """
        self.alpha = alpha
        self.merge_distance = merge_distance
        self.angle_threshold = angle_threshold
        self.intersection_threshold = intersection_threshold
        self.area_ratio_threshold = area_ratio_threshold
        self.sigma = sigma
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.background = None
        
        # ViBe background subtractor
        self.bg_subtractor = ViBeBackgroundSubtractor(
            n_samples=20, 
            min_match=2, 
            radius=20, 
            subsample_factor=16
        )
    
    def update_background(self, frame):
        """Arka plan görüntüsünü güncelle (Denklem 1)"""
        if self.background is None:
            self.background = frame.astype(np.float32)
        else:
            self.background = self.alpha * frame + (1 - self.alpha) * self.background
        return self.background.astype(np.uint8)
    
    def get_foreground_blobs(self, frame):
        """ViBe ile ön plan bloblarını al"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morfolojik işlemler - gürültüyü temizle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum alan eşiği
                x, y, w, h = cv2.boundingRect(contour)
                blobs.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'mask': fg_mask[y:y+h, x:x+w]
                })
        
        return blobs, fg_mask
    
    def compute_blob_flow(self, flow, bbox):
        """Bir blob için optik akış özelliklerini hesapla"""
        x, y, w, h = bbox
        blob_flow = flow[y:y+h, x:x+w]
        
        if blob_flow.size == 0:
            return {'mag_mean': 0, 'mag_std': 0, 'angle': 0}
        
        # Magnitude ve angle hesapla
        fx = blob_flow[:, :, 0]
        fy = blob_flow[:, :, 1]
        mag = np.sqrt(fx**2 + fy**2)
        angle = np.arctan2(fy, fx)
        
        # Merkez noktada angle
        cy, cx = h//2, w//2
        center_angle = angle[cy, cx] if cy < angle.shape[0] and cx < angle.shape[1] else 0
        
        return {
            'mag_mean': np.mean(mag),
            'mag_std': np.std(mag),
            'angle': center_angle
        }
    
    def check_merge_conditions(self, blob1, blob2, flow1_props, flow2_props):
        """İki blobu birleştirme koşullarını kontrol et (C1, C2, C3)"""
        x1, y1, w1, h1 = blob1['bbox']
        x2, y2, w2, h2 = blob2['bbox']
        
        # C1: Minimum mesafe kontrolü (Denklem 2)
        min_dist = self.compute_min_distance(blob1['bbox'], blob2['bbox'])
        if min_dist > self.merge_distance:
            return False
        
        # C2: Magnitude interval kontrolü (Denklem 3-6)
        min1 = flow1_props['mag_mean'] - flow1_props['mag_std']
        max1 = flow1_props['mag_mean'] + flow1_props['mag_std']
        min2 = flow2_props['mag_mean'] - flow2_props['mag_std']
        max2 = flow2_props['mag_mean'] + flow2_props['mag_std']
        
        # Kesişim kontrolü
        if max(min1, min2) > min(max1, max2):
            return False
        
        # C3: Açı kontrolü (Denklem 7)
        angle_diff = abs(flow1_props['angle'] - flow2_props['angle'])
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # En küçük açı farkı
        
        if angle_diff > self.angle_threshold:
            return False
        
        return True
    
    def compute_min_distance(self, bbox1, bbox2):
        """İki bbox arasındaki minimum mesafeyi hesapla"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # bbox köşe noktaları
        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1
        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2
        
        # Yatay ve dikey mesafeler
        dx = max(0, max(left1 - right2, left2 - right1))
        dy = max(0, max(top1 - bottom2, top2 - bottom1))
        
        return np.sqrt(dx**2 + dy**2)
    
    def merge_blobs(self, blobs, flow):
        """Benzer akışlı blobları birleştir (Bölüm III-B)"""
        if len(blobs) < 2:
            return blobs, {}
        
        merged_map = {}
        merged_indices = set()
        
        # Her blob çifti için
        for i in range(len(blobs)):
            if i in merged_indices:
                continue
                
            flow_i = self.compute_blob_flow(flow, blobs[i]['bbox'])
            
            for j in range(i+1, len(blobs)):
                if j in merged_indices:
                    continue
                
                flow_j = self.compute_blob_flow(flow, blobs[j]['bbox'])
                
                # Birleştirme koşullarını kontrol et
                if self.check_merge_conditions(blobs[i], blobs[j], flow_i, flow_j):
                    # Blobları birleştir (union)
                    x1, y1, w1, h1 = blobs[i]['bbox']
                    x2, y2, w2, h2 = blobs[j]['bbox']
                    
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_x2 = max(x1 + w1, x2 + w2)
                    new_y2 = max(y1 + h1, y2 + h2)
                    
                    blobs[i]['bbox'] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
                    merged_indices.add(j)
                    merged_map[i] = True
        
        # Birleştirilmemiş blobları tut
        result_blobs = [blobs[i] for i in range(len(blobs)) if i not in merged_indices]
        
        return result_blobs, merged_map
    
    def separate_opposite_flows(self, blobs, flow):
        """Zıt yönlü objeleri ayır (Bölüm III-C)"""
        separated_map = {}
        result_blobs = []
        
        for idx, blob in enumerate(blobs):
            x, y, w, h = blob['bbox']
            blob_flow = flow[y:y+h, x:x+w]
            
            if blob_flow.size == 0 or w < 10 or h < 10:
                result_blobs.append(blob)
                continue
            
            # Flow vektörlerini düzleştir
            flow_vectors = blob_flow.reshape(-1, 2)
            
            # K-means ile 3 cluster
            try:
                kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(flow_vectors)
                labels = kmeans.labels_.reshape(h, w)
                
                # Her cluster için bounding box
                boxes = []
                for k in range(3):
                    mask = (labels == k).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        bx, by, bw, bh = cv2.boundingRect(largest_contour)
                        boxes.append((bx, by, bw, bh))
                        
                # İki kutuyu ayır (Denklem 8)
                if len(boxes) >= 2:
                    separated = False
                    for i in range(len(boxes)):
                        for j in range(i+1, len(boxes)):
                            ratio = self.compute_intersection_ratio(boxes[i], boxes[j])
                            
                            if ratio < self.intersection_threshold:
                                # Açıları kontrol et
                                flow1 = self.compute_blob_flow(flow, 
                                    (x+boxes[i][0], y+boxes[i][1], boxes[i][2], boxes[i][3]))
                                flow2 = self.compute_blob_flow(flow, 
                                    (x+boxes[j][0], y+boxes[j][1], boxes[j][2], boxes[j][3]))
                                
                                angle_diff = abs(flow1['angle'] - flow2['angle'])
                                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                                
                                if angle_diff > self.angle_threshold:
                                    # İki obje olarak ayır
                                    blob1 = blob.copy()
                                    blob1['bbox'] = (x+boxes[i][0], y+boxes[i][1], boxes[i][2], boxes[i][3])
                                    blob2 = blob.copy()
                                    blob2['bbox'] = (x+boxes[j][0], y+boxes[j][1], boxes[j][2], boxes[j][3])
                                    
                                    result_blobs.extend([blob1, blob2])
                                    separated_map[idx] = True
                                    separated = True
                                    break
                        if separated:
                            break
                    
                    if not separated:
                        result_blobs.append(blob)
                else:
                    result_blobs.append(blob)
                    
            except:
                result_blobs.append(blob)
        
        return result_blobs, separated_map
    
    def compute_intersection_ratio(self, box1, box2):
        """İki kutu arasındaki kesişim oranını hesapla (Denklem 8)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Kesişim alanı
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        min_area = min(w1 * h1, w2 * h2)
        
        return intersection / min_area if min_area > 0 else 0
    
    def process_edges(self, frame, blob, background):
        """Kenar işleme (Bölüm III-D)"""
        x, y, w, h = blob['bbox']
        
        # Frame ve background'dan ROI al
        frame_roi = frame[y:y+h, x:x+w]
        bg_roi = background[y:y+h, x:x+w]
        
        if frame_roi.size == 0 or bg_roi.size == 0:
            return []
        
        # Gri tonlamaya çevir
        if len(frame_roi.shape) == 3:
            frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_roi
            
        if len(bg_roi.shape) == 3:
            bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        else:
            bg_gray = bg_roi
        
        # Canny eşikleri (Denklem 9-10)
        median_val = np.median(frame_gray)
        th = (1 + self.sigma) * median_val
        tl = (1 - self.sigma) * median_val
        
        # Kenarları bul
        edges_frame = cv2.Canny(frame_gray, tl, th)
        edges_bg = cv2.Canny(bg_gray, tl, th)
        
        # XOR ile sadece ön plan kenarlarını al
        edges_fg = cv2.bitwise_xor(edges_frame, edges_bg)
        
        # Konturları bul
        contours, _ = cv2.findContours(edges_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Manhattan mesafesi ile grupla
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                bx, by, bw, bh = cv2.boundingRect(contour)
                boxes.append((x + bx, y + by, bw, bh))
        
        return boxes
    
    def decision_algorithm(self, blob, edge_boxes, flow_box, separated_map, merged_map, blob_idx):
        """Hangi kutuları tutacağımıza karar ver (Bölüm III-E)"""
        # 1. Flow separation yapıldıysa, o kutuları kullan
        if blob_idx in separated_map:
            return [flow_box]
        
        # 2. Hem merge hem edge varsa, edge'i tercih et
        if blob_idx in merged_map and len(edge_boxes) > 0:
            return edge_boxes
        
        # 3. Edge 4+ kutu verdiyse, flow'u kullan
        if len(edge_boxes) >= 4:
            return [flow_box]
        
        # 4. Edge 2 kutu, flow 1 kutu verdiyse
        if len(edge_boxes) == 2 and flow_box is not None:
            # Alan oranını hesapla (Denklem 11)
            union_area = self.compute_union_area(edge_boxes[0], edge_boxes[1])
            flow_area = flow_box[2] * flow_box[3]
            
            ratio = union_area / flow_area if flow_area > 0 else 0
            
            if ratio <= self.area_ratio_threshold:
                return [flow_box]
            else:
                return edge_boxes
        
        # 5. Diğer durumlarda edge'i tercih et
        if len(edge_boxes) > 0:
            return edge_boxes
        else:
            return [flow_box] if flow_box is not None else []
    
    def compute_union_area(self, box1, box2):
        """İki kutunun union alanını hesapla"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        new_x = min(x1, x2)
        new_y = min(y1, y2)
        new_x2 = max(x1 + w1, x2 + w2)
        new_y2 = max(y1 + h1, y2 + h2)
        
        return (new_x2 - new_x) * (new_y2 - new_y)
    
    def create_final_image(self, frame_shape, final_boxes):
        """Yeni ön plan görüntüsünü oluştur (Bölüm III-F)"""
        h, w = frame_shape[:2]
        result = np.zeros((h, w), dtype=np.uint8)
        
        for box in final_boxes:
            x, y, bw, bh = box
            # XOR ile çiz (kesişimler siyah olur)
            roi = result[y:y+bh, x:x+bw]
            result[y:y+bh, x:x+bw] = cv2.bitwise_xor(roi, np.ones_like(roi) * 255)
        
        # Dilasyon uygula
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.dilate(result, kernel, iterations=1)
        
        return result
    
    def process_frame(self, frame, prev_gray, curr_gray, show_detailed=False):
        """Bir frame'i işle - ana fonksiyon"""
        # 1. Arka planı güncelle
        background = self.update_background(frame)
        
        # 2. ViBe ile ön plan bloblarını al
        blobs, fg_mask = self.get_foreground_blobs(frame)
        
        # ViBe'ın ham çıktısını kaydet (karşılaştırma için)
        vibe_raw_mask = fg_mask.copy()
        
        if len(blobs) == 0:
            if show_detailed:
                return fg_mask, [], vibe_raw_mask, background
            return fg_mask, []
        
        # 3. Optik akışı hesapla (HIZLI VERSİYON)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            0.5, 2, 10, 2, 5, 1.1, 0  # Daha az piramit seviyesi = daha hızlı
        )
        
        # 4. Blobları birleştir
        merged_blobs, merged_map = self.merge_blobs(blobs, flow)
        
        # 5. Zıt yönlü objeleri ayır
        separated_blobs, separated_map = self.separate_opposite_flows(merged_blobs, flow)
        
        # 6. Her blob için edge processing ve karar
        final_boxes = []
        for idx, blob in enumerate(separated_blobs):
            # Edge processing
            edge_boxes = self.process_edges(frame, blob, background)
            
            # Karar algoritması
            selected_boxes = self.decision_algorithm(
                blob, edge_boxes, blob['bbox'], 
                separated_map, merged_map, idx
            )
            
            final_boxes.extend(selected_boxes)
        
        # 7. Yeni ön plan görüntüsünü oluştur
        final_mask = self.create_final_image(frame.shape, final_boxes)
        
        if show_detailed:
            return final_mask, final_boxes, vibe_raw_mask, background
        return final_mask, final_boxes


def main():
    import os
    import time
    
    # Video yolunu kontrol et
    video_path = "video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video bulunamadı: {video_path}")
        print("📁 Mevcut dizin:", os.getcwd())
        print("📹 Lütfen video yolunu kontrol edin veya başka bir video deneyin")
        
        # Alternatif: webcam kullan
        choice = input("\nWebcam kullanmak ister misiniz? (e/h): ")
        if choice.lower() == 'e':
            video_path = 0
        else:
            return
    
    # Video aç
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Video açılamadı!")
        return
    
    # Video özellikleri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video açıldı!")
    print(f"📊 Çözünürlük: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    print(f"📹 Toplam frame: {total_frames}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('result_vibe.avi', fourcc, fps, (width*2, height))
    
    # Detector'ı başlat (ViBe ile)
    print("🔧 ViBe detector başlatılıyor...")
    detector = ObjectDetectionImprover()
    
    # İlk frame'i oku
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ İlk frame okunamadı!")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    print("🚀 ViBe ile işleme başlıyor... (ESC ile çıkış)")
    print("⏳ İlk birkaç frame'de ViBe modeli eğitiliyor...")
    print("-" * 60)
    
    frame_count = 0
    start_time = time.time()
    process_times = []
    
    # Görselleştirme modu (1 tuşu ile değişir)
    visualization_mode = 0  # 0: Normal, 1: ViBe Karşılaştırma, 2: Detaylı
    
    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Frame'i işle
        result = detector.process_frame(frame, prev_gray, curr_gray, show_detailed=True)
        improved_mask, detected_boxes, vibe_raw_mask, background = result
        
        frame_process_time = time.time() - frame_start
        process_times.append(frame_process_time)
        
        # Görselleştirme moduna göre ekranı hazırla
        if visualization_mode == 0:
            # Normal Mod: Original + Detections | Improved Mask
            result_frame = frame.copy()
            for box in detected_boxes:
                x, y, w, h = box
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            improved_mask_colored = cv2.cvtColor(improved_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([result_frame, improved_mask_colored])
            
            mode_text = "Mode: NORMAL (1:ViBe 2:Detail)"
            
        elif visualization_mode == 1:
            # ViBe Karşılaştırma Modu
            # Üst satır: Original | ViBe Ham Çıktı
            # Alt satır: Improved Detections | Improved Mask
            
            vibe_colored = cv2.cvtColor(vibe_raw_mask, cv2.COLOR_GRAY2BGR)
            top_row = np.hstack([frame, vibe_colored])
            
            result_frame = frame.copy()
            for box in detected_boxes:
                x, y, w, h = box
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            improved_mask_colored = cv2.cvtColor(improved_mask, cv2.COLOR_GRAY2BGR)
            bottom_row = np.hstack([result_frame, improved_mask_colored])
            
            # Etiketler ekle
            cv2.putText(top_row, "ORIGINAL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(top_row, "ViBe RAW OUTPUT", (width+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bottom_row, "IMPROVED DETECTIONS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bottom_row, "IMPROVED MASK", (width+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            combined = np.vstack([top_row, bottom_row])
            mode_text = "Mode: ViBe COMPARISON (0:Normal 2:Detail)"
            
        else:  # visualization_mode == 2
            # Detaylı Mod: 4 panel
            # Original | ViBe Raw | Background | Improved
            
            vibe_colored = cv2.cvtColor(vibe_raw_mask, cv2.COLOR_GRAY2BGR)
            background_resized = cv2.resize(background, (width, height))
            
            result_frame = frame.copy()
            for box in detected_boxes:
                x, y, w, h = box
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            improved_mask_colored = cv2.cvtColor(improved_mask, cv2.COLOR_GRAY2BGR)
            
            # Etiketler
            cv2.putText(frame, "ORIGINAL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vibe_colored, "ViBe OUTPUT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(background_resized, "BACKGROUND MODEL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(improved_mask_colored, "IMPROVED OUTPUT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            top_row = np.hstack([frame, vibe_colored])
            bottom_row = np.hstack([background_resized, improved_mask_colored])
            combined = np.vstack([top_row, bottom_row])
            
            mode_text = "Mode: DETAILED (0:Normal 1:ViBe)"
        
        # İstatistikleri ekle
        avg_fps = 1.0 / np.mean(process_times[-30:]) if len(process_times) > 0 else 0
        stats_text = f'Frame: {frame_count}/{total_frames} | Objects: {len(detected_boxes)} | FPS: {avg_fps:.1f}'
        
        # Stats ve mode bilgisini ekle
        cv2.putText(combined, stats_text, (10, combined.shape[0]-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, mode_text, (10, combined.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Göster
        cv2.imshow('ViBe Object Detection Improver', combined)
        out.write(combined)
        
        # Bir sonraki frame için güncelle
        prev_gray = curr_gray
        
        # Tuş kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n⏹️  Kullanıcı tarafından durduruldu")
            break
        elif key == 32:  # SPACE
            print("⏸️  Duraklatıldı... (Devam için herhangi bir tuşa basın)")
            cv2.waitKey(0)
        elif key == ord('0'):
            visualization_mode = 0
            print("📺 Normal Mod")
        elif key == ord('1'):
            visualization_mode = 1
            print("📺 ViBe Karşılaştırma Modu")
        elif key == ord('2'):
            visualization_mode = 2
            print("📺 Detaylı Mod")
        
        # Her 10 frame'de konsola yazdır
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count)
            print(f"📊 Frame: {frame_count}/{total_frames} | Objeler: {len(detected_boxes)} | "
                  f"İşlem Hızı: {frame_process_time*1000:.1f}ms | "
                  f"Ortalama FPS: {avg_fps:.1f} | "
                  f"Kalan Süre: {eta:.0f}s")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Toplam {frame_count} frame işlendi.")
    print(f"⏱️  Toplam süre: {total_time:.1f} saniye")
    print(f"⚡ Ortalama FPS: {frame_count/total_time:.2f}")
    print(f"📹 Sonuç video: result_vibe.avi")
    print(f"{'='*60}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()