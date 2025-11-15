import numpy as np
import cv2

def draw_flow(img, flow, step=16):
    """Optical flow vektorlerini cizer"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:      
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    """Optical flow'u HSV renk uzayinda gosterir"""
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(90/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

class ViBe:
    """ViBe Background Subtraction implementasyonu"""
    def __init__(self, num_samples=20, min_matches=2, radius=20, subsample_factor=16):
        self.num_samples = num_samples
        self.min_matches = min_matches
        self.radius = radius
        self.subsample_factor = subsample_factor
        self.samples = None
        
    def initialize(self, frame):
        """İlk frame ile modeli initialize et"""
        h, w = frame.shape[:2]
        self.samples = np.zeros((h, w, self.num_samples), dtype=np.uint8)
        
        # Her pixel icin random komsu pixellerden ornekler al
        for i in range(self.num_samples):
            self.samples[:, :, i] = frame
            
    def update(self, frame, mask):
        """Background modelini guncelle"""
        if self.samples is None:
            self.initialize(frame)
            return
            
        h, w = frame.shape[:2]
        
        # Sadece background pixelleri guncelle
        update_mask = (mask == 0)
        
        for y in range(h):
            for x in range(w):
                if update_mask[y, x] and np.random.randint(0, self.subsample_factor) == 0:
                    idx = np.random.randint(0, self.num_samples)
                    self.samples[y, x, idx] = frame[y, x]
                    
    def apply(self, frame):
        """Foreground maskesini hesapla"""
        if self.samples is None:
            self.initialize(frame)
            return np.zeros(frame.shape[:2], dtype=np.uint8)
            
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                matches = 0
                for i in range(self.num_samples):
                    if abs(int(frame[y, x]) - int(self.samples[y, x, i])) < self.radius:
                        matches += 1
                        if matches >= self.min_matches:
                            break
                            
                if matches < self.min_matches:
                    mask[y, x] = 255
                    
        return mask

def merge_blobs_with_flow(blobs, flow, distance_threshold=7, angle_threshold=np.pi/2):
    """
    Makaledeki gibi benzer optical flow'a sahip yakin bloblari birlestir
    """
    merged = []
    used = set()
    
    for i, blob_i in enumerate(blobs):
        if i in used:
            continue
            
        x_i, y_i, w_i, h_i = blob_i
        cx_i, cy_i = x_i + w_i//2, y_i + h_i//2
        
        # Bu blob icin flow hesapla
        flow_i = flow[cy_i, cx_i]
        mag_i = np.sqrt(flow_i[0]**2 + flow_i[1]**2)
        ang_i = np.arctan2(flow_i[1], flow_i[0])
        
        merged_blob = blob_i
        
        for j, blob_j in enumerate(blobs[i+1:], i+1):
            if j in used:
                continue
                
            x_j, y_j, w_j, h_j = blob_j
            cx_j, cy_j = x_j + w_j//2, y_j + h_j//2
            
            # Mesafe kontrolu
            dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            if dist > distance_threshold:
                continue
                
            # Flow benzerlik kontrolu
            flow_j = flow[cy_j, cx_j]
            mag_j = np.sqrt(flow_j[0]**2 + flow_j[1]**2)
            ang_j = np.arctan2(flow_j[1], flow_j[0])
            
            # Aci farki
            ang_diff = abs(ang_i - ang_j)
            if ang_diff > np.pi:
                ang_diff = 2*np.pi - ang_diff
                
            # Magnitude farki
            mag_diff = abs(mag_i - mag_j)
            
            if ang_diff < angle_threshold and mag_diff < mag_i * 0.5:
                # Birlestir
                x_new = min(x_i, x_j)
                y_new = min(y_i, y_j)
                w_new = max(x_i + w_i, x_j + w_j) - x_new
                h_new = max(y_i + h_i, y_j + h_j) - y_new
                merged_blob = (x_new, y_new, w_new, h_new)
                used.add(j)
                
        merged.append(merged_blob)
        used.add(i)
        
    return merged

def separate_opposite_flow(blob, flow):
    """
    Makaledeki gibi zit yonde hareket eden objeleri ayir
    """
    x, y, w, h = blob
    if w < 20 or h < 20:  # Cok kucuk bloblar icin yapma
        return [blob]
    
    # Blob icindeki flow vektorlerini al
    flow_region = flow[y:y+h, x:x+w]
    flow_vectors = flow_region.reshape(-1, 2)
    
    # K-means ile 3 cluster'a ayir
    if len(flow_vectors) < 3:
        return [blob]
        
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(flow_vectors.astype(np.float32), 3, None, 
                                     criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Her cluster icin bounding box hesapla
    labels = labels.reshape(h, w)
    boxes = []
    
    for cluster_id in range(3):
        mask = (labels == cluster_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            boxes.append((x + bx, y + by, bw, bh))
    
    # Zit yonde hareket edip etmedigini kontrol et
    if len(boxes) >= 2:
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                cx_i = boxes[i][0] + boxes[i][2]//2
                cy_i = boxes[i][1] + boxes[i][3]//2
                cx_j = boxes[j][0] + boxes[j][2]//2
                cy_j = boxes[j][1] + boxes[j][3]//2
                
                flow_i = flow[cy_i, cx_i]
                flow_j = flow[cy_j, cx_j]
                
                ang_i = np.arctan2(flow_i[1], flow_i[0])
                ang_j = np.arctan2(flow_j[1], flow_j[0])
                
                ang_diff = abs(ang_i - ang_j)
                if ang_diff > np.pi:
                    ang_diff = 2*np.pi - ang_diff
                    
                # Eger zit yonlerde hareket ediyorlarsa
                if ang_diff > 2*np.pi/3:  # ~120 derece
                    return [boxes[i], boxes[j]]
    
    return [blob]

def main():
    # Video yukle
    cam = cv2.VideoCapture("video.mp4")
    
    if not cam.isOpened():
        print("Video acilamadi! Lutfen 'data/atrium.avi' dosyasinin var oldugundan emin olun.")
        return
    
    p = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    l = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    
    # ViBe background subtractor'u olustur
    vibe = ViBe(num_samples=20, min_matches=2, radius=20)
    
    # Video writer
    out = cv2.VideoWriter('result_vibe_flow.avi', 
                          cv2.VideoWriter_fourcc('M','J','P','G'), 
                          fps, (p, l))
    
    ret, prev_frame = cam.read()
    if not ret:
        print("İlk frame okunamadi!")
        return
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    vibe.initialize(prev_gray)
    
    frame_count = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ViBe ile foreground maske hesapla
        fg_mask = vibe.apply(gray)
        
        # Morfolojik islemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Optical flow hesapla
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Contour bul
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Bounding boxlari al
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500 and area < 50000:  # Alan filtreleme
                x, y, w, h = cv2.boundingRect(c)
                if w < 400 and h < 600:  # Boyut filtreleme
                    blobs.append((x, y, w, h))
        
        # Benzer flow'a sahip bloblari birlestir
        if len(blobs) > 1:
            blobs = merge_blobs_with_flow(blobs, flow)
        
        # Zit yonde hareket eden objeleri ayir
        final_blobs = []
        for blob in blobs:
            separated = separate_opposite_flow(blob, flow)
            final_blobs.extend(separated)
        
        # Sonuclari ciz
        result = frame.copy()
        for (x, y, w, h) in final_blobs:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Merkez noktayi ciz
            cx, cy = x + w//2, y + h//2
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        
        # ViBe modelini guncelle
        vibe.update(gray, fg_mask)
        
        # Gorsellestirme
        flow_vis = draw_hsv(flow)
        
        cv2.imshow('Original', frame)
        cv2.imshow('ViBe Foreground', fg_mask)
        cv2.imshow('Optical Flow', flow_vis)
        cv2.imshow('Result', result)
        
        out.write(result)
        
        prev_gray = gray
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"İşlenen frame: {frame_count}")
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # ESC
            break
    
    cam.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Toplam {frame_count} frame işlendi. Sonuç: result_vibe_flow.avi")

if __name__ == "__main__":
    main()