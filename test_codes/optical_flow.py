"""
Optical Flow Tabanlı Nesne Takip Sistemi - Test Kodu
Bu kod AI kullanmadan geleneksel bilgisayar görüsü teknikleriyle nesne takibi yapar.

Gerekli Kütüphaneler:
pip install opencv-python numpy scikit-image scipy
"""

import cv2
import numpy as np
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
from skimage import transform as tf
from scipy import signal

# ===================== 1. FEATURE DETECTION =====================
def getFeatures(img, bbox, use_shi=False):
    """Bounding box içindeki köşe noktalarını tespit eder"""
    n_object = np.shape(bbox)[0]
    N = 0
    temp = np.empty((n_object,), dtype=np.ndarray)
    
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:].astype(int))
        roi = img[ymin:ymin+boxh, xmin:xmin+boxw]
        
        if use_shi:
            corner_response = corner_shi_tomasi(roi)
        else:
            corner_response = corner_harris(roi)
            
        coordinates = peak_local_max(corner_response, num_peaks=20, exclude_border=2)
        coordinates[:,1] += xmin
        coordinates[:,0] += ymin
        temp[i] = coordinates
        
        if coordinates.shape[0] > N:
            N = coordinates.shape[0]
    
    x = np.full((N, n_object), -1)
    y = np.full((N, n_object), -1)
    
    for i in range(n_object):
        n_feature = temp[i].shape[0]
        x[0:n_feature, i] = temp[i][:,1]
        y[0:n_feature, i] = temp[i][:,0]
    
    return x, y

# ===================== 2. INTERPOLATION =====================
WINDOW_SIZE = 25

def interp2(v, xq, yq):
    """Bilinear interpolasyon"""
    dim_input = 1
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()
    
    h = v.shape[0]
    w = v.shape[1]
    
    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)
    
    x_floor = np.clip(x_floor, 0, w-1)
    y_floor = np.clip(y_floor, 0, h-1)
    x_ceil = np.clip(x_ceil, 0, w-1)
    y_ceil = np.clip(y_ceil, 0, h-1)
    
    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]
    
    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw
    
    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    
    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4
    
    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val

# ===================== 3. OPTICAL FLOW =====================
def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    """Lucas-Kanade optik akış ile özellik noktasının yeni konumunu hesaplar"""
    X = startX
    Y = startY
    
    mesh_x, mesh_y = np.meshgrid(np.arange(WINDOW_SIZE), np.arange(WINDOW_SIZE))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    mesh_x_flat_fix = mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix = mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
    coor_fix = np.vstack((mesh_x_flat_fix, mesh_y_flat_fix))
    
    I1_value = interp2(img1_gray, coor_fix[[0],:], coor_fix[[1],:])
    Ix_value = interp2(Ix, coor_fix[[0],:], coor_fix[[1],:])
    Iy_value = interp2(Iy, coor_fix[[0],:], coor_fix[[1],:])
    I = np.vstack((Ix_value, Iy_value))
    A = I.dot(I.T)
    
    for _ in range(15):
        mesh_x_flat = mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat = mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor = np.vstack((mesh_x_flat, mesh_y_flat))
        
        I2_value = interp2(img2_gray, coor[[0],:], coor[[1],:])
        Ip = (I2_value - I1_value).reshape((-1, 1))
        b = -I.dot(Ip)
        
        solution = np.linalg.inv(A).dot(b)
        X += solution[0, 0]
        Y += solution[1, 0]
    
    return X, Y

def estimateAllTranslation(startXs, startYs, img1, img2):
    """Tüm özellik noktaları için optik akış hesaplar"""
    I = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I, (5, 5), 0.2)
    Iy, Ix = np.gradient(I.astype(float))
    
    startXs_flat = startXs.flatten()
    startYs_flat = startYs.flatten()
    newXs = np.full(startXs_flat.shape, -1, dtype=float)
    newYs = np.full(startYs_flat.shape, -1, dtype=float)
    
    for i in range(np.size(startXs)):
        if startXs_flat[i] != -1:
            newXs[i], newYs[i] = estimateFeatureTranslation(
                startXs_flat[i], startYs_flat[i], Ix, Iy, img1, img2
            )
    
    newXs = np.reshape(newXs, startXs.shape)
    newYs = np.reshape(newYs, startYs.shape)
    return newXs, newYs

# ===================== 4. GEOMETRIC TRANSFORMATION =====================
def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    """Geometrik dönüşüm uygulayarak bounding box'ları günceller"""
    n_object = bbox.shape[0]
    newbbox = np.zeros_like(bbox)
    Xs = newXs.copy()
    Ys = newYs.copy()
    
    for obj_idx in range(n_object):
        startXs_obj = startXs[:, [obj_idx]]
        startYs_obj = startYs[:, [obj_idx]]
        newXs_obj = newXs[:, [obj_idx]]
        newYs_obj = newYs[:, [obj_idx]]
        
        desired_points = np.hstack((startXs_obj, startYs_obj))
        actual_points = np.hstack((newXs_obj, newYs_obj))
        
        t = tf.SimilarityTransform()
        t.estimate(dst=actual_points, src=desired_points)
        mat = t.params
        
        # RANSAC benzeri outlier tespiti
        THRES = 1
        projected = mat.dot(np.vstack((
            desired_points.T.astype(float),
            np.ones([1, np.shape(desired_points)[0]])
        )))
        distance = np.square(projected[0:2,:].T - actual_points).sum(axis=1)
        
        actual_inliers = actual_points[distance < THRES]
        desired_inliers = desired_points[distance < THRES]
        
        if np.shape(desired_inliers)[0] < 4:
            print(f'  Çok az inlier noktası bulundu (obj {obj_idx})')
            actual_inliers = actual_points
            desired_inliers = desired_points
        
        t.estimate(dst=actual_inliers, src=desired_inliers)
        mat = t.params
        
        coords = np.vstack((bbox[obj_idx,:,:].T, np.array([1,1,1,1])))
        new_coords = mat.dot(coords)
        newbbox[obj_idx,:,:] = new_coords[0:2,:].T
        
        Xs[distance >= THRES, obj_idx] = -1
        Ys[distance >= THRES, obj_idx] = -1
    
    return Xs, Ys, newbbox

# ===================== 5. MAIN TRACKING FUNCTION =====================
def optical_flow_tracking(video_path, n_frames=100, draw_bb=True):
    """Ana takip fonksiyonu"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Video açılamadı!")
        return
    
    # İlk frame'i oku
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Video okuma hatası!")
        return
    
    print("✅ Video başarıyla açıldı!")
    print(f"📹 Çözünürlük: {first_frame.shape[1]}x{first_frame.shape[0]}")
    
    # Kullanıcıdan nesne seçimini al
    if draw_bb:
        print("\n📌 Lütfen takip etmek istediğiniz nesneyi seçin:")
        print("   - Fare ile dikdörtgen çizin")
        print("   - SPACE/ENTER ile onaylayın")
        print("   - ESC ile iptal edin\n")
        
        bbox = cv2.selectROI("Nesne Seç", first_frame, False, False)
        cv2.destroyWindow("Nesne Seç")
        
        if bbox[2] == 0 or bbox[3] == 0:
            print("❌ Geçerli bir nesne seçilmedi!")
            return
        
        xmin, ymin, boxw, boxh = bbox
        bboxs = np.array([[[xmin, ymin], 
                          [xmin+boxw, ymin],
                          [xmin, ymin+boxh],
                          [xmin+boxw, ymin+boxh]]]).astype(float)
    else:
        # Varsayılan bbox (ekranın ortası)
        h, w = first_frame.shape[:2]
        bboxs = np.array([[[w//2-50, h//2-50],
                          [w//2+50, h//2-50],
                          [w//2-50, h//2+50],
                          [w//2+50, h//2+50]]]).astype(float)
    
    # Video writer oluştur
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, 
                         (first_frame.shape[1], first_frame.shape[0]))
    
    # İlk frame için feature noktaları bul
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    startXs, startYs = getFeatures(cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY), 
                                   bboxs, use_shi=False)
    
    print(f"\n🚀 Takip başlıyor... ({n_frames} frame işlenecek)")
    print("=" * 50)
    
    frame_idx = 1
    while frame_idx < n_frames:
        ret, curr_frame = cap.read()
        if not ret:
            print(f"\n⚠️  Video sonu ({frame_idx}. frame'de)")
            break
        
        # Optik akış ile yeni pozisyonları hesapla
        newXs, newYs = estimateAllTranslation(startXs, startYs, prev_frame, curr_frame)
        
        # Geometrik dönüşüm uygula
        startXs, startYs, bboxs = applyGeometricTransformation(
            startXs, startYs, newXs, newYs, bboxs
        )
        
        # Kalan feature sayısını kontrol et
        n_features_left = np.sum(startXs != -1)
        print(f"Frame {frame_idx:3d} | Features: {n_features_left:2d}", end="")
        
        # Feature azaldıysa yenilerini bul
        if n_features_left < 15:
            print(" → Yeni feature'lar üretiliyor...")
            startXs, startYs = getFeatures(
                cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY), bboxs
            )
        else:
            print()
        
        # Görselleştirme
        display_frame = curr_frame.copy()
        n_object = bboxs.shape[0]
        
        for j in range(n_object):
            # Bounding box çiz
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[j,:,:].astype(int))
            cv2.rectangle(display_frame, (xmin, ymin), 
                         (xmin+boxw, ymin+boxh), (0, 255, 0), 2)
            
            # Feature noktalarını çiz
            for k in range(startXs.shape[0]):
                if startXs[k, j] != -1:
                    cv2.circle(display_frame, 
                             (int(startXs[k, j]), int(startYs[k, j])),
                             3, (0, 0, 255), -1)
        
        # Göster ve kaydet
        cv2.imshow('Optical Flow Tracking', display_frame)
        out.write(display_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("\n⚠️  Kullanıcı tarafından durduruldu")
            break
        
        prev_frame = curr_frame
        frame_idx += 1
    
    print("=" * 50)
    print(f"✅ İşlem tamamlandı! Toplam {frame_idx} frame işlendi")
    print(f"💾 Sonuç 'output.avi' dosyasına kaydedildi")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ===================== 6. TEST KODU =====================
if __name__ == "__main__":
    print("=" * 60)
    print("   OPTICAL FLOW NESNE TAKİP SİSTEMİ (AI-FREE)")
    print("=" * 60)
    print("\n📝 Kullanım:")
    print("   1. Video dosyasını belirtin")
    print("   2. Takip edilecek nesneyi seçin")
    print("   3. 'q' tuşu ile çıkış yapın\n")
    
    # Video yolunu buraya yazın
    video_path = "furkan/video.mp4"  # BURAYA KENDİ VİDEONUZU YAZIN
    
    # Örnek videolar için:
    # video_path = 0  # Webcam için
    # video_path = "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
    
    try:
        optical_flow_tracking(
            video_path=video_path,
            n_frames=200,      # İşlenecek frame sayısı
            draw_bb=True       # Manuel nesne seçimi
        )
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        print("\n💡 İpuçları:")
        print("   - Video yolunu kontrol edin")
        print("   - Gerekli kütüphaneleri yükleyin: pip install opencv-python numpy scikit-image scipy")
        print("   - Video formatını kontrol edin (MP4, AVI desteklenir)")