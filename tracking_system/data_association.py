"""
data_association.py - Hungarian Algorithm for Optimal Track-Detection Matching
Paper: Beaupre et al. 2018 + Shantaiya et al. 2015
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    """
    IoU (Intersection over Union) hesaplar
    
    Args:
        bbox1, bbox2: (x, y, w, h) format
    
    Returns:
        iou_score: [0, 1] arası değer
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Koordinatları al
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    
    # Union
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    
    # IoU
    if union_area == 0:
        return 0.0
    
    iou_score = inter_area / union_area
    return iou_score


def euclidean_distance(center1, center2):
    """
    İki merkez nokta arası Euclidean mesafe
    
    Args:
        center1, center2: (x, y) tuple
    
    Returns:
        distance: piksel cinsinden mesafe
    """
    x1, y1 = center1
    x2, y2 = center2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def associate_detections_to_tracks(tracks, detections, 
                                   metric='iou',
                                   distance_threshold=150,
                                   iou_threshold=0.3):
    """
    Hungarian algorithm ile track-detection matching
    
    Args:
        tracks: List of track states (from TrackerManager.get_active_tracks())
        detections: List of detection dicts
        metric: 'iou' veya 'euclidean'
        distance_threshold: Euclidean için max mesafe (piksel)
        iou_threshold: IoU için min overlap
    
    Returns:
        matches: List of (track_idx, detection_idx) tuples
        unmatched_detections: List of detection indices
        unmatched_tracks: List of track indices
    """
    if len(tracks) == 0:
        # Track yok → tüm detection'lar unmatched
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        # Detection yok → tüm track'ler unmatched
        return [], [], list(range(len(tracks)))
    
    # ========== COST MATRIX OLUŞTUR ==========
    num_tracks = len(tracks)
    num_detections = len(detections)
    cost_matrix = np.zeros((num_tracks, num_detections))
    
    for i, track in enumerate(tracks):
        track_bbox = track['bbox']
        track_center = track['center']
        
        for j, detection in enumerate(detections):
            det_bbox = detection['bbox']
            det_center = detection['center']
            
            if metric == 'iou':
                # IoU-based cost (1 - IoU, çünkü Hungarian minimum arar)
                iou_score = iou(track_bbox, det_bbox)
                cost_matrix[i, j] = 1 - iou_score
            
            elif metric == 'euclidean':
                # Euclidean distance
                dist = euclidean_distance(track_center, det_center)
                cost_matrix[i, j] = dist
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    # ========== HUNGARIAN ALGORITHM ==========
    # scipy'nin linear_sum_assignment: optimal assignment bulur
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    
    # ========== THRESHOLD FİLTRELEME ==========
    matches = []
    unmatched_detections = list(range(num_detections))
    unmatched_tracks = list(range(num_tracks))
    
    for track_idx, det_idx in zip(track_indices, detection_indices):
        cost = cost_matrix[track_idx, det_idx]
        
        # Threshold kontrolü
        if metric == 'iou':
            iou_score = 1 - cost
            if iou_score >= iou_threshold:
                matches.append((track_idx, det_idx))
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
        
        elif metric == 'euclidean':
            distance = cost
            if distance <= distance_threshold:
                matches.append((track_idx, det_idx))
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
    
    return matches, unmatched_detections, unmatched_tracks


def associate_with_motion(tracks, detections, 
                         distance_threshold=150,
                         velocity_weight=0.3):
    """
    Motion-aware association (Kalman prediction + velocity)
    
    Daha gelişmiş: Hem pozisyon hem de velocity benzerliğine bakar
    
    Args:
        tracks: List of track states
        detections: List of detection dicts
        distance_threshold: Max distance threshold
        velocity_weight: Velocity benzerliğinin ağırlığı [0, 1]
    
    Returns:
        matches, unmatched_detections, unmatched_tracks
    """
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))
    
    # Cost matrix
    num_tracks = len(tracks)
    num_detections = len(detections)
    cost_matrix = np.zeros((num_tracks, num_detections))
    
    for i, track in enumerate(tracks):
        track_center = track['center']
        track_vx, track_vy = track['velocity']
        
        for j, detection in enumerate(detections):
            det_center = detection['center']
            det_motion = detection.get('motion', {'vx': 0, 'vy': 0})
            det_vx = det_motion['vx']
            det_vy = det_motion['vy']
            
            # Position distance
            pos_dist = euclidean_distance(track_center, det_center)
            
            # Velocity distance
            vel_dist = np.sqrt((track_vx - det_vx)**2 + (track_vy - det_vy)**2)
            
            # Combined cost (weighted)
            cost_matrix[i, j] = (1 - velocity_weight) * pos_dist + velocity_weight * vel_dist
    
    # Hungarian
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    
    # Filter by threshold
    matches = []
    unmatched_detections = list(range(num_detections))
    unmatched_tracks = list(range(num_tracks))
    
    for track_idx, det_idx in zip(track_indices, detection_indices):
        cost = cost_matrix[track_idx, det_idx]
        
        if cost <= distance_threshold:
            matches.append((track_idx, det_idx))
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)
            if track_idx in unmatched_tracks:
                unmatched_tracks.remove(track_idx)
    
    return matches, unmatched_detections, unmatched_tracks


# ============================================================
# TEST KODU
# ============================================================
def test_data_association():
    """
    Data association fonksiyonlarını test et
    """
    print("="*60)
    print("DATA ASSOCIATION TEST")
    print("="*60)
    
    # Fake tracks
    tracks = [
        {'bbox': (100, 100, 50, 50), 'center': (125, 125), 'velocity': (2, 1)},
        {'bbox': (200, 200, 60, 60), 'center': (230, 230), 'velocity': (-1, 2)},
        {'bbox': (300, 300, 55, 55), 'center': (327, 327), 'velocity': (0, -2)},
    ]
    
    # Fake detections (biraz hareket etmiş)
    detections = [
        {'bbox': (102, 101, 50, 50), 'center': (127, 126), 'motion': {'vx': 2, 'vy': 1}},  # Track 0
        {'bbox': (199, 202, 60, 60), 'center': (229, 232), 'motion': {'vx': -1, 'vy': 2}}, # Track 1
        {'bbox': (500, 500, 40, 40), 'center': (520, 520), 'motion': {'vx': 3, 'vy': 0}},  # Yeni nesne
    ]
    
    print("\n📊 Test Case:")
    print(f"   Tracks: {len(tracks)}")
    print(f"   Detections: {len(detections)}")
    
    # Test 1: IoU-based
    print("\n🔹 Test 1: IoU-based Association")
    matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
        tracks, detections, metric='iou', iou_threshold=0.3
    )
    print(f"   Matches: {len(matches)} → {matches}")
    print(f"   Unmatched Detections: {unmatched_dets}")
    print(f"   Unmatched Tracks: {unmatched_tracks}")
    
    # Test 2: Euclidean-based
    print("\n🔹 Test 2: Euclidean-based Association")
    matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
        tracks, detections, metric='euclidean', distance_threshold=150
    )
    print(f"   Matches: {len(matches)} → {matches}")
    print(f"   Unmatched Detections: {unmatched_dets}")
    print(f"   Unmatched Tracks: {unmatched_tracks}")
    
    # Test 3: Motion-aware
    print("\n🔹 Test 3: Motion-aware Association")
    matches, unmatched_dets, unmatched_tracks = associate_with_motion(
        tracks, detections, distance_threshold=150, velocity_weight=0.3
    )
    print(f"   Matches: {len(matches)} → {matches}")
    print(f"   Unmatched Detections: {unmatched_dets}")
    print(f"   Unmatched Tracks: {unmatched_tracks}")
    
    print("\n" + "="*60)
    print("✅ Data Association Test Complete")
    print("="*60)


def associate_hybrid(tracks, detections, 
                    iou_threshold=0.3,
                    distance_threshold=100):
    """
    Hybrid association: IoU + Euclidean fallback
    
    Strateji:
    1. Önce IoU-based matching yap
    2. Unmatched track'ler için Euclidean dene
    
    Returns:
        matches, unmatched_detections, unmatched_tracks
    """
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))
    
    # ========== PHASE 1: IoU-based ==========
    matches_iou, unmatched_dets_iou, unmatched_tracks_iou = associate_detections_to_tracks(
        tracks, detections, 
        metric='iou', 
        iou_threshold=iou_threshold
    )
    
    # ========== PHASE 2: Euclidean fallback ==========
    # Unmatched track'ler için Euclidean dene
    if len(unmatched_tracks_iou) > 0 and len(unmatched_dets_iou) > 0:
        # Unmatched track'leri ve detection'ları al
        unmatched_tracks_subset = [tracks[i] for i in unmatched_tracks_iou]
        unmatched_dets_subset = [detections[i] for i in unmatched_dets_iou]
        
        matches_eucl, unmatched_dets_eucl, unmatched_tracks_eucl = associate_detections_to_tracks(
            unmatched_tracks_subset,
            unmatched_dets_subset,
            metric='euclidean',
            distance_threshold=distance_threshold
        )
        
        # Index'leri orijinal listeye map et
        matches_eucl_mapped = []
        for track_idx, det_idx in matches_eucl:
            original_track_idx = unmatched_tracks_iou[track_idx]
            original_det_idx = unmatched_dets_iou[det_idx]
            matches_eucl_mapped.append((original_track_idx, original_det_idx))
        
        # Unmatched'leri güncelle
        unmatched_dets_final = [unmatched_dets_iou[i] for i in unmatched_dets_eucl]
        unmatched_tracks_final = [unmatched_tracks_iou[i] for i in unmatched_tracks_eucl]
        
        # Tüm match'leri birleştir
        all_matches = matches_iou + matches_eucl_mapped
        
        return all_matches, unmatched_dets_final, unmatched_tracks_final
    
    else:
        # Phase 2'ye gerek yok
        return matches_iou, unmatched_dets_iou, unmatched_tracks_iou
    
    
if __name__ == "__main__":
    test_data_association()