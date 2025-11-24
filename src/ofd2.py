import numpy as np
import cv2
from ofd1 import *

def main():
    cam = cv2.VideoCapture("../data/mobese_3.mp4")
    p = int(cam.get(3))
    l = int(cam.get(4))

    ret, prev = cam.read()
    
    if not ret:
        print("Video okunamadı!")
        return
   
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    # MOG2
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )
    
    # Kerneller
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    
    # ========== BİRLEŞTİRME YÖNTEMİ SEÇİN ==========
    merge_method = "and"  # "or", "and", "weighted"
    # ==============================================
    
    frame_count = 0
    
    while True:
        ret, img = cam.read()
        
        if not ret:
            print("Video bitti!")
            break
            
        frame_count += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ========== 1. MOG2 ==========
        fg_mask = backSub.apply(img)
        fg_mask_no_shadow = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        fg_mask_clean = cv2.morphologyEx(fg_mask_no_shadow, cv2.MORPH_OPEN, kernel_open)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel_close)
        
        # ========== 2. OPTICAL FLOW ==========
        flow = cv2.calcOpticalFlowFarneback(
            prevgray, gray, None, 
            0.5, 3, 7, 3, 5, 1.1, 0
        )
        prevgray = gray
        
        gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
        thresh_flow = cv2.threshold(gray1, 3, 255, cv2.THRESH_BINARY)[1]
        thresh_flow = cv2.dilate(thresh_flow, None, iterations=2)
        
        # ========== 3. BİRLEŞTİRME ==========
        if merge_method == "or":
            # YÖNTEM 1: OR (Birleşim)
            merged_mask = cv2.bitwise_or(fg_mask_clean, thresh_flow)
            method_name = "OR (Union)"
            
        elif merge_method == "and":
            # YÖNTEM 2: AND (Kesişim)
            merged_mask = cv2.bitwise_and(fg_mask_clean, thresh_flow)
            method_name = "AND (Intersection)"
            
        elif merge_method == "weighted":
            # YÖNTEM 3: Ağırlıklı Birleşim (MAKALEDEKİ GİBİ)
            merged_mask = cv2.addWeighted(fg_mask_clean, 0.6, thresh_flow, 0.4, 0)
            _, merged_mask = cv2.threshold(merged_mask, 127, 255, cv2.THRESH_BINARY)
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel_close)
            method_name = "Weighted (60% MOG2 + 40% Flow)"
        
        # ========== 4. KONTUR BULMA ==========
        contours, _ = cv2.findContours(
            merged_mask, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # ========== 5. TESPİT ==========
        img_result = img.copy()
        
        for c in contours:
            area = cv2.contourArea(c)
            
            if area < 500 or area > 50000:
                continue
                
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
                
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            (x, y, w, h) = cv2.boundingRect(c)
            
            if w < 400 and h < 600:
                cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(img_result, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(img_result, f"{int(area)}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # ========== 6. GÖRSELLEŞTİRME ==========
        # Method ismini göster
        cv2.putText(img_result, f"Method: {method_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_result, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        #cv2.imshow('1. Original', img)
        #cv2.imshow('2. MOG2 Mask', fg_mask_clean)
        #cv2.imshow('3. Flow Mask', thresh_flow)
        #cv2.imshow('4. MERGED Mask', merged_mask)
        cv2.imshow('5. Final Detection', img_result)
        cv2.imshow('6. Optical Flow HSV', draw_hsv(flow))
                
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} - Method: {method_name}")
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('1'):
            merge_method = "or"
            print("Switched to OR method")
        elif k == ord('2'):
            merge_method = "and"
            print("Switched to AND method")
        elif k == ord('3'):
            merge_method = "weighted"
            print("Switched to WEIGHTED method")

    cam.release()
    cv2.destroyAllWindows()
    print(f"Toplam {frame_count} frame işlendi.")

if __name__ == "__main__":
    main()