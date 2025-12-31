import numpy as np
import cv2
import time

# | Fonksiyon | Ne anlatıyor           |
# | --------- | ---------------------- |
# | draw_flow | Flow = yönlü vektör    |
# | draw_hsv  | Flow = yön + hız       |
# | warp_flow | Flow = piksel eşlemesi |

count = 0

def draw_flow(img, flow, step=16):
    # flow → calcOpticalFlowFarneback çıktısı
    # step → kaç pikselde bir ok çizileceği 
    h, w = img.shape[:2]    # Görüntü yüksekliği ve genişliği
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    # (x, y) → okun başladığı noktalar

    fx, fy = flow[y,x].T
    # flow[y, x] = (dx, dy)
    # fx → yatay hareket
    # fy → dikey hareket

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis   = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:      
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis


def draw_hsv(flow):
    h, w   = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi    # direction
    v   = np.sqrt(fx*fx + fy*fy)        # Magnitude (velocity)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(90/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    # Hue (H)        → yön
    # Value (V)      → hız
    # Saturation (S) → sabit 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow    # Akış ters çevriliyor (backward warping)
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

