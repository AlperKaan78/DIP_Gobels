# Multi-Object Tracking System

**Real-time multi-object tracking using Kalman Filter, Optical Flow, and Background Subtraction**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.5%2B-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## 📖 Overview

This project implements a complete multi-object tracking (MOT) pipeline for urban traffic surveillance. The system combines classical computer vision techniques:

- **MOG2 Background Subtraction** for object detection
- **Dense Optical Flow** for motion estimation and blob refinement
- **Kalman Filter** for state prediction and tracking
- **Hungarian Algorithm** for optimal data association

### Key Features

✅ **AI-Free**: No neural networks, pure classical CV algorithms  
✅ **Real-time**: ~25 FPS on CPU (1080p video)  
✅ **Robust**: Handles occlusions, fragmentation, and shadows  
✅ **Modular**: Each component is independent and testable  
✅ **Configurable**: All parameters in one place  

---

## 📚 Papers & References

This implementation is based on:

1. **Beaupre et al. (2018)**  
   *"Improving Multiple Object Tracking with Optical Flow and Edge Preprocessing"*  
   - Optical flow for blob merging/splitting  
   - Edge processing for better segmentation

2. **Shantaiya et al. (2015)**  
   *"Multiple Object Tracking using Kalman Filter and Optical Flow"*  
   - Kalman filter for occlusion handling  
   - Predict-only mode for lost tracks

---

## 🏗️ System Architecture
```
┌─────────────┐
│  Video Input│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│               DETECTION MODULE                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │   MOG2   │→ │Morphology│→ │ Contours │         │
│  │Background│  │ (Clean)  │  │(Find Blobs)│        │
│  └──────────┘  └──────────┘  └──────────┘         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          OPTICAL FLOW MODULE                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │Farneback │→ │  Motion  │→ │   Blob   │         │
│  │Dense Flow│  │Extraction│  │Refinement│         │
│  └──────────┘  └──────────┘  └──────────┘         │
│                                 │                   │
│                        ┌────────┴────────┐          │
│                        │  MERGE  │ SPLIT │          │
│                        └─────────────────┘          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           DATA ASSOCIATION MODULE                   │
│  ┌──────────────────────────────────────┐          │
│  │    Hungarian Algorithm (IoU-based)   │          │
│  │    + Euclidean Distance Fallback     │          │
│  └──────────────────────────────────────┘          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              TRACKING MODULE                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  Kalman  │→ │  Track   │→ │  Lost    │         │
│  │  Predict │  │  Update  │  │ Handling │         │
│  └──────────┘  └──────────┘  └──────────┘         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
              ┌──────────┐
              │  Output  │
              │  Video   │
              └──────────┘
```

---

## 📁 Project Structure
```
DIP_Gobels/
├── Data/                          # Input videos
│   ├── mobese_3.mp4              # Main test video
│   ├── rouen_video.avi
│   ├── sherbrooke_video.avi
│   └── atrium.avi
│
├── outputs/                       # Generated outputs
│   └── tracking_result.avi       # Tracked video
│
├── test_codes/                    # Learning/testing scripts
│   ├── detection_without_of.py   # MOG2 only test
│   ├── mog2_test.py              # Background subtraction test
│   ├── optical_flow.py           # Optical flow test
│   ├── test_kalman.py            # Kalman filter demo
│   └── ofd1.py, ofd2.py          # OF visualization helpers
│
├── tracking_system/               # Main pipeline (⭐ Core)
│   ├── detection.py              # MOG2 detector
│   ├── optical_flow.py           # Optical flow processor
│   ├── tracker.py                # Kalman-based tracker
│   ├── data_association.py       # Hungarian matching
│   └── main.py                   # Full pipeline
│
└── README.md                      # This file
```

---

## 🔧 Module Details

### 1️⃣ `detection.py` - Object Detection

**Class:** `ObjectDetector`

**Purpose:** Detect moving objects using background subtraction

**Pipeline:**
1. **MOG2**: Adaptive background model
2. **Shadow Removal**: Threshold to eliminate shadows (127→0)
3. **Morphology**: Open (denoise) + Close (fill holes)
4. **Contours**: Find object blobs
5. **Filtering**: Remove too small/large blobs

**Output:** List of blobs with `{bbox, center, area}`

---

### 2️⃣ `optical_flow.py` - Motion Analysis

**Class:** `OpticalFlowProcessor`

**Purpose:** Compute dense optical flow and refine blobs

**Pipeline:**
1. **Farneback Flow**: Calculate motion vectors per pixel
2. **Motion Extraction**: Get average motion per blob
3. **Blob Merging**: Merge fragmented blobs (similar motion + close distance)
4. **Blob Splitting**: Split occluded objects (opposite motion)

**Output:** Refined blobs with motion info `{vx, vy, speed, angle}`

---

### 3️⃣ `data_association.py` - Matching

**Functions:**
- `associate_detections_to_tracks()`: IoU or Euclidean
- `associate_hybrid()`: IoU + Euclidean fallback
- `associate_with_motion()`: Motion-aware matching

**Purpose:** Match detections to existing tracks optimally

**Algorithm:** Hungarian (linear_sum_assignment from scipy)

**Output:** `(matches, unmatched_detections, unmatched_tracks)`

---

### 4️⃣ `tracker.py` - State Tracking

**Classes:**
- `Track`: Individual object track with Kalman filter
- `TrackerManager`: Manages all tracks

**Purpose:** Maintain temporal consistency of tracked objects

**Features:**
- **Kalman Filter**: Predict position when detection is lost
- **Confidence Score**: Track reliability metric
- **Occlusion Handling**: Predict-only mode
- **Lifecycle Management**: Create/Update/Remove tracks

**State:** `[x, y, vx, vy]` (position + velocity)

---

### 5️⃣ `main.py` - Full Pipeline

**Purpose:** Integrate all modules into complete system

**Features:**
- ✅ Config management (all parameters in `Config` class)
- ✅ Statistics collection (FPS, match rate, etc.)
- ✅ Command-line interface
- ✅ Video output
- ✅ Real-time visualization

---

## 🚀 Installation

### Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy
- SciPy (for Hungarian algorithm)
- scikit-image (for optical flow helpers)

### Setup
```bash
# Clone/navigate to project
cd ~/DIP_Gobels

# Create virtual environment (recommended)
python3 -m venv dip_env
source dip_env/bin/activate  # Linux/Mac
# or
dip_env\Scripts\activate     # Windows

# Install dependencies
pip install opencv-python numpy scipy scikit-image
```

---

## 💻 Usage

### Basic Usage
```bash
cd tracking_system
python main.py
```

**This will:**
- Process `../Data/mobese_3.mp4`
- Display 3 windows: Tracking Result, FG Mask, Optical Flow
- Save output to `../outputs/tracking_result.avi`
- Print statistics every 30 frames

### Advanced Usage

#### 1. Process Different Video
```bash
python main.py --video ../Data/rouen_video.avi
```

#### 2. Custom Output Path
```bash
python main.py --output ../outputs/my_tracking.avi
```

#### 3. Headless Mode (No Display)
```bash
python main.py --no-display
```

Useful for:
- Running on server without GUI
- Batch processing multiple videos
- Automation scripts

#### 4. Adjust Tracking Sensitivity
```bash
python main.py --max-misses 40
```

Higher `max-misses` = tracks live longer when detection is lost  
Lower `max-misses` = tracks deleted faster (more new IDs)

#### 5. Combine Options
```bash
python main.py \
  --video ../Data/sherbrooke_video.avi \
  --output ../outputs/sherbrooke_result.avi \
  --max-misses 25 \
  --no-display
```

---

## ⚙️ Configuration

All parameters are in `main.py` → `Config` class:
```python
class Config:
    # Video
    VIDEO_PATH = "../Data/mobese_3.mp4"
    OUTPUT_PATH = "../outputs/tracking_result.avi"
    
    # Detection (MOG2)
    MOG2_HISTORY = 500              # Background model frames
    MOG2_VAR_THRESHOLD = 16         # Sensitivity (lower = more sensitive)
    
    # Optical Flow (Blob Refinement)
    OF_MERGE_DISTANCE = 80          # Max distance to merge blobs (px)
    OF_MERGE_ANGLE = np.pi / 3      # Max angle difference (60°)
    
    # Tracking (Kalman)
    MAX_MISSES = 30                 # Max lost frames before deletion
    
    # Data Association
    IOU_THRESHOLD = 0.25            # Min IoU for match
    DISTANCE_THRESHOLD = 120        # Max center distance (px)
```

### Parameter Tuning Guide

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `MOG2_VAR_THRESHOLD` | More sensitive (detect small movements) | Less sensitive (ignore noise) |
| `OF_MERGE_DISTANCE` | Merge only very close blobs | Merge distant blobs (risk: false merge) |
| `MAX_MISSES` | Delete tracks quickly (more new IDs) | Keep tracks longer (more robust) |
| `IOU_THRESHOLD` | Strict matching | Relaxed matching |

---

## 🎮 Controls

When running with display:

| Key | Action |
|-----|--------|
| `ESC` | Exit program |
| `SPACE` | Pause/Resume video |

---

## 📊 Output

### Console Output
```
============================================================
CONFIGURATION SUMMARY
============================================================
Video: ../Data/mobese_3.mp4
Detection: MOG2 History=500, VarThresh=16
Tracking: Max Misses=30
Data Association: hybrid (IoU=0.25, Dist=120px)
============================================================

Frame   30/628 (4.8%) | Active:  5 | Det:  3 | Match:  3/3 | FPS: 24.3
Frame   60/628 (9.6%) | Active:  4 | Det:  4 | Match:  4/4 | FPS: 25.1
...

============================================================
TRACKING STATISTICS
============================================================
Total Frames Processed: 628
Match Rate: 90.1%
Tracks Created: 28
Average FPS: 24.5
============================================================
```

### Video Output

Saved to `outputs/tracking_result.avi` with:
- 🟢 **Green boxes**: Healthy tracks (high confidence)
- 🟡 **Yellow boxes**: Low confidence tracks
- 🔴 **Red boxes**: Lost tracks (predict-only mode)
- 🟣 **Purple dots**: Object centers
- Track ID and confidence displayed

---

## 🧪 Testing Individual Modules

Each module can be tested independently:

### Test Detection Only
```bash
cd tracking_system
python detection.py
```

Shows:
- Original frame
- Cleaned foreground mask
- Detection result with bounding boxes

### Test Optical Flow + Refinement
```bash
python optical_flow.py
```

Shows:
- FG mask
- Optical flow visualization (HSV)
- Detection + motion vectors

### Test Full Tracking
```bash
python tracker.py
```

Shows:
- Complete tracking with Kalman filter
- ID assignment
- Lost track handling

### Test Data Association
```bash
python data_association.py
```

Runs unit tests for matching algorithms

---

## 📈 Performance

**Hardware:** Intel i5-8250U @ 1.6GHz (4 cores)  
**Video:** 1920x1080 @ 30fps  

| Metric | Value |
|--------|-------|
| **Average FPS** | 24-26 fps |
| **Processing Time** | ~40 ms/frame |
| **Match Rate** | 85-92% |
| **Memory Usage** | ~150 MB |

**Bottlenecks:**
1. Dense optical flow (~15ms)
2. MOG2 background subtraction (~10ms)
3. Morphological operations (~5ms)

---

## 🐛 Troubleshooting

### Video not found
```
❌ Video bulunamadı: ../Data/video.mp4
```

**Solution:** Check video path, use absolute path:
```bash
python main.py --video /full/path/to/video.mp4
```

### Too many tracks created

**Symptom:** `Total Created: 50+` for 300 frames

**Causes:**
- `MAX_MISSES` too low → tracks deleted too fast
- `IOU_THRESHOLD` too high → poor matching

**Solution:**
```bash
python main.py --max-misses 40
```

Or edit `Config.IOU_THRESHOLD = 0.20` (lower = easier match)

### Low match rate (<70%)

**Causes:**
- Poor detection quality
- Fast-moving objects
- Heavy occlusions

**Solution:**
1. Lower `MOG2_VAR_THRESHOLD` (more sensitive detection)
2. Increase `DISTANCE_THRESHOLD` (relaxed matching)
3. Use `associate_with_motion()` (motion-aware matching)

---

## 🔬 Algorithm Details

### Kalman Filter State Transition
```
State: [x, y, vx, vy]

Prediction:
x(t+1) = x(t) + vx(t) * Δt
y(t+1) = y(t) + vy(t) * Δt
vx(t+1) = vx(t)  (constant velocity model)
vy(t+1) = vy(t)

Measurement: [x, y]  (only position observed)
```

### Hungarian Algorithm Cost Matrix
```
        Detection 1   Detection 2   Detection 3
Track 1      0.8          0.2          0.9     
Track 2      0.3          0.7          0.4
Track 3      0.9          0.4          0.1

Optimal assignment: T1→D2, T2→D1, T3→D3
```

---

## 📝 Future Improvements

### Short-term
- [ ] Add Re-ID (track buffer for deleted tracks)
- [ ] Implement edge processing (Paper 1)
- [ ] Add trajectory visualization
- [ ] Support multiple videos batch processing

### Long-term
- [ ] Deep learning detector integration (YOLO)
- [ ] Multi-camera support
- [ ] GPU acceleration (CUDA)
- [ ] Real-time dashboard (web UI)

---

## 🤝 Contributing

This is an academic project. For improvements:

1. Test on your video dataset
2. Tune parameters in `Config`
3. Share results and optimal settings
4. Report bugs with video samples

---

## 📄 License

MIT License - Feel free to use for academic purposes

---

## 👥 Authors

**Yunus** - Computer Vision Project (DIP_Gobels)

---

## 🙏 Acknowledgments

- Papers by Beaupre et al. and Shantaiya et al.
- OpenCV community
- scikit-image contributors

---

## 📞 Contact

For questions or issues, check:
- Code comments in each module
- Paper references in `PAPER__*.pdf`

---

**Last Updated:** December 2024