# Enhanced Eye Tracking System - YOLOv8 + MediaPipe

This upgraded wheelchair control system combines **YOLOv8** for robust face detection with **MediaPipe** for detailed eye tracking, providing enhanced accuracy and new interaction capabilities.

## 🆕 New Features

### **Enhanced Face Detection**
- **YOLOv8 Integration**: More robust face detection in challenging conditions
- **Fallback System**: Automatically falls back to MediaPipe if YOLOv8 unavailable
- **Better Performance**: Improved accuracy in low light and complex backgrounds

### **Advanced Eye Tracking**
- **Gaze Direction Estimation**: Track where users are looking
- **Saccade Detection**: Detect rapid eye movements
- **Smoothed Gaze Points**: Reduce jitter with temporal smoothing
- **Enhanced Blink Detection**: More accurate eye aspect ratio calculations

### **Visual Debugging**
- **Real-time Gaze Visualization**: Blue dot shows estimated gaze point
- **YOLO Detection Boxes**: Green boxes show face detection confidence
- **Enhanced Overlays**: More detailed tracking information
- **Method Indicator**: Shows which tracking method is active

## 🛠️ Installation

### **Option 1: Automatic Setup (Recommended)**
```bash
cd Backend(Ml_python)
python setup_enhanced.py
```

### **Option 2: Manual Installation**
```bash
# Install core dependencies
pip install opencv-python mediapipe websockets numpy scipy

# Install YOLOv8 (optional but recommended)
pip install ultralytics

# Or install all at once
pip install -r requirements.txt
```

## 🚀 Usage

### **Running the Enhanced System**
```bash
# Start the WebSocket server (Terminal 1)
python ws_server.py

# Start the enhanced eye tracking (Terminal 2)  
python movements.py

# Start the frontend dashboard (Terminal 3)
cd ../Frontend
npm run dev
```

### **Controls & Interactions**

#### **Blink Gestures (Same as before)**
- **Single Blink**: STOP → WHEELCHAIR mode, or navigate places
- **Double Blink**: STOP → PLACE mode, or select highlighted place  
- **Long Blink** (>0.7s): Return to STOP mode from any mode

#### **Head Movement Controls (Enhanced)**
- **FORWARD/BACKWARD/LEFT/RIGHT**: Control wheelchair direction
- **Enhanced Tracking**: More accurate with YOLOv8 face detection
- **Better Stability**: Reduced false movements

#### **Keyboard Controls (Debug)**
- **Q**: Quit application
- **C**: Calibrate head position
- **E**: Reset eye baseline
- **M**: Manual mode switching (for testing)

## 🎯 Technical Improvements

### **Face Detection Pipeline**
```
Camera Input → YOLOv8 Face Detection → MediaPipe Landmarks → Eye Analysis
                      ↓ (if unavailable)
              MediaPipe Face Detection → MediaPipe Landmarks → Eye Analysis
```

### **Enhanced Eye Tracking Features**

#### **Gaze Estimation**
- Calculates average eye center positions
- Estimates where user is looking on screen
- Smooths gaze points over time to reduce noise

#### **Saccade Detection**
- Detects rapid eye movements between fixation points
- Useful for advanced UI interactions
- Logged for debugging and future features

#### **Improved Blink Detection**
- Uses detailed eye landmarks (16 points per eye vs 4)
- Calculates Eye Aspect Ratio with better precision
- More robust to head rotation and lighting changes

## 📊 System Architecture

### **YOLOv8EyeTracker Class**
- `detect_faces_yolo()`: Face detection using YOLOv8
- `calculate_enhanced_eye_ratio()`: Improved EAR calculation
- `estimate_gaze_direction()`: Gaze point estimation
- `detect_eye_interactions()`: Comprehensive eye analysis

### **Enhanced State Management**
- Gaze history tracking
- Interaction zone mapping (future feature)
- Calibration data storage
- Performance metrics

## 🔧 Configuration

### **System Settings** (movements.py)
```python
# YOLOv8 Settings
YOLO_AVAILABLE = True/False  # Auto-detected
YOLO_CONFIDENCE = 0.5        # Face detection threshold

# Enhanced Eye Tracking
SACCADE_THRESHOLD = 0.02     # Minimum movement for saccade
GAZE_SMOOTHING_WINDOW = 5    # Frames for smoothing
EYE_INTERACTION_THRESHOLD = 0.15  # UI interaction sensitivity
```

### **Debug Visualization**
```python
SHOW_YOLO_BOXES = True       # Show face detection boxes
SHOW_GAZE_POINTS = True      # Show gaze estimation
SHOW_EYE_LANDMARKS = False   # Show detailed landmarks
```

## 🐛 Troubleshooting

### **YOLOv8 Issues**
```bash
# If YOLOv8 fails to install
pip install --upgrade pip
pip install ultralytics --no-cache-dir

# If CUDA issues occur
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### **Camera Issues**
- Ensure camera is not used by other applications
- Try different camera indices (0, 1, 2...)
- Check camera permissions on Windows/macOS

### **Performance Issues**
- YOLOv8 is more CPU intensive than MediaPipe alone
- System automatically falls back to MediaPipe-only if needed
- Reduce resolution for better performance

## 📈 Performance Comparison

| Feature | MediaPipe Only | YOLOv8 + MediaPipe |
|---------|---------------|-------------------|
| **Face Detection** | Good | Excellent |
| **Low Light Performance** | Fair | Good |
| **Complex Backgrounds** | Fair | Excellent |
| **CPU Usage** | Low | Medium |
| **Accuracy** | Good | Excellent |
| **Robustness** | Good | Excellent |

## 🔮 Future Enhancements

### **Planned Features**
- **Direct Gaze Control**: Click UI elements by looking at them
- **Eye Gesture Recognition**: Wink detection, eye patterns
- **Adaptive Calibration**: Automatic recalibration during use
- **Multi-user Support**: Different calibration profiles
- **Advanced Analytics**: Gaze patterns, usage statistics

### **Possible Integrations**
- **Voice Commands**: Combine with speech recognition
- **Gesture Recognition**: Hand gestures for additional control
- **IoT Integration**: Control smart home devices
- **Accessibility Features**: Text-to-speech, magnification

## 📄 API Changes

### **New WebSocket Events**
- `GAZE_POINT`: Real-time gaze coordinates
- `SACCADE_DETECTED`: Rapid eye movement events
- `EYE_INTERACTION`: UI interaction events
- `TRACKING_METHOD`: Current detection method

### **Enhanced Existing Events**
- `HEAD_MOVE`: Now includes gaze information
- `SYSTEM_STATUS`: Added eye tracking metrics

## 🤝 Contributing

The enhanced system maintains backward compatibility while adding new capabilities. All existing functionality continues to work exactly as before, with optional enhancements available when YOLOv8 is installed.

## 📞 Support

For issues specific to the enhanced features:
1. Check that YOLOv8 installed correctly: `python -c "from ultralytics import YOLO; print('YOLOv8 OK')"`
2. Verify camera access: `python -c "import cv2; cap=cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"`
3. Check the debug overlay for tracking method confirmation
4. Review console logs for detailed error information