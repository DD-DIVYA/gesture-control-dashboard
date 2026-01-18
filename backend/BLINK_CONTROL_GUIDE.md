# 👁️ Blink Control System - User Guide

## 🎯 Complete Control System Overview

Your wheelchair control dashboard uses a sophisticated **blink + head movement** control system for hands-free operation. Here's everything you need to know:

---

## 🚀 Getting Started

### **System Activation**
1. **Start the system**: Follow the running instructions in README_Enhanced.md
2. **Position yourself**: Sit comfortably in front of the camera
3. **Face detection**: Wait for your face to be detected (green indicator)
4. **Ready to control**: The system starts in STOP mode - use blinks to activate!

---

## 👁️ Blink Control Commands

### **🔵 Single Blink (Quick blink)**
- **From STOP mode** → Activates **Wheelchair Controls**
- **In Place mode** → Navigate through available places
- **Timing**: Quick, natural blink (~0.1-0.5 seconds)

### **🟣 Double Blink (Quick-Quick)**
- **From STOP mode** → Activates **Place Selection**
- **In Place mode** → Select the highlighted place
- **Timing**: Two blinks within 1.2 seconds (more tolerant timing)
- **Tip**: The system prioritizes double blinks over single blinks to prevent accidental wheelchair activation

### **🔴 Long Blink (Hold & Release)**
- **From ANY mode** → Return to STOP mode (Safety Reset)
- **Timing**: Hold eyes closed for 0.7+ seconds, then release
- **Emergency**: Works from any mode for immediate stop

---

## 🎮 Control Modes Explained

### **⚪ STOP Mode (Default)**
- **Status**: System ready, all controls inactive
- **Visual**: Gray/inactive interface elements
- **Available Actions**:
  - Single blink → Enter Wheelchair Mode
  - Double blink → Enter Place Selection Mode

### **🔵 Wheelchair Mode**
- **Status**: Head movement controls active
- **Visual**: Blue highlighting, active control buttons
- **Controls**:
  - **Move head forward** → Wheelchair moves forward
  - **Move head backward** → Wheelchair moves backward  
  - **Move head left** → Wheelchair turns left
  - **Move head right** → Wheelchair turns right
  - **Center head** → Wheelchair stops
  - **Long blink** → Return to STOP mode

### **🟣 Place Selection Mode**
- **Status**: Eye-controlled place navigation
- **Visual**: Purple highlighting, place list active
- **Controls**:
  - **Single blink** → Navigate through places (Kitchen → Bedroom → Living Room → Restroom → Kitchen...)
  - **Double blink** → Select the currently highlighted place
  - **Long blink** → Return to STOP mode

---

## 📊 Visual Feedback System

### **Dashboard Indicators**
| Indicator | Meaning |
|-----------|---------|
| 🟢 Green pulse | Blink detection active |
| 🔵 Blue highlight | Wheelchair controls active |
| 🟣 Purple highlight | Place selection active |
| 🟡 Yellow highlight | Place highlighted for selection |
| ✅ Green checkmark | Place selected successfully |

### **Camera Feed Overlay (if enabled)**
- **Eye Status**: Shows "OPEN" or "CLOSED" with colors
- **Blink Timing**: Real-time closed duration counter
- **Recent Blinks**: Count of blinks in last 2 seconds
- **Gaze Point**: Yellow dot showing where you're looking
- **Mode Display**: Current system mode

---

## 🛡️ Safety Features

### **Automatic Safety Timeouts**
- **No activity**: Auto-return to STOP after 10 seconds of inactivity
- **Face lost**: Auto-return to STOP after 3 seconds without face detection
- **Emergency reset**: Long blink always works from any mode

### **Debouncing Protection**
- **Prevents accidental triggers**: Minimum 0.08s between blink processing
- **Double blink detection**: Smart timing to distinguish single vs double blinks
- **Head movement smoothing**: Reduces false movements from minor head shifts

---

## 💡 Pro Tips for Better Control

### **Optimal Blink Technique**
1. **Single blinks**: Natural, relaxed blink
2. **Double blinks**: Quick succession, like "blink-blink"
3. **Long blinks**: Deliberately hold closed, then release (think "meditation pause")

### **Head Movement Tips**
1. **Calibrate first**: Use the calibration button for better accuracy
2. **Gentle movements**: Small head movements are more precise
3. **Return to center**: Centering your head stops the wheelchair
4. **Practice**: Try movements gradually to understand sensitivity

### **Lighting Conditions**
- **Good lighting**: Bright, even lighting improves detection
- **Avoid backlighting**: Don't sit with bright light behind you
- **Camera position**: Keep camera at eye level for best face detection

---

## 🔧 Troubleshooting

### **Blinks Not Detected**
- Check lighting conditions
- Ensure face is clearly visible to camera
- Try recalibrating eyes (Eye Calibration button)
- Make blinks more deliberate

### **Double Blink Issues**
- **Problem**: Single blink activates wheelchair when trying to double blink
- **Solution**: The system now waits longer (1.5s) to detect double blinks
- **Tip**: Try making your double blinks closer together (within 1.2 seconds)
- **Practice**: Try "blink-pause-blink" rhythm, not too fast, not too slow

### **False Blink Detection**
- Reduce rapid head movements
- Ensure stable camera mounting
- Check for reflections or shadows on face

### **Head Movement Issues**
- Use Head Calibration to reset reference position
- Ensure face is centered in camera view
- Check for stable sitting position

### **Connection Issues**
- Ensure backend is running (movements.py)
- Check WebSocket connection (ws://localhost:5000)
- Restart the system if needed

---

## 📱 Real-Time Metrics

The dashboard shows live system information:

- **🔋 Battery**: Simulated battery percentage (decreases with movement)
- **⚡ Motor Speed**: Current wheelchair speed (0-100%)
- **📈 Movement Intensity**: Head movement strength indicator
- **📏 Distance**: Total distance traveled this session
- **⏱️ Session Time**: Time since system started
- **👤 Face Tracking**: Shows if face is currently detected

---

## 🎯 Complete Workflow Example

1. **Start** → System in STOP mode
2. **Single blink** → Enter Wheelchair mode
3. **Move head forward** → Wheelchair moves forward
4. **Center head** → Wheelchair stops
5. **Long blink** → Return to STOP mode
6. **Double blink** → Enter Place selection
7. **Single blink** → Navigate to "Kitchen"
8. **Single blink** → Navigate to "Bedroom"  
9. **Double blink** → Select "Bedroom"
10. **Long blink** → Return to STOP mode

---

## 🔮 Advanced Features (YOLOv8 Enhanced)

If YOLOv8 is available, you get:
- **Better face detection** in challenging lighting
- **More robust tracking** with head movements
- **Enhanced accuracy** for blink detection
- **Improved stability** in various conditions

---

## 📞 Need Help?

- **Check the terminal**: Backend shows detailed logging
- **Camera feed**: Enable SHOW_WINDOW for visual debugging
- **Dashboard notifications**: Watch for system messages
- **Recalibrate**: Use head/eye calibration buttons when needed

**Remember**: Long blink is your emergency reset - it always works! 👁️