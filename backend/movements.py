import cv2
import mediapipe as mp
import time
import asyncio
import websockets
import json
import logging
import random
import math
import numpy as np
from threading import Thread, Event, Lock
from scipy.spatial.distance import euclidean

# Try to import YOLOv8, fallback to MediaPipe if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLOv8 not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

# ================== CONFIG ==================
WS_SERVER_URL = "ws://localhost:5000"

# GUI preview
SHOW_WINDOW = True
WINDOW_NAME = "Gesture Controls (q quit, c head-cal, e eye-cal, m toggle-mode)"

# Logging
LOG_TO_CONSOLE = True

ROOMS = ["Kitchen", "Bedroom", "Living Room", "Restroom"]

# --- Blink timing (tune here) ---
BLINK_MIN = 0.06          # short blink lower bound (s) - more sensitive
BLINK_MAX = 0.50          # short blink upper bound (s) - more tolerant
LONG_BLINK_MIN = 0.60     # long blink lower bound (s) - more lenient

# Simple blink classification windows
SINGLE_BLINK_WINDOW = 0.6   # time to wait before confirming single blink
DOUBLE_BLINK_GAP = 1.0      # max gap between blinks for double detection (more lenient)
BLINK_DEBOUNCE_TIME = 0.1   # min time between processing blink events

# Safety and timeout settings
SAFETY_TIMEOUT = 10.0       # auto-reset to STOP after inactivity (s)
FACE_LOST_TIMEOUT = 3.0     # timeout when face tracking is lost

# --- Dynamic eye thresholding ---
USE_DYNAMIC_EYE_THR = True
EYE_CLOSE_FACTOR = 0.70    # threshold = open_ema * factor - balanced sensitivity
EYE_EMA_BETA = 0.90        # EMA for "open" ratio
EYE_MIN_THRESHOLD = 0.03   # never go below this - more sensitive

# Head tilt thresholds (normalized coords)
HEAD_THR_ENTER = 0.09
HEAD_THR_EXIT  = 0.06

# Head sampling debounce
HEAD_EVENT_MIN_INTERVAL = 0.08
TRACK_LOST_TIMEOUT = 1.0

# Modes
MODE_STOP = "STOP"
MODE_WHEELCHAIR = "WHEELCHAIR"
MODE_PLACE = "PLACE"

# ================= Enhanced Eye Tracking =================
mp_face_mesh = mp.solutions.face_mesh

# Enhanced eye landmark indices for better tracking
LEFT_EYE = (33, 133, 159, 145)  # (outer, inner, upper, lower)
RIGHT_EYE = (362, 263, 386, 374)
NOSE_TIP = 1

# Detailed eye landmarks for advanced tracking
LEFT_EYE_DETAILED = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_DETAILED = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Eye tracking thresholds
SACCADE_THRESHOLD = 0.02  # Minimum movement to detect saccade
FIXATION_DURATION = 0.3   # Time to confirm fixation
GAZE_SMOOTHING_WINDOW = 5 # Frames for gaze smoothing
EYE_INTERACTION_THRESHOLD = 0.15  # Eye movement threshold for UI interaction

# ================= Logging Setup =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("FaceWS")

# ================= Helpers =================
# ================= Blink Event Handling =================
def classify_blink_sequence(timestamps, current_time):
    """
    Simple blink classification for reliable detection.
    Returns: ("single", "double", None)
    """
    if not timestamps:
        return None

    # Check for double blink first (2 blinks within gap)
    if len(timestamps) >= 2:
        time_between = timestamps[-1] - timestamps[-2]
        log.debug(f"🔍 Checking double blink: {time_between:.3f}s gap (max: {DOUBLE_BLINK_GAP}s)")
        if time_between <= DOUBLE_BLINK_GAP:
            log.info(f"👁️👁️ DOUBLE BLINK DETECTED: {time_between:.3f}s gap")
            return "double"
    
    # Check for single blink (oldest blink that's waited long enough)
    if len(timestamps) >= 1:
        time_since_oldest = current_time - timestamps[0]
        if time_since_oldest >= SINGLE_BLINK_WINDOW:
            log.info(f"👁️ SINGLE BLINK CONFIRMED: {time_since_oldest:.3f}s wait")
            return "single"
        else:
            log.debug(f"⏳ Waiting for single blink: {time_since_oldest:.3f}s < {SINGLE_BLINK_WINDOW}s")
    
    return None

def handle_blink_event(blink_type, ws_client):
    """
    Handle blink events and manage state transitions.
    """
    global STATE
    
    now = time.time()
    STATE.last_activity_time = now
    
    with STATE.lock:
        current_mode = STATE.current_mode
        
        if blink_type == "long":
            # Long blink always returns to STOP from any mode
            if current_mode != MODE_STOP:
                log.info(f"🔄 Long blink detected: {current_mode} → STOP")
                STATE.current_mode = MODE_STOP
                STATE.last_head_dir = "STOP"
                broadcast_mode_change(ws_client)
                broadcast_system_reset(ws_client)
                # Send notification
                ws_client.emit("BLINK_EVENT", {
                    "type": "long",
                    "action": "return_to_stop",
                    "from_mode": current_mode,
                    "message": "Long blink: Returned to STOP mode"
                })
            else:
                log.debug("Long blink detected, already in STOP mode")
        
        elif blink_type == "single":
            if current_mode == MODE_STOP:
                # Single blink from STOP → enter WHEELCHAIR_MODE
                log.info("👁️ Single blink: STOP → WHEELCHAIR mode activated")
                STATE.current_mode = MODE_WHEELCHAIR
                STATE.last_head_dir = "STOP"
                broadcast_mode_change(ws_client)
                ws_client.emit("BLINK_EVENT", {
                    "type": "single",
                    "action": "activate_wheelchair",
                    "message": "Single blink: Wheelchair controls activated"
                })
            
            elif current_mode == MODE_PLACE:
                # Single blink in PLACE_MODE → move forward in list
                old_place = STATE.highlighted_place
                STATE.current_index = (STATE.current_index + 1) % len(ROOMS)
                STATE.highlighted_place = ROOMS[STATE.current_index]
                log.info(f"👁️ Single blink: Place navigation {old_place} → {STATE.highlighted_place}")
                broadcast_place_highlight(ws_client, STATE.highlighted_place)
                ws_client.emit("BLINK_EVENT", {
                    "type": "single",
                    "action": "navigate_place",
                    "from_place": old_place,
                    "to_place": STATE.highlighted_place,
                    "message": f"Single blink: Navigated to {STATE.highlighted_place}"
                })
            
            else:
                # Ignore single blinks in WHEELCHAIR_MODE
                log.debug(f"Single blink ignored in {current_mode} mode")
        
        elif blink_type == "double":
            if current_mode == MODE_STOP:
                # Double blink from STOP → enter PLACE_MODE
                log.info("👁️👁️ Double blink: STOP → PLACE mode activated")
                STATE.current_mode = MODE_PLACE
                STATE.highlighted_place = ROOMS[STATE.current_index]
                broadcast_mode_change(ws_client)
                broadcast_place_highlight(ws_client, STATE.highlighted_place)
                ws_client.emit("BLINK_EVENT", {
                    "type": "double",
                    "action": "activate_places",
                    "highlighted_place": STATE.highlighted_place,
                    "message": "Double blink: Place selection activated"
                })
            
            elif current_mode == MODE_PLACE:
                # Double blink in PLACE_MODE → select highlighted place
                selected = STATE.highlighted_place
                STATE.selected_place = selected
                log.info(f"✅ PLACE SELECTED: Double blink confirmed '{selected}' (was highlighted: {STATE.highlighted_place})")
                broadcast_place_select(ws_client, STATE.selected_place)
                ws_client.emit("BLINK_EVENT", {
                    "type": "double",
                    "action": "select_place",
                    "selected_place": selected,
                    "message": f"✅ Place selected: {selected}"
                })
            
            else:
                # Ignore double blinks in WHEELCHAIR_MODE
                log.debug(f"Double blink ignored in {current_mode} mode")

def broadcast_mode_change(ws_client):
    """Send MODE_CHANGE event."""
    ws_client.emit("MODE_CHANGE", {"mode": STATE.current_mode})

def broadcast_system_reset(ws_client):
    """Send SYSTEM_RESET event."""
    ws_client.emit("SYSTEM_RESET", {})

def broadcast_place_highlight(ws_client, place):
    """Send PLACE_HIGHLIGHT event."""
    ws_client.emit("PLACE_HIGHLIGHT", {"place": place})

def broadcast_place_select(ws_client, place):
    """Send PLACE_SELECT event."""
    ws_client.emit("PLACE_SELECT", {"place": place})

def broadcast_head_move(ws_client, direction):
    """Send HEAD_MOVE event with enhanced movement data."""
    # Calculate motor speed based on direction and movement intensity
    speed = 0.0
    if direction.upper() != "STOP":
        # Simulate motor speed based on movement intensity and direction
        base_speed = 30 + (STATE.movement_intensity * 50)  # 30-80% speed range
        speed = min(100.0, max(0.0, base_speed))
        STATE.motor_speed = speed
        
        # Simulate distance calculation (approximate)
        if STATE.last_head_emit > 0:
            time_delta = time.time() - STATE.last_head_emit
            distance_delta = (speed / 100.0) * 2.0 * time_delta  # 2 units per second at full speed
            STATE.total_distance += distance_delta
    else:
        STATE.motor_speed = 0.0
    
    # Update battery simulation (decreases during movement)
    update_battery_simulation(direction.upper() != "STOP")
    
    ws_client.emit("HEAD_MOVE", {
        "direction": direction.upper(),
        "motor_speed": STATE.motor_speed,
        "movement_intensity": STATE.movement_intensity,
        "battery_percentage": STATE.battery_percentage,
        "total_distance": round(STATE.total_distance, 2),
        "session_time": int(time.time() - STATE.session_start)
    })

def check_safety_timeout(ws_client):
    """Check for safety timeout and auto-reset to STOP."""
    global STATE
    
    now = time.time()
    time_since_activity = now - STATE.last_activity_time
    time_since_face = now - STATE.last_seen_face
    
    # Auto-reset conditions
    if (time_since_activity > SAFETY_TIMEOUT or 
        time_since_face > FACE_LOST_TIMEOUT):
        
        with STATE.lock:
            if STATE.current_mode != MODE_STOP:
                STATE.current_mode = MODE_STOP
                STATE.last_head_dir = "STOP"
                broadcast_mode_change(ws_client)
                broadcast_system_reset(ws_client)
                log.info(f"Safety timeout: auto-reset to STOP (activity: {time_since_activity:.1f}s, face: {time_since_face:.1f}s)")

def should_ignore_head_movement():
    """Check if head movements should be ignored based on current mode."""
    return STATE.current_mode != MODE_WHEELCHAIR

def update_battery_simulation(is_moving=False):
    """Simulate battery drain based on usage."""
    now = time.time()
    time_delta = now - STATE.last_metrics_update
    
    if time_delta > 1.0:  # Update every second
        # Battery drain: faster when moving, slower when idle
        drain_rate = 0.02 if is_moving else 0.005  # % per second
        STATE.battery_percentage = max(0.0, STATE.battery_percentage - drain_rate)
        
        # Add some random fluctuation for realism
        if STATE.battery_percentage > 20:
            STATE.battery_percentage += random.uniform(-0.1, 0.05)
            STATE.battery_percentage = max(0.0, min(100.0, STATE.battery_percentage))
        
        STATE.last_metrics_update = now

def calculate_movement_intensity(dx, dy):
    """Calculate movement intensity from head position deltas."""
    distance = math.sqrt(dx * dx + dy * dy)
    # Normalize to 0-1 range (adjust multiplier as needed)
    intensity = min(1.0, distance * 10.0)
    return intensity

def broadcast_system_status(ws_client):
    """Send periodic system status updates."""
    ws_client.emit("SYSTEM_STATUS", {
        "battery_percentage": STATE.battery_percentage,
        "motor_speed": STATE.motor_speed,
        "current_mode": STATE.current_mode,
        "total_distance": round(STATE.total_distance, 2),
        "session_time": int(time.time() - STATE.session_start),
        "face_tracking": time.time() - STATE.last_seen_face < 1.0
    })

# ================= YOLOv8 Enhanced Eye Tracker =================
class YOLOv8EyeTracker:
    def __init__(self):
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # Try to load YOLOv8 face detection model
                self.yolo_model = YOLO('yolov8n.pt')  # Using general model, can be replaced with face-specific
                log.info("YOLOv8 model loaded successfully")
            except Exception as e:
                log.warning(f"Failed to load YOLOv8 model: {e}. Falling back to MediaPipe only.")
                self.yolo_model = None
        
        # Enhanced MediaPipe face mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    
    def detect_faces_yolo(self, frame):
        """Use YOLOv8 to detect faces (fallback to MediaPipe if not available)"""
        if not self.yolo_model:
            return None
        
        try:
            results = self.yolo_model(frame, verbose=False, classes=[0])  # Class 0 is person
            faces = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence > 0.5:  # Person detection confidence
                            # Estimate face region (upper portion of person detection)
                            face_h = (y2 - y1) * 0.3  # Face is roughly 30% of person height
                            face_y2 = y1 + face_h
                            faces.append({
                                'bbox': (int(x1), int(y1), int(x2), int(face_y2)),
                                'confidence': confidence
                            })
            
            return faces if faces else None
        except Exception as e:
            log.warning(f"YOLOv8 detection failed: {e}")
            return None
    
    def calculate_enhanced_eye_ratio(self, landmarks, eye_indices, w, h):
        """Enhanced eye aspect ratio calculation"""
        try:
            eye_points = []
            for idx in eye_indices[:6]:  # Use first 6 points for EAR calculation
                x = landmarks[idx].x * w
                y = landmarks[idx].y * h
                eye_points.append((x, y))
            
            # Vertical distances
            A = euclidean(eye_points[1], eye_points[5])
            B = euclidean(eye_points[2], eye_points[4])
            
            # Horizontal distance
            C = euclidean(eye_points[0], eye_points[3])
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C) if C > 0 else 0
            return ear
        except:
            return 0.0
    
    def estimate_gaze_direction(self, landmarks, w, h):
        """Estimate gaze direction using eye landmarks"""
        try:
            # Get eye centers
            left_eye_center = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE_DETAILED[:8]], axis=0)
            right_eye_center = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE_DETAILED[:8]], axis=0)
            
            # Calculate average gaze point
            gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
            gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
            
            return (gaze_x, gaze_y)
        except:
            return None
    
    def detect_eye_interactions(self, landmarks, w, h):
        """Detect eye-based interactions"""
        interactions = {
            'blink': False,
            'saccade': False,
            'gaze_point': None,
            'interaction_zone': None,
            'fixation': False
        }
        
        try:
            # Enhanced blink detection
            left_ear = self.calculate_enhanced_eye_ratio(landmarks, LEFT_EYE_DETAILED, w, h)
            right_ear = self.calculate_enhanced_eye_ratio(landmarks, RIGHT_EYE_DETAILED, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            interactions['blink'] = avg_ear < 0.25  # Adjusted threshold
            
            # Gaze direction estimation
            gaze_point = self.estimate_gaze_direction(landmarks, w, h)
            interactions['gaze_point'] = gaze_point
            
            # Saccade detection (rapid eye movements)
            if gaze_point and STATE.last_gaze_point:
                movement_distance = euclidean(gaze_point, STATE.last_gaze_point)
                interactions['saccade'] = movement_distance > SACCADE_THRESHOLD * w  # Scale by frame width
            
            # Update gaze history for smoothing
            if gaze_point:
                STATE.gaze_history.append(gaze_point)
                if len(STATE.gaze_history) > GAZE_SMOOTHING_WINDOW:
                    STATE.gaze_history.pop(0)
                
                # Smooth gaze point
                if len(STATE.gaze_history) >= 3:
                    smooth_gaze = np.mean(STATE.gaze_history, axis=0)
                    interactions['gaze_point'] = tuple(smooth_gaze)
                
                STATE.last_gaze_point = gaze_point
            
            return interactions
        except Exception as e:
            log.warning(f"Eye interaction detection failed: {e}")
            return interactions

# ================= Shared State =================
class SharedState:
    def __init__(self):
        self.current_index = 0
        self.ref_x = None
        self.ref_y = None
        self.last_head_dir = "STOP"
        self.last_head_emit = 0.0
        self.last_seen_face = time.time()
        self.last_activity_time = time.time()
        self.calibrate_requested = Event()      # head
        self.eye_calibrate_requested = Event()  # eyes (baseline reset)
        self.reset_places_requested = Event()
        self.current_mode = MODE_STOP           # start in STOP mode

        # Blink detection state
        self.blink_timestamps = []              # rolling window of blink times
        self.last_blink_process = 0.0           # debounce blink processing
        self.pending_blinks = []                # queue for blink classification

        # Place selection state
        self.highlighted_place = ROOMS[0] if ROOMS else None
        self.selected_place = None

        self.lock = Lock()
        
        # WebSocket client connection
        self.ws_client = None
        self.connected = False
        
        # Enhanced metrics for dashboard
        self.battery_percentage = 85.0          # Simulated battery level
        self.motor_speed = 0.0                  # Current motor speed (0-100%)
        self.movement_intensity = 0.0           # Head movement intensity
        self.last_metrics_update = time.time()  # For metric simulation timing
        self.total_distance = 0.0               # Total distance traveled
        self.session_start = time.time()        # Session start time
        
        # Enhanced eye tracking state
        self.gaze_history = []                  # History for gaze smoothing
        self.last_gaze_point = None             # Last detected gaze point
        self.interaction_zones = {}             # UI interaction zones
        self.calibration_points = {}            # Gaze calibration data
        self.is_gaze_calibrated = False         # Calibration status
        self.eye_interaction_enabled = True     # Eye interaction toggle
        self.current_gaze_target = None         # Current UI element being gazed at

STATE = SharedState()
EYE_TRACKER = YOLOv8EyeTracker()

# ================= WebSocket Client =================
class WSClient:
    def __init__(self, url):
        self.url = url
        self.websocket = None
        self.loop = None
        self.connected = False
        
    async def connect(self):
        """Connect to WebSocket server and start message listener"""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            STATE.connected = True
            
            log.info(f"Connected to WebSocket server at {self.url}")
            
        except Exception as e:
            log.error(f"Failed to connect to WebSocket server: {e}")
            self.connected = False
            STATE.connected = False
            raise

    async def start_listening(self):
        """Start listening for messages (separate from connect)"""
        if self.websocket and self.connected:
            await self.listen_for_messages()
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            STATE.connected = False
            log.info("Disconnected from WebSocket server")
    
    async def send_message(self, message):
        """Send message to WebSocket server"""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(json.dumps(message))
                if LOG_TO_CONSOLE:
                    log.info(f"→ Sent: {message}")
            except Exception as e:
                log.error(f"Error sending message: {e}")
                self.connected = False
                STATE.connected = False
        else:
            log.warning(f"Not connected, cannot send: {message}")
    
    def emit(self, event, payload=None):
        """Thread-safe emit (called from camera thread)"""
        if self.loop and self.connected:
            message = {"event": event}
            if payload is not None:
                message["payload"] = payload
            asyncio.run_coroutine_threadsafe(self.send_message(message), self.loop)
    
    async def listen_for_messages(self):
        """Listen for messages from WebSocket server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_server_message(data)
                except json.JSONDecodeError:
                    log.warning("Received invalid JSON from server")
                except Exception as e:
                    log.error(f"Error handling server message: {e}")
        except websockets.ConnectionClosed:
            log.info("WebSocket connection closed")
            self.connected = False
            STATE.connected = False
        except Exception as e:
            log.error(f"Error in message listener: {e}")
            self.connected = False
            STATE.connected = False
    
    async def handle_server_message(self, data):
        """Handle messages received from server"""
        event = data.get("event")
        
        if event == "CALIBRATE":
            STATE.calibrate_requested.set()
        elif event == "CALIBRATE_EYES":
            STATE.eye_calibrate_requested.set()
        elif event == "RESET_PLACES":
            STATE.reset_places_requested.set()
        elif event == "GET_STATUS":
            # Send current status back to frontend
            with STATE.lock:
                self.emit("MODE_CHANGE", {"mode": STATE.current_mode})
                if STATE.current_mode == MODE_PLACE and STATE.highlighted_place:
                    self.emit("PLACE_HIGHLIGHT", {"place": STATE.highlighted_place})
                if STATE.selected_place:
                    self.emit("PLACE_SELECT", {"place": STATE.selected_place})
                # Send comprehensive system status
                broadcast_system_status(self)
        else:
            log.debug(f"Received event from frontend: {event}")

# ================= Vision Utilities =================
def eye_ratio(landmarks, idxs, w, h):
    lx, rx, uy, ly = idxs
    left = landmarks[lx].x * w
    right = landmarks[rx].x * w
    top = landmarks[uy].y * h
    bottom = landmarks[ly].y * h
    vertical = abs(top - bottom)
    horizontal = abs(right - left)
    return (vertical / horizontal) if horizontal != 0 else 0.0

def detect_blink_events_dynamic(landmarks, w, h, ctx):
    """
    Dynamic-threshold blink detector.
    Returns: ("blink"|"long"|None, ratio, thr, is_closed)
    ctx keys: is_closed, closed_t0, open_ema (float|None), thr (float|None)
    """
    l = eye_ratio(landmarks, LEFT_EYE, w, h)
    r = eye_ratio(landmarks, RIGHT_EYE, w, h)
    ratio = (l + r) / 2.0

    # Update open baseline when eye is open
    if ctx.get("open_ema") is None:
        ctx["open_ema"] = ratio
    else:
        # Only update when likely open (helps stability)
        if not ctx.get("is_closed", False):
            ctx["open_ema"] = EYE_EMA_BETA * ctx["open_ema"] + (1 - EYE_EMA_BETA) * ratio

    thr = max(EYE_MIN_THRESHOLD, ctx["open_ema"] * EYE_CLOSE_FACTOR)
    ctx["thr"] = thr

    now = time.time()
    ev = None

    if ratio < thr:  # closed
        if not ctx.get("is_closed", False):
            ctx["is_closed"] = True
            ctx["closed_t0"] = now
    else:            # open
        if ctx.get("is_closed", False):
            dur = now - ctx["closed_t0"]
            if BLINK_MIN <= dur <= BLINK_MAX:
                ev = "blink"
            elif dur >= LONG_BLINK_MIN:
                ev = "long"
            ctx["is_closed"] = False

    return ev, ratio, thr, ctx.get("is_closed", False)

def detect_blink_events_fixed(landmarks, w, h, ctx):
    """Legacy fixed-threshold version."""
    l = eye_ratio(landmarks, LEFT_EYE, w, h)
    r = eye_ratio(landmarks, RIGHT_EYE, w, h)
    ratio = (l + r) / 2.0
    thr = 0.20
    now = time.time()
    ev = None
    if ratio < thr:
        if not ctx["is_closed"]:
            ctx["is_closed"] = True
            ctx["closed_t0"] = now
    else:
        if ctx["is_closed"]:
            dur = now - ctx["closed_t0"]
            if BLINK_MIN <= dur <= BLINK_MAX:
                ev = "blink"
            elif dur >= LONG_BLINK_MIN:
                ev = "long"
            ctx["is_closed"] = False
    return ev, ratio, thr, ctx["is_closed"]

def map_head_direction(landmarks, ref_x, ref_y, thr_enter=HEAD_THR_ENTER, thr_exit=HEAD_THR_EXIT):
    nose_x = landmarks[NOSE_TIP].x
    nose_y = landmarks[NOSE_TIP].y
    dx = nose_x - ref_x
    dy = nose_y - ref_y

    def axis_state(delta, pos, neg):
        if delta > thr_enter:  return pos
        if delta < -thr_enter: return neg
        if -thr_exit <= delta <= thr_exit: return "STOP"
        return None

    horizontal = axis_state(dx, "RIGHT", "LEFT")
    vertical   = axis_state(dy, "BACKWARD", "FORWARD")  # down(+dy)=backward, up(-dy)=forward

    if vertical and vertical != "STOP":
        return vertical
    if horizontal and horizontal != "STOP":
        return horizontal
    if vertical == "STOP" or horizontal == "STOP":
        return "STOP"
    return None

# ================= Camera Loop =================
def camera_loop(ws_client: WSClient):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        ws_client.emit("ERROR", {"message": "Camera not available"})
        if LOG_TO_CONSOLE:
            log.error("Camera not available")
        return

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)

    # Enhanced tracking context
    blink_ctx = {"is_closed": False, "closed_t0": 0.0, "open_ema": None, "thr": None}
    face_detection_fallback = True  # Use MediaPipe as fallback
    
    log.info(f"Starting camera loop with {'YOLOv8 + MediaPipe' if EYE_TRACKER.yolo_model else 'MediaPipe only'} tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with enhanced face mesh
        result = EYE_TRACKER.face_mesh.process(rgb)

        # Try YOLOv8 face detection first, fallback to MediaPipe
        faces_detected = False
        yolo_faces = None
        
        if EYE_TRACKER.yolo_model and face_detection_fallback:
            yolo_faces = EYE_TRACKER.detect_faces_yolo(frame)
            if yolo_faces:
                # Draw YOLO face detection boxes for debugging
                for face in yolo_faces:
                    x1, y1, x2, y2 = face['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face: {face['confidence']:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if result.multi_face_landmarks:
            faces_detected = True
            lms = result.multi_face_landmarks[0].landmark
            STATE.last_seen_face = time.time()
            
            # Enhanced eye interaction detection
            eye_interactions = EYE_TRACKER.detect_eye_interactions(lms, w, h)
            
            # Log gaze information for debugging
            if eye_interactions['gaze_point'] and SHOW_WINDOW:
                gx, gy = eye_interactions['gaze_point']
                cv2.circle(frame, (int(gx), int(gy)), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"Gaze: ({int(gx)}, {int(gy)})", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Head calibration request
                if STATE.calibrate_requested.is_set():
                    STATE.ref_x, STATE.ref_y = lms[NOSE_TIP].x, lms[NOSE_TIP].y
                    STATE.calibrate_requested.clear()
                    ws_client.emit("CALIBRATED")

                # Eye baseline reset request
                if STATE.eye_calibrate_requested.is_set():
                    blink_ctx["open_ema"] = None
                    blink_ctx["thr"] = None
                    STATE.eye_calibrate_requested.clear()
                    ws_client.emit("CALIBRATED_EYES")

                # First-time head calibration
                if STATE.ref_x is None or STATE.ref_y is None:
                    STATE.ref_x, STATE.ref_y = lms[NOSE_TIP].x, lms[NOSE_TIP].y
                    ws_client.emit("CALIBRATED")

                # ---- Enhanced Eye logic with YOLOv8 + MediaPipe ----
                # Use enhanced blink detection
                if USE_DYNAMIC_EYE_THR:
                    ev, ratio, thr, is_closed = detect_blink_events_dynamic(lms, w, h, blink_ctx)
                else:
                    ev, ratio, thr, is_closed = detect_blink_events_fixed(lms, w, h, blink_ctx)
                
                # Process enhanced eye interactions
                if eye_interactions['saccade']:
                    log.debug("Saccade detected - rapid eye movement")
                
                if eye_interactions['gaze_point'] and STATE.eye_interaction_enabled:
                    # Future: Map gaze to UI interaction zones
                    gaze_x, gaze_y = eye_interactions['gaze_point']
                    # This could be used for direct UI control in future versions

                now = time.time()

                # Process blink events with debouncing
                if ev in ["blink", "long"]:
                    # Debounce blink processing
                    if now - STATE.last_blink_process >= BLINK_DEBOUNCE_TIME:
                        STATE.last_blink_process = now
                        
                        if ev == "long":
                            # Handle long blink immediately
                            handle_blink_event("long", ws_client)
                            STATE.blink_timestamps.clear()
                        
                        elif ev == "blink":
                            # Add to blink sequence for classification
                            STATE.blink_timestamps.append(now)

                # Simple blink processing on every frame
                if STATE.blink_timestamps:
                    # Clean old timestamps (keep only recent ones)
                    cutoff_time = now - 2.0  # Keep blinks for 2 seconds
                    STATE.blink_timestamps = [t for t in STATE.blink_timestamps if t > cutoff_time]
                    
                    # Try to classify blink sequence
                    if len(STATE.blink_timestamps) > 0:
                        timestamps_debug = [f"{now - t:.2f}s" for t in STATE.blink_timestamps]
                        log.debug(f"🔍 Processing {len(STATE.blink_timestamps)} blinks: {timestamps_debug}")
                    
                    blink_type = classify_blink_sequence(STATE.blink_timestamps, now)
                    if blink_type:
                        log.info(f"🎯 BLINK EVENT: {blink_type} in mode {STATE.current_mode}")
                        handle_blink_event(blink_type, ws_client)
                        
                        # Remove processed timestamps
                        if blink_type == "double":
                            # Remove the two most recent blinks
                            STATE.blink_timestamps = STATE.blink_timestamps[:-2] if len(STATE.blink_timestamps) >= 2 else []
                        elif blink_type == "single":
                            # Remove the oldest blink
                            STATE.blink_timestamps = STATE.blink_timestamps[1:]

                # ---- Head logic (only in wheelchair mode) ----
                if not should_ignore_head_movement():
                    new_dir = map_head_direction(lms, STATE.ref_x, STATE.ref_y)
                    tnow = time.time()
                    
                    # Calculate movement intensity
                    if STATE.ref_x is not None and STATE.ref_y is not None:
                        dx = lms[NOSE_TIP].x - STATE.ref_x
                        dy = lms[NOSE_TIP].y - STATE.ref_y
                        STATE.movement_intensity = calculate_movement_intensity(dx, dy)
                    
                    if new_dir is not None:
                        if new_dir != STATE.last_head_dir or (tnow - STATE.last_head_emit) >= HEAD_EVENT_MIN_INTERVAL:
                            STATE.last_head_dir = new_dir
                            STATE.last_head_emit = tnow
                            STATE.last_activity_time = tnow
                            broadcast_head_move(ws_client, new_dir)
                
                # Check for safety timeout
                check_safety_timeout(ws_client)
                
                # Update battery and send periodic status updates
                update_battery_simulation(STATE.current_mode == MODE_WHEELCHAIR and STATE.last_head_dir != "STOP")
                
                # Send periodic system status (every 2 seconds)
                if now - STATE.last_metrics_update > 2.0:
                    broadcast_system_status(ws_client)

                # ---- Overlay ----
                if SHOW_WINDOW:
                    with STATE.lock:
                        current_mode = STATE.current_mode
                        highlighted = STATE.highlighted_place
                        selected = STATE.selected_place
                    cv2.putText(frame, f"Mode: {current_mode}", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    if current_mode == MODE_WHEELCHAIR:
                        cv2.putText(frame, f"Head: {STATE.last_head_dir}", (10, 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    elif current_mode == MODE_PLACE:
                        cv2.putText(frame, f"Highlight: {highlighted}", (10, 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        if selected:
                            cv2.putText(frame, f"Selected: {selected}", (10, 78),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                    # Enhanced eye debug overlay with blink timing
                    blink_status = "CLOSED" if is_closed else "OPEN"
                    eye_color = (0, 0, 255) if is_closed else (0, 255, 0)  # Red when closed, green when open
                    cv2.putText(frame, f"Eyes: {blink_status} ratio:{ratio:.3f} thr:{(thr or 0):.3f}",
                                (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
                    
                    # Show blink timing if eyes are closed
                    if is_closed and blink_ctx.get("closed_t0"):
                        closed_duration = now - blink_ctx["closed_t0"]
                        duration_color = (0, 165, 255)  # Orange
                        status_text = f"Closed: {closed_duration:.2f}s"
                        
                        if closed_duration >= LONG_BLINK_MIN:
                            duration_color = (0, 0, 255)  # Red for long blinks
                            status_text = f"LONG BLINK: {closed_duration:.2f}s ✓"
                        elif closed_duration >= BLINK_MIN:
                            duration_color = (255, 255, 0)  # Cyan for potential blinks
                            status_text = f"Blink: {closed_duration:.2f}s"
                        else:
                            status_text = f"Closing: {closed_duration:.2f}s"
                            
                        cv2.putText(frame, status_text, (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, duration_color, 2)
                        
                        # Show progress bar for long blink
                        if closed_duration > 0.2:  # Show progress after 0.2s
                            progress = min(1.0, closed_duration / LONG_BLINK_MIN)
                            bar_width = 200
                            bar_height = 10
                            cv2.rectangle(frame, (10, 150), (10 + bar_width, 150 + bar_height), (50, 50, 50), -1)
                            cv2.rectangle(frame, (10, 150), (10 + int(bar_width * progress), 150 + bar_height), duration_color, -1)
                    
                    # Show recent blink events count
                    if STATE.blink_timestamps:
                        recent_blinks = len([t for t in STATE.blink_timestamps if now - t <= 2.0])  # Blinks in last 2 seconds
                        cv2.putText(frame, f"Recent blinks: {recent_blinks}", (10, 154),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Enhanced tracking info
                    if eye_interactions['gaze_point']:
                        gx, gy = eye_interactions['gaze_point']
                        cv2.putText(frame, f"Gaze smoothed: {len(STATE.gaze_history)} samples",
                                   (10, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 2)
                    
                    if eye_interactions['saccade']:
                        cv2.putText(frame, "SACCADE", (10, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Show tracking method
                    method = "YOLOv8+MP" if EYE_TRACKER.yolo_model else "MediaPipe"
                    cv2.putText(frame, f"Tracking: {method}", (10, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)

        else:
            faces_detected = False
        
        # Handle no face detection
        if not faces_detected:
            if time.time() - STATE.last_seen_face > FACE_LOST_TIMEOUT:
                check_safety_timeout(ws_client)
            if SHOW_WINDOW:
                method = "YOLOv8+MediaPipe" if EYE_TRACKER.yolo_model else "MediaPipe"
                cv2.putText(frame, f"No face detected ({method})", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show YOLO detection status if available
                if yolo_faces:
                    cv2.putText(frame, f"YOLO detected {len(yolo_faces)} face(s)", (10, 56),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        if SHOW_WINDOW:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                STATE.calibrate_requested.set()
            if key == ord('e'):
                STATE.eye_calibrate_requested.set()
            if key == ord('m'):
                # Manual mode toggle for testing
                with STATE.lock:
                    if STATE.current_mode == MODE_STOP:
                        STATE.current_mode = MODE_WHEELCHAIR
                    elif STATE.current_mode == MODE_WHEELCHAIR:
                        STATE.current_mode = MODE_PLACE
                        STATE.highlighted_place = ROOMS[STATE.current_index]
                    else:
                        STATE.current_mode = MODE_STOP
                broadcast_mode_change(ws_client)
                if STATE.current_mode == MODE_PLACE:
                    broadcast_place_highlight(ws_client, STATE.highlighted_place)
                elif STATE.current_mode == MODE_STOP:
                    broadcast_system_reset(ws_client)

        time.sleep(0.005)

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

# ================= Main =================
async def main():
    # Create WebSocket client
    ws_client = WSClient(WS_SERVER_URL)
    ws_client.loop = asyncio.get_event_loop()
    
    # Start camera loop in a separate thread
    camera_thread = Thread(target=camera_loop, args=(ws_client,), daemon=True)
    camera_thread.start()
    
    log.info("Movement detection client started. Press Ctrl+C to exit.")
    
    try:
        while True:
            try:
                # Try to connect
                await ws_client.connect()
                
                # Start listening for messages (this will block until connection is lost)
                await ws_client.start_listening()
                
            except websockets.ConnectionClosed:
                log.warning("Connection closed by server")
                ws_client.connected = False
                STATE.connected = False
                
            except Exception as e:
                log.error(f"Connection error: {e}")
                ws_client.connected = False
                STATE.connected = False
                
            # Wait before attempting to reconnect
            if not ws_client.connected:
                log.warning("Lost connection to server, attempting to reconnect...")
                await asyncio.sleep(2)
                
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        await ws_client.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Application terminated by user")
