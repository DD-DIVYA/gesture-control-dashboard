import cv2
import mediapipe as mp
import time
import asyncio
import websockets
import json
import logging
from threading import Thread, Event, Lock

# ================== CONFIG ==================
WS_SERVER_URL = "ws://localhost:5000"

# GUI preview
SHOW_WINDOW = True
WINDOW_NAME = "Face Controls (q quit, c head-cal, e eye-cal, m toggle-mode)"

# Logging
LOG_TO_CONSOLE = True

ROOMS = ["Kitchen", "Bedroom", "Living Room", "Restroom"]

# --- Blink timing (tune here) ---
BLINK_MIN = 0.08          # short blink lower bound (s)
BLINK_MAX = 0.45          # short blink upper bound (s)
LONG_BLINK_MIN = 0.50     # long blink lower bound (s)

# Triple detection windows
TRIPLE_WINDOW = 1.50      # 3 short blinks must all occur within this window (s)
TRIPLE_GAP_MAX = 0.60     # max gap between consecutive short blinks (s)
SINGLE_SELECT_DELAY = 0.35  # wait to commit single in place-mode (avoid triple)

# --- Dynamic eye thresholding ---
USE_DYNAMIC_EYE_THR = True
EYE_CLOSE_FACTOR = 0.72    # threshold = open_ema * factor
EYE_EMA_BETA = 0.90        # EMA for "open" ratio
EYE_MIN_THRESHOLD = 0.05   # never go below this

# Head tilt thresholds (normalized coords)
HEAD_THR_ENTER = 0.09
HEAD_THR_EXIT  = 0.06

# Head sampling debounce
HEAD_EVENT_MIN_INTERVAL = 0.08
TRACK_LOST_TIMEOUT = 1.0

# Modes
MODE_WHEEL = "wheelchair"
MODE_PLACE = "place"

# ================= Mediapipe =================
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE  = (33, 133, 159, 145)  # (outer, inner, upper, lower)
RIGHT_EYE = (362, 263, 386, 374)
NOSE_TIP = 1

# ================= Logging Setup =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("FaceWS")

# ================= Helpers =================
def room_slug(name: str) -> str:
    """lowercase, no spaces/punct → 'Living Room' -> 'livingroom'"""
    return "".join(ch for ch in name.lower() if ch.isalnum())

# ================= Shared State =================
class SharedState:
    def __init__(self):
        self.current_index = 0
        self.ref_x = None
        self.ref_y = None
        self.last_head_dir = "stop"
        self.last_head_emit = 0.0
        self.last_seen_face = time.time()
        self.calibrate_requested = Event()      # head
        self.eye_calibrate_requested = Event()  # eyes (baseline reset)
        self.reset_places_requested = Event()
        self.mode = MODE_WHEEL                  # start in wheelchair mode

        # NEW: place selection state
        self.selected_room = None               # str or None
        self.selection_committed = False        # True after PLACE_SELECT until highlight changes

        self.lock = Lock()
        
        # WebSocket client connection
        self.ws_client = None
        self.connected = False

STATE = SharedState()

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
        if -thr_exit <= delta <= thr_exit: return "stop"
        return None

    horizontal = axis_state(dx, "right", "left")
    vertical   = axis_state(dy, "backward", "forward")  # down(+dy)=backward, up(-dy)=forward

    if vertical and vertical != "stop":
        return vertical
    if horizontal and horizontal != "stop":
        return horizontal
    if vertical == "stop" or horizontal == "stop":
        return "stop"
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

    # Blink context
    blink_ctx = {"is_closed": False, "closed_t0": 0.0, "open_ema": None, "thr": None}

    short_times = []            # timestamps of short blinks (for triple)
    pending_single_time = None  # delay to commit single in place-mode

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                STATE.last_seen_face = time.time()

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

                # ---- Eye logic (mode & triple aware) ----
                if USE_DYNAMIC_EYE_THR:
                    ev, ratio, thr, is_closed = detect_blink_events_dynamic(lms, w, h, blink_ctx)
                else:
                    ev, ratio, thr, is_closed = detect_blink_events_fixed(lms, w, h, blink_ctx)

                now = time.time()

                # Keep only recent short blinks for triple logic
                short_times = [t for t in short_times if now - t <= TRIPLE_WINDOW]

                # ------------- Blink decisions -------------
                if ev == "blink":
                    short_times.append(now)
                    # cancel pending single (we're possibly building a burst)
                    if pending_single_time is not None:
                        pending_single_time = None

                    # Check triple
                    if len(short_times) >= 3:
                        t3, t2, t1 = short_times[-3], short_times[-2], short_times[-1]
                        cond_window = (t1 - t3) <= TRIPLE_WINDOW
                        cond_gaps = (t1 - t2) <= TRIPLE_GAP_MAX and (t2 - t3) <= TRIPLE_GAP_MAX
                        if cond_window and cond_gaps:
                            with STATE.lock:
                                STATE.mode = MODE_PLACE if STATE.mode == MODE_WHEEL else MODE_WHEEL
                                mode_now = STATE.mode
                                if mode_now == MODE_WHEEL:
                                    STATE.last_head_dir = "stop"
                            ws_client.emit("MODE_CHANGE", {"mode": mode_now})
                            if mode_now == MODE_PLACE:
                                ws_client.emit("PLACE_HIGHLIGHT", {"room": ROOMS[STATE.current_index]})
                                ws_client.emit("HEAD_MOVE", {"direction": "stop"})
                            short_times.clear()
                            pending_single_time = None
                        else:
                            # Not triple yet—if in PLACE mode, arm single select
                            if STATE.mode == MODE_PLACE and pending_single_time is None:
                                pending_single_time = now
                    else:
                        # Single candidate (PLACE mode only)
                        if STATE.mode == MODE_PLACE and pending_single_time is None:
                            pending_single_time = now

                elif ev == "long":
                    if STATE.mode == MODE_PLACE:
                        with STATE.lock:
                            if STATE.reset_places_requested.is_set():
                                STATE.current_index = 0
                                STATE.reset_places_requested.clear()
                            else:
                                STATE.current_index = (STATE.current_index + 1) % len(ROOMS)
                            room = ROOMS[STATE.current_index]
                            # changing highlight cancels previous selection
                            STATE.selection_committed = False
                            STATE.selected_room = None
                        ws_client.emit("PLACE_HIGHLIGHT", {"room": room})
                        short_times.clear()
                        pending_single_time = None
                    # In wheelchair mode: ignore long blink

                # Commit single-blink actions after delay, only in PLACE mode
                if STATE.mode == MODE_PLACE and pending_single_time is not None:
                    if (now - pending_single_time) >= SINGLE_SELECT_DELAY and len(short_times) == 1:
                        with STATE.lock:
                            room = ROOMS[STATE.current_index]
                            slug = room_slug(room)
                            if not STATE.selection_committed or STATE.selected_room != room:
                                # First single → SELECT
                                STATE.selection_committed = True
                                STATE.selected_room = room
                                # Structured + raw command
                                ws_client.emit("PLACE_SELECT", {"room": room})
                                ws_client.emit("COMMAND", {"text": f"{slug},select"})
                            else:
                                # Subsequent single → GO
                                ws_client.emit("PLACE_GO", {"room": STATE.selected_room})
                                ws_client.emit("COMMAND", {"text": f"{room_slug(STATE.selected_room)},go"})
                        short_times.clear()
                        pending_single_time = None

                # ---- Head logic (only in wheelchair mode) ----
                if STATE.mode == MODE_WHEEL:
                    new_dir = map_head_direction(lms, STATE.ref_x, STATE.ref_y)
                    tnow = time.time()
                    if new_dir is not None:
                        if new_dir != STATE.last_head_dir or (tnow - STATE.last_head_emit) >= HEAD_EVENT_MIN_INTERVAL:
                            STATE.last_head_dir = new_dir
                            STATE.last_head_emit = tnow
                            ws_client.emit("HEAD_MOVE", {"direction": new_dir})

                # ---- Overlay ----
                if SHOW_WINDOW:
                    with STATE.lock:
                        room = ROOMS[STATE.current_index]
                        mode_now = STATE.mode
                        sel_txt = f"Selected: {STATE.selected_room}" if STATE.selection_committed else "Selected: -"
                    cv2.putText(frame, f"Mode: {mode_now}", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    if mode_now == MODE_WHEEL:
                        cv2.putText(frame, f"Head: {STATE.last_head_dir}", (10, 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"Highlight: {room}", (10, 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, sel_txt, (10, 78),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                    # Eye debug overlay
                    cv2.putText(frame, f"Eye ratio:{ratio:.3f} thr:{(thr or 0):.3f} closed:{int(is_closed)}",
                                (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            else:
                # No face found
                if time.time() - STATE.last_seen_face > TRACK_LOST_TIMEOUT:
                    ws_client.emit("TRACKING", {"status": "lost"})
                if SHOW_WINDOW:
                    cv2.putText(frame, "No face detected", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
                    with STATE.lock:
                        STATE.mode = MODE_PLACE if STATE.mode == MODE_WHEEL else MODE_WHEEL
                        mode_now = STATE.mode
                    ws_client.emit("MODE_CHANGE", {"mode": mode_now})
                    if mode_now == MODE_PLACE:
                        ws_client.emit("PLACE_HIGHLIGHT", {"room": ROOMS[STATE.current_index]})
                        ws_client.emit("HEAD_MOVE", {"direction": "stop"})

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
