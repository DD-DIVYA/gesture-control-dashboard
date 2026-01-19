import asyncio
import json
import logging
import base64
import cv2
import numpy as np
from aiohttp import web, WSMsgType
import os

# Cloud configuration
PORT = int(os.environ.get('PORT', 10000))
HOST = '0.0.0.0'

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GestureControl")

# Try to import MediaPipe for face detection with robust error handling
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully")
    log.info("✅ MediaPipe loaded successfully")
except Exception as e:
    print(f"❌ MediaPipe initialization error: {e}")
    log.error(f"❌ MediaPipe initialization error: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp_face_mesh = None
    mp_face_detection = None
    mp_drawing = None
    mp_drawing_styles = None

if MEDIAPIPE_AVAILABLE:
    log.info("✅ Face detection enabled")
else:
    log.info("🌩️ Running in cloud simulation mode without face detection")

# Connected WebSocket clients
connected_clients = set()

class FaceDetector:
    def __init__(self):
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE and mp_face_mesh:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                log.info("✅ Face mesh model initialized for blink detection")
            except Exception as e:
                log.error(f"❌ Failed to initialize face mesh: {e}")
                self.face_mesh = None
        else:
            log.info("🌩️ Face mesh disabled - will simulate responses")
    
    def detect_faces(self, image_data):
        """Detect faces and extract landmarks for blink detection from base64 image data"""
        if not MEDIAPIPE_AVAILABLE or not self.face_mesh:
            # Return simulated response when MediaPipe not available
            return {
                "faces_detected": True,  # Simulate face detection for demo
                "face_count": 1,
                "landmarks": [],  # Return empty list instead of boolean
                "status": "simulated",
                "message": "MediaPipe not available - simulated response"
            }
        
        try:
            # Decode base64 image (remove data:image/jpeg;base64, prefix)
            img_data = image_data.split(',')[1] if ',' in image_data else image_data
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"faces_detected": False, "face_count": 0, "landmarks": [], "error": "Invalid image"}
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find face landmarks
            results = self.face_mesh.process(rgb_image)
            
            face_count = 0
            landmarks_data = []
            
            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                landmarks_data = results.multi_face_landmarks
                
                # Log successful processing for debugging
                log.info(f"📷 Processed frame: {face_count} face(s) detected with landmarks")
            
            return {
                "faces_detected": face_count > 0,
                "face_count": face_count,
                "landmarks": landmarks_data,  # Return actual landmark data
                "status": "success"
            }
            
        except Exception as e:
            log.error(f"Face detection error: {e}")
            return {"faces_detected": False, "face_count": 0, "landmarks": [], "error": str(e)}

# System state for mode management
class SystemState:
    def __init__(self):
        self.current_mode = 'STOP'
        self.highlighted_place = None
        self.selected_place = None
        self.places = ["Kitchen", "Bedroom", "Living Room", "Restroom"]
        self.place_index = 0
        
    def handle_blink(self, blink_type):
        """Handle different blink types and return events to send"""
        events = []
        
        if blink_type == "long":
            # Long blink always goes to STOP mode
            self.current_mode = 'STOP'
            self.highlighted_place = None
            self.selected_place = None
            events.append({
                "event": "MODE_CHANGE",
                "payload": {"mode": "STOP"}
            })
            events.append({
                "event": "BLINK_EVENT",
                "payload": {
                    "type": "long",
                    "message": "Long blink - STOP mode activated",
                    "action": "stop"
                }
            })
            
        elif blink_type == "single":
            if self.current_mode == 'STOP':
                # Single blink from STOP → WHEELCHAIR
                self.current_mode = 'WHEELCHAIR'
                events.append({
                    "event": "MODE_CHANGE",
                    "payload": {"mode": "WHEELCHAIR"}
                })
                events.append({
                    "event": "BLINK_EVENT",
                    "payload": {
                        "type": "single",
                        "message": "Single blink - WHEELCHAIR mode activated",
                        "action": "mode_change"
                    }
                })
                
            elif self.current_mode == 'PLACE':
                # Single blink in PLACE mode → Navigate through places
                self.place_index = (self.place_index + 1) % len(self.places)
                self.highlighted_place = self.places[self.place_index]
                events.append({
                    "event": "PLACE_HIGHLIGHT",
                    "payload": {"place": self.highlighted_place}
                })
                events.append({
                    "event": "BLINK_EVENT",
                    "payload": {
                        "type": "single",
                        "message": f"Navigating to {self.highlighted_place}",
                        "action": "navigate_place"
                    }
                })
                
        elif blink_type == "double":
            # Double blink ALWAYS activates PLACE mode (from any mode)
            if self.current_mode != 'PLACE':
                # Enter PLACE mode and highlight first place
                self.current_mode = 'PLACE'
                self.place_index = 0
                self.highlighted_place = self.places[0]
                events.append({
                    "event": "MODE_CHANGE", 
                    "payload": {"mode": "PLACE"}
                })
                events.append({
                    "event": "PLACE_HIGHLIGHT",
                    "payload": {"place": self.highlighted_place}
                })
                events.append({
                    "event": "BLINK_EVENT",
                    "payload": {
                        "type": "double",
                        "message": f"Double blink - PLACE mode activated, {self.highlighted_place} highlighted",
                        "action": "mode_change"
                    }
                })
            else:
                # Already in PLACE mode → Select highlighted place and return to WHEELCHAIR
                if self.highlighted_place:
                    self.selected_place = self.highlighted_place
                    events.append({
                        "event": "PLACE_SELECT",
                        "payload": {"place": self.selected_place}
                    })
                    
                    # Return to WHEELCHAIR mode after selection
                    self.current_mode = 'WHEELCHAIR'
                    events.append({
                        "event": "MODE_CHANGE",
                        "payload": {"mode": "WHEELCHAIR"}
                    })
                    events.append({
                        "event": "BLINK_EVENT",
                        "payload": {
                            "type": "double", 
                            "message": f"Double blink - Selected {self.selected_place}, returning to WHEELCHAIR",
                            "action": "select_place"
                        }
                    })
                    
        return events

# Global system state
system_state = SystemState()

face_detector = FaceDetector()

# Blink detection logic using eye landmarks
class BlinkDetector:
    def __init__(self):
        self.blink_threshold = 0.25
        self.long_blink_threshold = 2.0  # Increase to 2.0 seconds for long blink
        self.double_blink_window = 4.0   # Increase to 4 seconds between double blinks
        
        self.last_blink_time = 0
        self.blink_cooldown = 0.1  # Reduce cooldown to 0.1 seconds 
        self.pending_first_blink = False
        self.first_blink_time = 0
        
        # Long blink detection
        self.eyes_closed_start = 0
        self.eyes_currently_closed = False
        
        # EAR tracking
        self.ear_history = []
        self.ear_threshold = 0.21
        
    def calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio from face landmarks"""
        try:
            if not landmarks or len(landmarks) == 0:
                return 0.3  # Default open eyes value
            
            face_landmarks = landmarks[0].landmark
            
            # Left eye landmark indices (MediaPipe Face Mesh)
            left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            # Right eye landmark indices  
            right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Calculate EAR for left eye (simplified)
            left_p1 = face_landmarks[159]  # Left eye top
            left_p2 = face_landmarks[145]  # Left eye bottom
            left_p3 = face_landmarks[133]  # Left eye left corner
            left_p4 = face_landmarks[33]   # Left eye right corner
            
            # Calculate vertical distance
            left_vertical = abs(left_p1.y - left_p2.y)
            # Calculate horizontal distance  
            left_horizontal = abs(left_p3.x - left_p4.x)
            
            # Calculate EAR for right eye (simplified)
            right_p1 = face_landmarks[386]  # Right eye top
            right_p2 = face_landmarks[374]  # Right eye bottom
            right_p3 = face_landmarks[362]  # Right eye left corner
            right_p4 = face_landmarks[263]  # Right eye right corner
            
            right_vertical = abs(right_p1.y - right_p2.y)
            right_horizontal = abs(right_p3.x - right_p4.x)
            
            # Calculate EAR (Eye Aspect Ratio)
            if left_horizontal > 0 and right_horizontal > 0:
                left_ear = left_vertical / left_horizontal
                right_ear = right_vertical / right_horizontal
                ear = (left_ear + right_ear) / 2.0
                return ear
            else:
                return 0.3  # Default value
                
        except Exception as e:
            log.error(f"EAR calculation error: {e}")
            return 0.3
        
    def detect_blink(self, landmarks):
        """Detect single, double, and long blinks using real eye tracking"""
        import time
        current_time = time.time()
        
        if not landmarks or len(landmarks) == 0:
            return None
            
        try:
            # Calculate current EAR
            current_ear = self.calculate_ear(landmarks)
            
            # Keep a history of EAR values (last 10 frames)
            self.ear_history.append(current_ear)
            if len(self.ear_history) > 10:
                self.ear_history.pop(0)
                
            # Check if eyes are currently closed
            eyes_closed = current_ear < self.ear_threshold
            
            if eyes_closed and not self.eyes_currently_closed:
                # Eyes just closed
                self.eyes_currently_closed = True
                self.eyes_closed_start = current_time
                log.info(f"👁️ Eyes closed - EAR: {current_ear:.3f}")
                
            elif not eyes_closed and self.eyes_currently_closed:
                # Eyes just opened - blink detected
                self.eyes_currently_closed = False
                blink_duration = current_time - self.eyes_closed_start
                
                log.info(f"👁️ Eyes opened - Blink duration: {blink_duration:.2f}s")
                
                # Determine blink type based on duration
                if blink_duration >= self.long_blink_threshold:
                    # Long blink detected
                    self.last_blink_time = current_time
                    self.pending_first_blink = False
                    log.info("👁️ LONG BLINK detected!")
                    return {"type": "long", "timestamp": current_time}
                    
                else:
                    # Short blink - check for single/double
                    if self.pending_first_blink:
                        # This is the second blink - check timing
                        time_since_first = current_time - self.first_blink_time
                        if time_since_first < self.double_blink_window:
                            # Double blink detected
                            self.pending_first_blink = False
                            self.last_blink_time = current_time
                            log.info(f"👁️ DOUBLE BLINK detected! Time between blinks: {time_since_first:.2f}s")
                            return {"type": "double", "timestamp": current_time}
                        else:
                            # Too late for double blink, process first blink as single and start new sequence
                            self.pending_first_blink = False
                            self.last_blink_time = current_time
                            log.info(f"👁️ SINGLE BLINK detected (too late for double - {time_since_first:.2f}s)!")
                            # Return the old single blink first
                            return {"type": "single", "timestamp": self.first_blink_time}
                    else:
                        # First blink - wait to see if there's a second one
                        self.pending_first_blink = True
                        self.first_blink_time = current_time
                        log.info(f"👁️ First blink detected - waiting {self.double_blink_window:.1f}s for potential second...")
                        
            # Check timeout - use shorter timeout in PLACES mode for faster navigation
            elif self.pending_first_blink:
                # Import system_state to check current mode
                from __main__ import system_state
                
                # Use shorter timeout in PLACES mode (1 second) vs other modes (4 seconds)
                timeout_window = 1.0 if system_state.current_mode == 'PLACE' else self.double_blink_window
                
                if current_time - self.first_blink_time > timeout_window:
                    # Timeout - treat as single blink
                    time_waited = current_time - self.first_blink_time
                    self.pending_first_blink = False
                    self.last_blink_time = current_time
                    log.info(f"👁️ SINGLE BLINK detected (timeout after {time_waited:.2f}s in {system_state.current_mode} mode)!")
                    return {"type": "single", "timestamp": self.first_blink_time}
                
        except Exception as e:
            log.error(f"Blink detection error: {e}")
            
        return None

# Head movement detection logic using head pose
class HeadMovementDetector:
    def __init__(self):
        self.center_x = 0.5  # Center reference point
        self.center_y = 0.5
        self.movement_threshold = 0.02  # Sensitivity for head movement
        self.last_direction = 'STOP'
        self.last_movement_time = 0
        self.movement_cooldown = 0.5  # 0.5 second cooldown between movements
        
    def detect_head_movement(self, landmarks):
        """Detect real head movement direction from face landmarks"""
        if not landmarks or len(landmarks) == 0:
            return None
            
        import time
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_movement_time < self.movement_cooldown:
            return None
            
        try:
            # Get first face landmarks
            face_landmarks = landmarks[0].landmark
            
            # Key landmark indices for head pose estimation
            # Nose tip
            nose_tip = face_landmarks[1]
            # Chin
            chin = face_landmarks[18] 
            # Forehead center  
            forehead = face_landmarks[10]
            # Left face edge
            left_face = face_landmarks[234]
            # Right face edge
            right_face = face_landmarks[454]
            
            # Calculate head orientation
            # Left/Right movement: compare left and right face edges
            left_right_diff = right_face.x - left_face.x
            head_center_x = (left_face.x + right_face.x) / 2
            
            # Up/Down movement: compare nose tip and chin positions
            nose_chin_diff = chin.y - nose_tip.y
            head_center_y = (nose_tip.y + chin.y) / 2
            
            direction = None
            motor_speed = 0.0
            movement_intensity = 0.0
            
            # Detect left/right head tilt
            if head_center_x < 0.48:  # Head tilted right (person's right)
                direction = 'RIGHT'
                motor_speed = min(0.8, (0.5 - head_center_x) * 4)
                movement_intensity = min(0.9, (0.5 - head_center_x) * 5)
                
            elif head_center_x > 0.52:  # Head tilted left (person's left) 
                direction = 'LEFT'
                motor_speed = min(0.8, (head_center_x - 0.5) * 4)
                movement_intensity = min(0.9, (head_center_x - 0.5) * 5)
                
            # Detect forward/backward head movement
            elif head_center_y < 0.48:  # Head moved up (forward)
                direction = 'FORWARD' 
                motor_speed = min(0.8, (0.5 - head_center_y) * 4)
                movement_intensity = min(0.9, (0.5 - head_center_y) * 5)
                
            elif head_center_y > 0.52:  # Head moved down (backward)
                direction = 'BACKWARD'
                motor_speed = min(0.8, (head_center_y - 0.5) * 4) 
                movement_intensity = min(0.9, (head_center_y - 0.5) * 5)
                
            # If no significant movement, return STOP
            else:
                if self.last_direction != 'STOP':
                    direction = 'STOP'
                    motor_speed = 0.0
                    movement_intensity = 0.0
                else:
                    return None  # No change needed
            
            # Only send if direction changed
            if direction and direction != self.last_direction:
                self.last_direction = direction
                self.last_movement_time = current_time
                
                return {
                    'direction': direction,
                    'motor_speed': motor_speed,
                    'movement_intensity': movement_intensity,
                    'battery_percentage': 85.0,  # Static for demo
                    'total_distance': 25.5,     # Static for demo
                    'session_time': int(current_time % 3600)
                }
                    
        except Exception as e:
            log.error(f"Head movement detection error: {e}")
            
        return None

blink_detector = BlinkDetector()
head_movement_detector = HeadMovementDetector()

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    connected_clients.add(ws)
    log.info(f"✅ WebSocket client connected. Total clients: {len(connected_clients)}")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get('type', data.get('event'))
                
                if msg_type == 'camera_frame':
                    # Process camera frame for face detection
                    image_data = data.get('image')
                    if image_data:
                        log.info("📩 Received camera frame for processing")
                        result = face_detector.detect_faces(image_data)
                        
                        # Send back face detection results
                        await ws.send_json({
                            "type": "face_detection_result",
                            "event": "FACE_STATUS",
                            "payload": {
                                "active": result['faces_detected'],
                                "faces": result['face_count']
                            }
                        })
                        
                        # Check for blinks if face is detected
                        if result['faces_detected'] and result.get('landmarks'):
                            landmarks = result.get('landmarks', [])
                            blink_result = blink_detector.detect_blink(landmarks)
                            
                            if blink_result:
                                # Handle different blink types
                                blink_type = blink_result["type"]
                                events = system_state.handle_blink(blink_type)
                                
                                # Send all events
                                for event in events:
                                    await ws.send_json(event)
                                    log.info(f"📤 Sent event: {event['event']} - {event['payload']}")
                            
                            # Check for head movements when in WHEELCHAIR mode
                            if system_state.current_mode == 'WHEELCHAIR':
                                head_movement = head_movement_detector.detect_head_movement(landmarks)
                                if head_movement:
                                    await ws.send_json({
                                        "event": "HEAD_MOVE",
                                        "payload": head_movement
                                    })
                                    log.info(f"🎯 Head movement: {head_movement['direction']} - Speed: {head_movement['motor_speed']:.2f}")
                
                elif msg_type == 'ping':
                    # Keep connection alive
                    await ws.send_json({
                        "type": "pong",
                        "payload": {"status": "ok"}
                    })
                
                elif msg_type == 'CALIBRATE':
                    # Handle calibration request
                    await ws.send_json({
                        "event": "CALIBRATED",
                        "payload": {"status": "calibrated"}
                    })
                
                # Broadcast other messages to all clients
                else:
                    log.info(f"📩 Received: {msg_type}")
                    await broadcast_message(data, exclude=ws)

            elif msg.type == WSMsgType.ERROR:
                log.error(f"WebSocket error: {ws.exception()}")

    except Exception as e:
        log.error(f"WebSocket exception: {e}")

    finally:
        connected_clients.discard(ws)
        log.info(f"🔌 WebSocket client disconnected. Remaining: {len(connected_clients)}")

    return ws

async def broadcast_message(data, exclude=None):
    """Broadcast message to all connected clients except the sender"""
    global connected_clients
    if not connected_clients:
        return
    
    dead_clients = set()
    for client in connected_clients:
        if client == exclude:
            continue
        try:
            await client.send_json(data)
        except Exception:
            dead_clients.add(client)
    
    # Remove dead clients
    connected_clients -= dead_clients

async def health_check(request):
    return web.json_response({
        "status": "healthy",
        "service": "gesture-control-backend-camera",
        "clients": len(connected_clients),
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "features": ["face_detection", "websocket_communication", "camera_processing"]
    })

# Background status broadcaster
async def status_broadcaster():
    global connected_clients
    while True:
        await asyncio.sleep(5)
        if connected_clients:
            message = {
                "event": "SYSTEM_STATUS", 
                "payload": {
                    "mode": "WHEELCHAIR",
                    "battery": 85,
                    "signal": "excellent",
                    "connected_clients": len(connected_clients),
                    "face_detection": MEDIAPIPE_AVAILABLE
                }
            }
            dead_clients = set()
            for ws in connected_clients:
                try:
                    await ws.send_json(message)
                except:
                    dead_clients.add(ws)
            connected_clients -= dead_clients

# Create the web application
app = web.Application()
app.router.add_get('/', health_check)
app.router.add_get('/health', health_check)
app.router.add_get('/ws', websocket_handler)

async def main():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()
    
    log.info(f"🌐 Gesture Control Server started on http://{HOST}:{PORT}")
    log.info(f"🔗 WebSocket endpoint: ws://{HOST}:{PORT}/ws")
    log.info(f"❤️  Health check: http://{HOST}:{PORT}/health")
    log.info(f"📷 Camera processing: {'✅ Enabled' if MEDIAPIPE_AVAILABLE else '❌ Disabled'}")
    
    # Start background status broadcaster
    asyncio.create_task(status_broadcaster())
    
    # Keep server running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        log.info("🛑 Server shutdown")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("👋 Goodbye!")
