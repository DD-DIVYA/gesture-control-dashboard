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

# Try to import MediaPipe for face detection
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    log.info("✅ MediaPipe loaded for face detection")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    log.warning("❌ MediaPipe not available - running without face detection")

# Connected WebSocket clients
connected_clients = set()

class FaceDetector:
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
        else:
            self.face_detection = None
    
    def detect_faces(self, image_data):
        """Detect faces in base64 image data from browser"""
        if not MEDIAPIPE_AVAILABLE or not self.face_detection:
            return {"faces_detected": False, "face_count": 0}
        
        try:
            # Decode base64 image (remove data:image/jpeg;base64, prefix)
            img_data = image_data.split(',')[1] if ',' in image_data else image_data
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"faces_detected": False, "face_count": 0, "error": "Invalid image"}
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find faces
            results = self.face_detection.process(rgb_image)
            
            face_count = 0
            faces_info = []
            
            if results.detections:
                face_count = len(results.detections)
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    faces_info.append({
                        "confidence": detection.score[0],
                        "bbox": {
                            "x": bbox.xmin,
                            "y": bbox.ymin, 
                            "width": bbox.width,
                            "height": bbox.height
                        }
                    })
            
            return {
                "faces_detected": face_count > 0,
                "face_count": face_count,
                "faces": faces_info,
                "status": "success"
            }
            
        except Exception as e:
            log.error(f"Face detection error: {e}")
            return {"faces_detected": False, "face_count": 0, "error": str(e)}

face_detector = FaceDetector()

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
                        result = face_detector.detect_faces(image_data)
                        
                        # Send back face detection results
                        await ws.send_json({
                            "type": "face_detection_result",
                            "event": "FACE_DETECTED" if result['faces_detected'] else "NO_FACE",
                            "payload": result
                        })
                
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