import os
import asyncio
import json
import logging
from aiohttp import web

# ---------------- CONFIG ----------------
PORT = int(os.environ.get("PORT", 10000))
HOST = "0.0.0.0"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RenderWS")

connected_clients = set()

# ---------------- WEBSOCKET HANDLER ----------------
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    connected_clients.add(ws)
    log.info("✅ WebSocket client connected")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                log.info(f"📩 Received: {data}")

                # Echo back to frontend
                await ws.send_json({
                    "event": "CONNECTED",
                    "payload": {
                        "status": "ok"
                    }
                })

            elif msg.type == web.WSMsgType.ERROR:
                log.error(f"WebSocket error: {ws.exception()}")

    except Exception as e:
        log.error(f"WebSocket exception: {e}")

    finally:
        connected_clients.remove(ws)
        log.info("🔌 WebSocket client disconnected")

    return ws

# ---------------- HEALTH CHECK ----------------
async def health(request):
    return web.json_response({
        "status": "healthy",
        "clients": len(connected_clients)
    })

# ---------------- BACKGROUND STATUS PUSH ----------------
async def status_broadcaster():
    while True:
        await asyncio.sleep(3)
        if connected_clients:
            message = {
                "event": "SYSTEM_STATUS",
                "payload": {
                    "mode": "STOP",
                    "battery": 90,
                    "connected_clients": len(connected_clients)
                }
            }
            for ws in list(connected_clients):
                try:
                    await ws.send_json(message)
                except:
                    connected_clients.discard(ws)

# ---------------- MAIN ----------------
async def main():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, HOST, PORT)
    await site.start()

    log.info(f"🚀 Server running on http://{HOST}:{PORT}")
    log.info(f"🔗 WebSocket endpoint: /ws")

    asyncio.create_task(status_broadcaster())

    await asyncio.Event().wait()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
