import asyncio
import websockets
import logging
import os
from aiohttp import web
from aiohttp.web import Response
import json

# Cloud deployment configuration
WS_PORT = int(os.environ.get('PORT', 5000))
WS_HOST = os.environ.get('WS_HOST', '0.0.0.0')

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("WSServer")

class WSServer:
    def __init__(self):
        self.clients = set()

    async def handle_client(self, websocket, path=None):
        self.clients.add(websocket)
        peer = getattr(websocket, "remote_address", None)
        log.info(f"✅ Client connected: {peer}")

        try:
            async for message in websocket:
                log.info(f"📩 Received: {message}")

                # Broadcast to everyone else
                dead = []
                for client in list(self.clients):
                    if client is websocket:
                        continue
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        dead.append(client)
                    except Exception as err:
                        log.error(f"❌ Error sending to client: {err}")
                        dead.append(client)
                for d in dead:
                    self.clients.discard(d)

        except websockets.exceptions.ConnectionClosed as e:
            log.info(f"🔌 Client closed: code={e.code} reason={e.reason}")
        except Exception as err:
            log.error(f"❌ Handler error: {err}")
        finally:
            self.clients.discard(websocket)
            log.info("❌ Client disconnected")

    # Health check endpoint for cloud platforms
    async def health_check(self, request):
        return Response(
            text=json.dumps({
                "status": "healthy",
                "clients": len(self.clients),
                "service": "gesture-control-ws"
            }),
            content_type="application/json"
        )

async def main():
    server = WSServer()
    
    # Create HTTP server for health checks
    app = web.Application()
    app.router.add_get('/health', server.health_check)
    app.router.add_get('/', lambda r: Response(text="Gesture Control WebSocket Server"))
    
    # Start HTTP server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, WS_HOST, WS_PORT)
    await site.start()
    
    # Start WebSocket server on different port for development
    # In production, use same port with path routing
    ws_port = WS_PORT if os.environ.get('PORT') else 5001
    
    log.info(f"🚀 Starting WebSocket server on {WS_HOST}:{ws_port}")
    log.info(f"🌐 Health check available at http://{WS_HOST}:{WS_PORT}/health")
    
    async with websockets.serve(
        server.handle_client, 
        WS_HOST, 
        ws_port,
        ping_interval=20,
        ping_timeout=10,
        compression=None
    ):
        log.info("✅ WebSocket server started successfully")
        # Keep the server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            log.info("🛑 Server shutdown requested")
        finally:
            await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())