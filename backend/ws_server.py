import asyncio
import websockets
import logging

WS_PORT = 5000

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("WSServer")

class WSServer:
    def __init__(self):
        self.clients = set()

    async def handle_client(self, websocket, path=None):   # <- path optional for compatibility
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

async def main():
    server = WSServer()

    # Use async context manager to keep server alive and avoid GC.
    # Add gentle keepalive; adjust if your network is flaky.
    async with websockets.serve(
        server.handle_client,
        host="127.0.0.1",  # or "0.0.0.0" if connecting from other devices
        port=WS_PORT,
        ping_interval=20,
        ping_timeout=20,
        max_size=2**20
    ):
        log.info(f"🚀 WebSocket server started on ws://127.0.0.1:{WS_PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("🛑 Server stopped by user")
