
import asyncio
import threading
import serial
import websockets

COM_PORT = 'COM4'
BAUD_RATE = 115200

connected_clients = set()


# WebSocket Handler
async def ws_handler(websocket):
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)

async def broadcast_telemetry(msg):
    if connected_clients:
        tasks = [asyncio.create_task(client.send(msg)) for client in connected_clients]
        await asyncio.gather(*tasks, return_exceptions=True)

# Serial Communication Thread: relays the MAX30003 telemetry coming from the
# STM32 to the browser dashboard. No synthetic ECG is generated here anymore.
def serial_worker(loop):
    print(f"[Serial] Connecting to STM32 on {COM_PORT} @ {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.02)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"[Serial] Connected successfully on {COM_PORT}!")
    except Exception as e:
        print(f"[Error] Failed to open {COM_PORT}: {e}")
        print("Please check Device Manager and ensure MATLAB or Chrome is not holding the port.")
        return

    print("[Serial] Relaying MAX30003 ECG telemetry from STM32...")

    rx_buf = ""
    while True:
        try:
            if ser.in_waiting > 0:
                raw_bytes = ser.read(ser.in_waiting)
                rx_buf += raw_bytes.decode('ascii', errors='ignore')

                while '\n' in rx_buf:
                    line, rx_buf = rx_buf.split('\n', 1)
                    line = line.strip()
                    if line.startswith("ECG:"):
                        asyncio.run_coroutine_threadsafe(broadcast_telemetry(line), loop)
        except Exception as e:
            print(f"[Serial RX Error] {e}")
            break

async def main():
    loop = asyncio.get_running_loop()
    
    # Start background serial worker
    t = threading.Thread(target=serial_worker, args=(loop,), daemon=True)
    t.start()

    print("[WebSocket] Starting local server on ws://localhost:8765...")
    async with websockets.serve(ws_handler, "localhost", 8765):
        print("[Ready] Open dashboard.html in your browser to view the live ECG signal!")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Stopped] Exiting bridge.")
