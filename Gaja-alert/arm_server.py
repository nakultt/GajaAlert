import asyncio
import websockets
import cv2
import numpy as np
import sounddevice as sd
import threading
import queue

# Queues to pass data from async websocket thread to main processing threads
video_queue = queue.Queue(maxsize=10)
audio_queue = queue.Queue(maxsize=50)

# 1. AUDIO THREAD
def audio_player_thread():
    # Android is sending 16kHz, Mono, 16-bit PCM
    stream = sd.OutputStream(samplerate=16000, channels=1, dtype='int16')
    stream.start()
    print("[Audio] Player started")
    while True:
        try:
            audio_chunk = audio_queue.get()
            stream.write(audio_chunk)
        except Exception as e:
            print(f"Audio playback error: {e}")

# 2. WEBSOCKET ASYNC SERVER
async def handler(websocket):
    print("Client connected.")
    try:
        async for message in websocket:
            if not isinstance(message, bytes) or len(message) == 0:
                continue
                
            header = message[0]
            payload = message[1:]
            
            if header == 0x01:  # Video
                # Drop frames if queue is full to avoid lag
                if not video_queue.full():
                    video_queue.put(payload)
            elif header == 0x02: # Audio
                # Convert bytes to numpy int16 array
                audio_data = np.frombuffer(payload, dtype=np.int16)
                if not audio_queue.full():
                    audio_queue.put(audio_data)
            else:
                print(f"Unknown header: {header}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def run_server():
    print("Edge Server (Python) listening on 0.0.0.0:9000")
    async with websockets.serve(handler, "0.0.0.0", 9000):
        await asyncio.Future()  # run forever

def start_asyncio_server():
    asyncio.run(run_server())

if __name__ == "__main__":
    # Start Audio thread
    threading.Thread(target=audio_player_thread, daemon=True).start()
    
    # Start WebSocket server in a background thread
    threading.Thread(target=start_asyncio_server, daemon=True).start()

    # Main thread handles OpenCV (GUI requires main thread on Windows)
    cv2.namedWindow("Edge Video Stream", cv2.WINDOW_NORMAL)
    print("[Vision] Waiting for frames...")
    
    while True:
        try:
            # Block until a frame is received (timeout allows window to stay responsive)
            payload = video_queue.get(timeout=0.1)
            
            # Decode JPEG
            np_arr = np.frombuffer(payload, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                cv2.imshow("Edge Video Stream", frame)
                
        except queue.Empty:
            pass
            
        # OpenCV needs waitKey to render the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
