import asyncio
import datetime
import json
import sys

import websockets

from audio_classifier import ElephantAudioClassifier

# Default IP (can be overridden via command line)
ARM_PC_IP = "localhost"
ARM_PC_PORT = 9000
ARDUINO_PORT = 8000

# Global reference to the outgoing connection to ARM PC
arm_pc_socket = None

print("[Arduino Audio] Loading elephant-sound classifier (YAMNet + XGBoost)...")
audio_classifier = ElephantAudioClassifier()
print("[Arduino Audio] Classifier ready.")

def run_vision_inference(payload_bytes: bytes):
    """Stub function for local Vision Inference on Arduino."""
    # OpenCV processing would go here if needed locally
    print(f"[Arduino Vision] Running inference on frame payload of size: {len(payload_bytes)} bytes")

async def run_audio_inference(payload_bytes: bytes):
    """Feeds mic PCM from the phone into the local elephant-sound classifier;
    on a trigger, reports a compact 0x04 event to the ARM PC instead of
    forwarding raw audio (arm_server.py's sensor_handler already expects
    this exact schema)."""
    # feed_bytes runs blocking ONNX/XGBoost inference on hops; offload to a
    # thread so it never stalls the event loop (and the 0ms-latency video
    # relay above it).
    event = await asyncio.to_thread(audio_classifier.feed_bytes, payload_bytes)
    if event is None:
        return
    print(f"[Arduino Audio] ELEPHANT sound detected (confidence={event.confidence:.2f})")
    if arm_pc_socket is None:
        print("[Arduino Audio] No ARM PC connection -- trigger not reported.")
        return
    when = datetime.datetime.fromtimestamp(event.timestamp).astimezone().isoformat(timespec="seconds")
    body = json.dumps({
        "event": "audio_elephant",
        "confidence": round(event.confidence, 3),
        "sound_type": "elephant_vocalization",
        "timestamp": when,
    }).encode()
    try:
        await asyncio.wait_for(arm_pc_socket.send(b"\x04" + body), timeout=0.5)
    except Exception as e:
        print(f"[Arduino Audio] Failed to report trigger to ARM PC: {e}")

async def forward_to_arm_pc():
    global arm_pc_socket
    uri = f"ws://{ARM_PC_IP}:{ARM_PC_PORT}"
    
    print(f"[Arduino Network] Attempting to connect to ARM PC at {uri}...")
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print(f"[Arduino Network] Connected successfully to ARM PC!")
                arm_pc_socket = websocket
                # Keep connection alive until it drops
                await asyncio.Future()
        except Exception as e:
            arm_pc_socket = None
            print(f"[Arduino Network] Failed to connect to ARM PC. Retrying in 2 seconds...")
            await asyncio.sleep(2)

async def handle_mobile_client(websocket):
    global arm_pc_socket
    print("[Arduino Server] Mobile Phone connected!")
    
    current_video_task = None
    
    try:
        async for message in websocket:
            # We received a binary frame from the phone
            if isinstance(message, bytes) and len(message) > 0:
                header = message[0]
                payload = message[1:] # Strip header for local inference
                
                if header == 0x01:
                    run_vision_inference(payload)
                elif header == 0x02:
                    await run_audio_inference(payload)
                
                # Instantly push the exact original message (with header) to the ARM PC
                if arm_pc_socket is not None:
                    try:
                        if header == 0x01:
                            # VIDEO: Drop frame if previous video frame is still sending!
                            if current_video_task is None or current_video_task.done():
                                current_video_task = asyncio.create_task(
                                    asyncio.wait_for(arm_pc_socket.send(message), timeout=0.2)
                                )
                            else:
                                pass # Drop frame to maintain 0ms latency
                    except Exception:
                        pass # Ignore send timeouts, connection loop handles reconnects
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print("[Arduino Server] Mobile Phone disconnected.")

async def main():
    print("=======================================")
    print(f" Arduino Headless Python Server Started")
    print(f" Target ARM PC IP: {ARM_PC_IP}")
    print("=======================================\n")
    
    # 1. Start the background task to constantly maintain connection to ARM PC
    asyncio.create_task(forward_to_arm_pc())
    
    # 2. Start the WebSocket server to accept incoming video/audio from the Mobile Phone
    print(f"[Arduino Server] Listening for Mobile Phone on 0.0.0.0:{ARDUINO_PORT}")
    async with websockets.serve(handle_mobile_client, "0.0.0.0", ARDUINO_PORT):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # If the user provides an IP address via command line, use it!
    if len(sys.argv) > 1:
        ARM_PC_IP = sys.argv[1]
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
