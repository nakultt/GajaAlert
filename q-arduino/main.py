import asyncio
import websockets
import sys

# Default IP (can be overridden via command line)
ARM_PC_IP = "localhost" 
ARM_PC_PORT = 9000
ARDUINO_PORT = 8000

# Global reference to the outgoing connection to ARM PC
arm_pc_socket = None

def run_vision_inference(payload_bytes: bytes):
    """Stub function for local Vision Inference on Arduino."""
    # OpenCV processing would go here if needed locally
    print(f"[Arduino Vision] Running inference on frame payload of size: {len(payload_bytes)} bytes")

def run_audio_inference(payload_bytes: bytes):
    """Stub function for local Audio Inference on Arduino."""
    print(f"[Arduino Audio] Running inference on audio chunk of size: {len(payload_bytes)} bytes")

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
    
    try:
        async for message in websocket:
            # We received a binary frame from the phone
            if isinstance(message, bytes) and len(message) > 0:
                header = message[0]
                payload = message[1:] # Strip header for local inference
                
                if header == 0x01:
                    run_vision_inference(payload)
                elif header == 0x02:
                    run_audio_inference(payload)
                
                # Instantly push the exact original message (with header) to the ARM PC
                if arm_pc_socket is not None:
                    try:
                        await arm_pc_socket.send(message)
                    except Exception:
                        pass # Ignore send errors, the connection loop will handle reconnects
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
