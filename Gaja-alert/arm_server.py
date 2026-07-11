import asyncio
import websockets
import cv2
import numpy as np
import threading
import queue
import time
import ncnn

# Attempt to load sounddevice (often fails on Windows ARM64 due to missing DLLs)
try:
    import sounddevice as sd
    AUDIO_ENABLED = True
except OSError as e:
    print(f"[Warning] Audio playback disabled: {e}")
    AUDIO_ENABLED = False

# YOLO Configuration
MODEL_PARAM_PATH = "yolo26n_ncnn_model/model.ncnn.param"
MODEL_BIN_PATH = "yolo26n_ncnn_model/model.ncnn.bin"
CONFIDENCE = 0.40
INPUT_W, INPUT_H = 640, 640

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

first_detection_frame = None
vlm_active = False

def run_vlm(image):
    """
    Replace this function with your VLM inference.
    'image' is the first detected frame stored in memory.
    """
    print("VLM Activated")
    print("Image shape:", image.shape)

# Load NCNN Net
net = ncnn.Net()
net.load_param(MODEL_PARAM_PATH)
net.load_model(MODEL_BIN_PATH)

# Queues to pass data from async websocket thread to main processing threads
video_queue = queue.Queue(maxsize=2)
audio_queue = queue.Queue(maxsize=100)

# 1. AUDIO THREAD
def audio_player_thread():
    if not AUDIO_ENABLED:
        print("[Audio] Audio playback disabled.")
        return

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
                # If queue is full, drop the OLDEST frame to make room for this NEWEST frame!
                if video_queue.full():
                    try:
                        video_queue.get_nowait()
                    except queue.Empty:
                        pass
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
                display = frame.copy()
                h, w = frame.shape[:2]

                # ---------- Preprocess ----------
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (INPUT_W, INPUT_H))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                
                # ---------- Inference ----------
                with net.create_extractor() as ex:
                    ex.input("in0", ncnn.Mat(img).clone())
                    _, out0 = ex.extract("out0")
                    
                    arr = np.array(out0).T # shape: (8400, 84)

                    boxes = arr[:, :4]
                    scores = np.max(arr[:, 4:], axis=1)
                    class_ids = np.argmax(arr[:, 4:], axis=1)
                    
                    mask = scores > CONFIDENCE
                    boxes = boxes[mask]
                    scores = scores[mask]
                    class_ids = class_ids[mask]

                    elephant_found = False

                    if len(boxes) > 0:
                        x = boxes[:, 0] - boxes[:, 2] / 2
                        y = boxes[:, 1] - boxes[:, 3] / 2
                        bw = boxes[:, 2]
                        bh = boxes[:, 3]
                        boxes_xywh = np.stack((x, y, bw, bh), axis=1)
                        
                        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), CONFIDENCE, 0.45)

                        for i in indices:
                            idx = i if isinstance(i, int) else i[0]
                            box = boxes_xywh[idx]
                            class_id = class_ids[idx]
                            score = scores[idx]
                            label = CLASSES[class_id]

                            x1 = int(box[0] * w / INPUT_W)
                            y1 = int(box[1] * h / INPUT_H)
                            x2 = int((box[0] + box[2]) * w / INPUT_W)
                            y2 = int((box[1] + box[3]) * h / INPUT_H)

                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                display,
                                f"{label} {score:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                            if label == "elephant":
                                elephant_found = True
                                if first_detection_frame is None:
                                    first_detection_frame = frame.copy()
                                if not vlm_active:
                                    vlm_active = True
                                    run_vlm(first_detection_frame)

                    if not elephant_found:
                        first_detection_frame = None
                        vlm_active = False

                cv2.imshow("Edge Video Stream", display)
                
        except queue.Empty:
            pass
            
        # OpenCV needs waitKey to render the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
