#!/usr/bin/env python3
"""Run webcam inference with a YOLO segmentation model.

Prefer using the `ultralytics` package. If unavailable, the script will
attempt to load a TorchScript model via `torch.jit.load`.

Usage:
  python webcam_inference.py --model yolo26n-seg.pt
"""
import argparse
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


def draw_boxes_and_masks(frame, res):
    # Draw boxes
    try:
        if hasattr(res, 'boxes') and res.boxes is not None:
            xyxy = getattr(res.boxes, 'xyxy', None)
            if xyxy is not None:
                boxes = xyxy.cpu().numpy()
                confs = getattr(res.boxes, 'conf', None)
                clss = getattr(res.boxes, 'cls', None)
                confs = confs.cpu().numpy() if confs is not None else None
                clss = clss.cpu().numpy() if clss is not None else None
                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = b.astype(int)
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = ''
                    if clss is not None:
                        label = f"{int(clss[i])}"
                    if confs is not None:
                        label = (label + ' ' if label else '') + f"{confs[i]:.2f}"
                    if label:
                        cv2.putText(frame, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except Exception:
        pass

    # Draw masks (if available)
    try:
        if hasattr(res, 'masks') and res.masks is not None:
            masks = getattr(res.masks, 'data', None)
            if masks is None:
                # some versions expose masks as .masks.masks
                masks = getattr(res.masks, 'masks', None)
            if masks is not None:
                masks = masks.cpu().numpy()
                for m in masks:
                    mask = (m > 0.5).astype(np.uint8) * 255
                    color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
                    colored = np.zeros_like(frame, dtype=np.uint8)
                    colored[mask == 255] = color
                    frame = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)
    except Exception:
        pass

    return frame


def run_ultralytics(model_path, cam_idx, imgsz, conf, device):
    print('Using ultralytics YOLO from package `ultralytics`.')
    model = YOLO(model_path)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {cam_idx}')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
            if len(results) > 0:
                res = results[0]
                frame = draw_boxes_and_masks(frame, res)
            cv2.imshow('webcam', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_torchscript(model_path, cam_idx, imgsz, device):
    if torch is None:
        raise RuntimeError('torch not found; install torch or ultralytics')
    print('Attempting to load TorchScript model with torch.jit.load')
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {cam_idx}')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.resize(frame, (imgsz, imgsz))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
            # Generic handling: just print shape and show raw frame
            cv2.putText(frame, f'Output shape: {getattr(out, "shape", str(type(out)))}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('webcam', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='yolo26n-seg.pt', help='Path to model file (.pt or supported by ultralytics)')
    parser.add_argument('--cam', '-c', type=int, default=0, help='Webcam index')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', default='cpu', help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    # Prefer ultralytics for segmentation-capable models
    if YOLO is not None:
        run_ultralytics(args.model, args.cam, args.imgsz, args.conf, args.device)
        return

    # Fallback: try loading TorchScript
    if torch is not None:
        try:
            run_torchscript(args.model, args.cam, args.imgsz, args.device)
            return
        except Exception as e:
            print('TorchScript fallback failed:', e)

    raise RuntimeError('No supported model runtime available. Install `ultralytics` or provide a TorchScript model.')


if __name__ == '__main__':
    main()
