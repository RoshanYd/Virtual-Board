"""Main Application Entrypoint.

Flask web server that orchestrates the webcam feed, hand tracking,
gesture classification, and mouse control. Streams annotated frames
to a browser dashboard and emits Server-Sent Events (SSE) for
real-time UI updates.
"""

import base64
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Iterator

# Add project root to path so 'src' module can be resolved regardless of PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

from src.hand_tracker import HandTracker
from src.gesture_classifier import classify_whiteboard_gesture
from src.whiteboard_controller import WhiteboardController
from src.models import GestureLabel, GestureResult

# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Core components (lazy-init in video thread or via get_tracker to avoid import-time camera grab)
tracker = None
tracker_lock = threading.Lock()
controller = WhiteboardController()

def get_tracker() -> HandTracker:
    """Thread-safe accessor for HandTracker instance."""
    global tracker
    with tracker_lock:
        if tracker is None:
            tracker = HandTracker(max_hands=1, fps_window=30)
    return tracker

# Cross-thread communication
sse_queue: queue.Queue = queue.Queue(maxsize=5)
frame_lock = threading.Lock()
current_frame: bytes | None = None

# Global state flags
kill_switch_active = False
video_thread_alive = False
system_paused = False

# Placeholder frame shown before the camera starts or when running in Cloud/Docker mode
_PLACEHOLDER = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(_PLACEHOLDER, "Virtual Board Cloud Mode", (130, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.putText(_PLACEHOLDER, "Click 'Start Browser Webcam' above to draw!", (80, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 1)
_, _buf = cv2.imencode('.jpg', _PLACEHOLDER)
current_frame = _buf.tobytes()


# ---------------------------------------------------------------------------
# Background Video Thread
# ---------------------------------------------------------------------------
def video_loop() -> None:
    """Reads from webcam, processes frames, and updates controllers."""
    global current_frame, kill_switch_active, video_thread_alive, tracker

    prev_landmarks = None
    last_time = time.time()

    # --- Open camera with retries ---
    cap = None
    for attempt in range(3):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"[INFO] Camera opened on attempt {attempt + 1}.")
            break
        print(f"[WARNING] Camera open attempt {attempt + 1}/3 failed, retrying...")
        time.sleep(0.5)

    if cap is None or not cap.isOpened():
        print("[INFO] No hardware camera available on host (running in Container/Cloud mode).")
        print("[INFO] Client-side browser webcam streaming is available via /process_frame.")
        video_thread_alive = False
        try:
            get_tracker()
            print("[INFO] MediaPipe HandTracker pre-warmed and ready for client streams.")
        except Exception as e:
            print(f"[WARNING] Tracker pre-warm failed: {e}")
        return

    video_thread_alive = True
    print("[INFO] Hardware camera streaming active.")

    # --- Initialise tracker ---
    try:
        tracker = get_tracker()
    except Exception as e:
        print(f"[ERROR] Failed to initialise HandTracker: {e}")
        traceback.print_exc()
        cap.release()
        video_thread_alive = False
        return

    consecutive_failures = 0

    while not kill_switch_active:
        try:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("[ERROR] Camera returned 30 consecutive bad frames. Exiting.")
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Flip horizontally for selfie-view
            frame = cv2.flip(frame, 1)

            # 1. Track Hands
            hand_result = tracker.process_frame(frame)
            fps = tracker.get_fps()

            # 2. Classify Gesture
            now = time.time()
            dt = now - last_time
            last_time = now

            if hand_result.detected:
                gesture_result = classify_whiteboard_gesture(
                    hand_result.landmarks, prev_landmarks, dt
                )
            else:
                gesture_result = GestureResult(
                    label=GestureLabel.NONE, confidence=0.0
                )

            prev_landmarks = (
                hand_result.landmarks if hand_result.detected else None
            )

            # 3. Determine System Mode & execute actions
            ui_state = {}
            if system_paused:
                mode = "PAUSED"
            else:
                mode = "DRAWING"
                ui_state = controller.process_gesture(gesture_result, hand_result.landmarks, frame.shape[:2])

            # 4. Publish updates to Web UI via SSE
            disp_gesture = gesture_result.label.name
            if system_paused:
                disp_gesture = "SYSTEM PAUSED"
                
            sse_data = {
                "fps": round(fps, 1),
                "gesture": disp_gesture,
                "mode": mode,
            }
            # Merge canvas updates into SSE data
            sse_data.update(ui_state)

            # Non-blocking put (drop old data if frontend is slow)
            try:
                sse_queue.put_nowait(sse_data)
            except queue.Full:
                try:
                    sse_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sse_queue.put_nowait(sse_data)
                except queue.Full:
                    pass

            # 5. Encode frame for MJPEG stream
            annotated = hand_result.annotated_frame
            annotated = controller.overlay(annotated)

            ok, buffer = cv2.imencode('.jpg', annotated)
            if ok:
                with frame_lock:
                    current_frame = buffer.tobytes()

        except Exception as e:
            # Catch ANY error so the video thread never dies silently
            print(f"[ERROR] Exception in video loop: {e}")
            traceback.print_exc()
            time.sleep(0.1)  # Prevent tight error-loop

    # Cleanup
    cap.release()
    if tracker is not None:
        tracker.release()
    video_thread_alive = False
    print("[INFO] Video loop terminated.")


# ---------------------------------------------------------------------------
# Stream Generators
# ---------------------------------------------------------------------------
def generate_frames() -> Iterator[bytes]:
    """Generator for MJPEG stream."""
    while True:
        with frame_lock:
            frame_bytes = current_frame

        if frame_bytes is None:
            time.sleep(0.1)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        time.sleep(0.033)  # ~30 fps cap


def generate_sse() -> Iterator[str]:
    """Generator for Server-Sent Events (with timeout to avoid dead locks)."""
    while True:
        try:
            data = sse_queue.get(timeout=2.0)
            yield f"data: {json.dumps(data)}\n\n"
        except queue.Empty:
            # Send a keep-alive comment to prevent browser timeout
            yield ": keepalive\n\n"


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/events')
def sse_events():
    return Response(generate_sse(), mimetype='text/event-stream')


@app.route('/stop', methods=['POST'])
def trigger_stop():
    global system_paused
    system_paused = True
    print("[INFO] Emergency STOP triggered. System paused.")
    return jsonify({"status": "paused"})

@app.route('/resume', methods=['POST'])
def trigger_resume():
    global system_paused
    system_paused = False
    print("[INFO] Tracking resumed via UI.")
    return jsonify({"status": "resumed"})

@app.route('/clear', methods=['POST'])
def trigger_clear():
    controller.canvas.clear()
    return jsonify({"status": "cleared"})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame sent from the client's browser webcam.
    
    Accepts JSON payload:
        {
            "image": "data:image/jpeg;base64,...",
            "flip": true  (optional, defaults to true)
        }
    Returns JSON:
        {
            "image": "data:image/jpeg;base64,...",
            "fps": float,
            "gesture": str,
            "mode": str,
            "color": str (optional)
        }
    """
    global system_paused
    try:
        data = request.get_json(silent=True) or {}
        img_data = data.get('image', '')
        if not img_data:
            return jsonify({"error": "No image data provided"}), 400

        if ',' in img_data:
            img_data = img_data.split(',', 1)[1]

        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Flip horizontally for selfie mirror view if requested
        if data.get('flip', True):
            frame = cv2.flip(frame, 1)

        # 1. Track Hands
        trk = get_tracker()
        hand_result = trk.process_frame(frame)
        fps = trk.get_fps()

        # 2. Classify Gesture
        if hand_result.detected:
            gesture_result = classify_whiteboard_gesture(hand_result.landmarks)
        else:
            gesture_result = GestureResult(label=GestureLabel.NONE, confidence=0.0)

        # 3. Canvas action
        ui_state = {}
        if system_paused:
            mode = "PAUSED"
        else:
            mode = "DRAWING"
            ui_state = controller.process_gesture(gesture_result, hand_result.landmarks, frame.shape[:2])

        # 4. Composite drawing over frame
        annotated = hand_result.annotated_frame
        annotated = controller.overlay(annotated)

        # 5. Encode output frame
        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return jsonify({"error": "Failed to encode frame"}), 500

        out_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        disp_gesture = gesture_result.label.name
        if system_paused:
            disp_gesture = "SYSTEM PAUSED"

        response = {
            "image": out_b64,
            "fps": round(fps, 1),
            "gesture": disp_gesture,
            "mode": mode,
        }
        response.update(ui_state)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint for Docker, Kubernetes, and uptime probes."""
    return jsonify({
        "status": "healthy",
        "video_thread_alive": video_thread_alive,
        "kill_switch": kill_switch_active,
        "paused": system_paused,
        "model_loaded": tracker is not None,
    })


# Automatically initialize video service on import (compatible with both Gunicorn and direct Python)
_video_worker_started = False
def start_background_service():
    global _video_worker_started
    if not _video_worker_started:
        _video_worker_started = True
        t = threading.Thread(target=video_loop, daemon=True)
        t.start()

start_background_service()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"[INFO] Starting Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
