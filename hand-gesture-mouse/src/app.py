"""Main Application Entrypoint.

Flask web server that orchestrates the webcam feed, hand tracking,
gesture classification, and mouse control. Streams annotated frames
to a browser dashboard and emits Server-Sent Events (SSE) for
real-time UI updates.
"""

import json
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
from flask import Flask, render_template, Response, jsonify

from src.hand_tracker import HandTracker
from src.gesture_classifier import classify_whiteboard_gesture
from src.whiteboard_controller import WhiteboardController
from src.models import GestureLabel, GestureResult

# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Core components (lazy-init in video thread to avoid import-time camera grab)
tracker = None
controller = WhiteboardController()

# Cross-thread communication
sse_queue: queue.Queue = queue.Queue(maxsize=5)
frame_lock = threading.Lock()
current_frame: bytes | None = None

# Global state flags
kill_switch_active = False
video_thread_alive = True
system_paused = False

# Placeholder frame shown before the camera starts
_PLACEHOLDER = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(_PLACEHOLDER, "Starting camera...", (140, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
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
    for attempt in range(5):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"[INFO] Camera opened on attempt {attempt + 1}.")
            break
        print(f"[WARNING] Camera open attempt {attempt + 1}/5 failed, retrying...")
        time.sleep(1)

    if cap is None or not cap.isOpened():
        print("[ERROR] Cannot open webcam after 5 attempts.")
        video_thread_alive = False
        return

    # --- Initialise tracker ---
    try:
        tracker = HandTracker(max_hands=1, fps_window=30)
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

@app.route('/health')
def health():
    """Quick health-check endpoint for debugging."""
    return jsonify({
        "video_thread_alive": video_thread_alive,
        "kill_switch": kill_switch_active,
        "paused": system_paused,
    })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting video thread...")
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()

    print("[INFO] Starting Flask server on http://127.0.0.1:5000")
    # Turn off reloader to prevent double-starting the camera
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
