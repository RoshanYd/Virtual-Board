"""Hand Tracker module — MediaPipe Tasks API wrapper.

Provides a clean, reusable interface around MediaPipe's Hand Landmarker
(Tasks API, v0.10.14+).  Returns normalised (0-1) landmarks per frame
so downstream consumers are resolution-independent.

Uses the ``HandLandmarker`` from ``mediapipe.tasks.vision`` instead of
the legacy ``mp.solutions.hands`` (removed in 0.10.14+ for Python 3.13).
"""

from __future__ import annotations

import os
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.models import HandResult, Landmark

# ── Constants ──────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"

# MediaPipe hand connections for drawing (pairs of landmark indices)
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle  (was 5,9 - corrected)
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring    (was 9,13)
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky   (was 13,17)
    (5, 9), (9, 13), (13, 17),             # Palm cross-connections
]


def _ensure_model() -> str:
    """Download the hand-landmarker model if it doesn't exist locally.

    Returns:
        Absolute path to the ``.task`` model file.
    """
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)

    print(f"[INFO] Downloading hand_landmarker model to {_MODEL_PATH} ...")
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print("[INFO] Model downloaded successfully.")
    return str(_MODEL_PATH)


def _draw_landmarks(
    frame: np.ndarray,
    landmarks: list[Landmark],
    h: int,
    w: int,
) -> np.ndarray:
    """Draw hand skeleton on *frame* using OpenCV primitives.

    Args:
        frame:     BGR image to annotate (modified in-place).
        landmarks: 21 ``Landmark`` objects with normalised coords.
        h:         Frame height in pixels.
        w:         Frame width in pixels.

    Returns:
        The annotated frame.
    """
    # Convert normalised coords to pixel coords
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw connections
    for i, j in _HAND_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], (0, 255, 128), 2, cv2.LINE_AA)

    # Draw landmark dots
    for idx, pt in enumerate(pts):
        colour = (0, 0, 255) if idx in (4, 8, 12, 16, 20) else (255, 128, 0)
        cv2.circle(frame, pt, 5, colour, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ── Hand Tracker ───────────────────────────────────────────────────────
class HandTracker:
    """Wraps MediaPipe Hand Landmarker for frame-by-frame detection.

    Parameters:
        max_hands:      Maximum number of hands to detect (default 1).
        detection_conf: Minimum detection confidence (0-1).
        tracking_conf:  Minimum tracking confidence (0-1).
        fps_window:     Number of frames used for the rolling FPS average.
    """

    def __init__(
        self,
        max_hands: int = 1,
        detection_conf: float = 0.5,
        tracking_conf: float = 0.5,
        fps_window: int = 30,
    ) -> None:
        model_path = _ensure_model()

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self._fps_window = fps_window
        self._frame_times: deque[float] = deque(maxlen=fps_window)
        self._last_time: float = time.perf_counter()
        self._frame_count: int = 0

    # ── public API ─────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> HandResult:
        """Detect hand landmarks in *frame* and return a ``HandResult``.

        Args:
            frame: BGR image from OpenCV (``np.ndarray``).

        Returns:
            A ``HandResult`` containing landmarks, handedness, and an
            annotated copy of the frame.
        """
        # Update FPS bookkeeping
        now = time.perf_counter()
        self._frame_times.append(now - self._last_time)
        self._last_time = now
        self._frame_count += 1

        h, w = frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in milliseconds (must be monotonically increasing)
        timestamp_ms = int(self._frame_count * 33)  # ~30 FPS assumption
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated = frame.copy()

        if not results.hand_landmarks:
            return HandResult(annotated_frame=annotated, detected=False)

        # Take the first detected hand
        hand_lms = results.hand_landmarks[0]
        handedness_label = (
            results.handedness[0][0].category_name
            if results.handedness
            else "Unknown"
        )

        # Convert to our Landmark dataclass list
        landmarks = [
            Landmark(id=i, x=lm.x, y=lm.y, z=lm.z)
            for i, lm in enumerate(hand_lms)
        ]

        # Draw skeleton on annotated frame
        _draw_landmarks(annotated, landmarks, h, w)

        return HandResult(
            landmarks=landmarks,
            handedness=handedness_label,
            annotated_frame=annotated,
            detected=True,
        )

    def get_fps(self) -> float:
        """Return the rolling-average frames-per-second.

        Computed over the last ``fps_window`` frames.  Returns 0.0 if
        fewer than 2 frames have been processed.
        """
        if len(self._frame_times) < 2:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()


# ── Smoke-test entry point (Phase 1 verification) ─────────────────────
def _smoke_test() -> None:
    """Quick verification: open webcam, print landmark count per frame."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam — check camera permissions.")
        return

    tracker = HandTracker()
    print("[INFO] Smoke test running — press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = tracker.process_frame(frame)
            n = len(result.landmarks)
            fps = tracker.get_fps()
            print(f"Landmarks detected: {n}  |  FPS: {fps:.1f}", end="\r")

            cv2.imshow("Hand Tracker — Smoke Test", result.annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Smoke test finished.")


if __name__ == "__main__":
    _smoke_test()
