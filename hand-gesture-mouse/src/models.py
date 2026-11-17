"""Shared data models for the Virtual Mouse pipeline.

This module contains pure data structures with **zero** external
dependencies (no mediapipe, cv2, or pyautogui imports), so it can
be safely imported in unit tests and CI environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


# ── Hand tracking models ──────────────────────────────────────────────
@dataclass(frozen=True)
class Landmark:
    """A single hand landmark with normalised coordinates.

    Attributes:
        id: MediaPipe landmark index (0-20).
        x:  Horizontal position, 0.0 (left) → 1.0 (right).
        y:  Vertical position, 0.0 (top) → 1.0 (bottom).
        z:  Depth estimate (smaller = closer to camera).
    """

    id: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandResult:
    """Result of processing a single video frame.

    Attributes:
        landmarks:       List of 21 ``Landmark`` objects, or empty if no hand.
        handedness:      ``'Left'`` or ``'Right'`` (from camera's perspective).
        annotated_frame: Copy of the input frame with skeleton overlay.
        detected:        Whether a hand was found in the frame.
    """

    landmarks: list[Landmark] = field(default_factory=list)
    handedness: str = "Unknown"
    annotated_frame: np.ndarray = field(default_factory=lambda: np.empty(0))
    detected: bool = False


# ── Gesture classification models ─────────────────────────────────────
class Finger(Enum):
    """Finger identifiers mapped to MediaPipe landmark indices."""

    THUMB = auto()
    INDEX = auto()
    MIDDLE = auto()
    RING = auto()
    PINKY = auto()


# MediaPipe landmark indices per finger: (TIP, DIP, PIP, MCP)
FINGER_LANDMARKS: dict[Finger, tuple[int, int, int, int]] = {
    Finger.THUMB:  (4, 3, 2, 1),
    Finger.INDEX:  (8, 7, 6, 5),
    Finger.MIDDLE: (12, 11, 10, 9),
    Finger.RING:   (16, 15, 14, 13),
    Finger.PINKY:  (20, 19, 18, 17),
}


class GestureLabel(str, Enum):
    """All recognised gesture labels."""

    MOVE = "MOVE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DRAG = "DRAG"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    PAUSE = "PAUSE"
    OPEN_PALM = "OPEN_PALM"
    DRAW = "DRAW"
    HOVER = "HOVER"
    CLEAR = "CLEAR"
    SELECT_COLOR = "SELECT_COLOR"
    UNDO = "UNDO"
    NONE = "NONE"


@dataclass(frozen=True)
class GestureResult:
    """Output of the gesture classifier.

    Attributes:
        label:      The detected ``GestureLabel``.
        confidence: Detection confidence in [0, 1].
        metadata:   Extra info (e.g. ``scroll_direction``, ``drag_delta``).
    """

    label: GestureLabel = GestureLabel.NONE
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
