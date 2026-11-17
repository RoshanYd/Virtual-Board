"""Gesture Classifier — pure-function landmark analysis.

Every function in this module operates on ``Landmark`` dataclass lists
and returns deterministic results.  **No** side-effects, no OS calls,
no OpenCV — fully unit-testable with synthetic coordinates.

Gesture detection uses *normalised distance ratios* relative to
``hand_scale`` (wrist↔middle-MCP distance), so thresholds are robust
to camera distance and resolution changes.
"""

from __future__ import annotations

import math
from typing import Optional

from src.models import (
    Finger,
    FINGER_LANDMARKS,
    GestureLabel,
    GestureResult,
    Landmark,
)


# ── Tunable thresholds (as fractions of hand_scale) ────────────────────
PINCH_THRESHOLD: float = 0.30        # tip-to-tip distance / hand_scale
SCROLL_VELOCITY_THRESHOLD: float = 0.02  # normalised units per frame
FINGER_EXTENDED_MARGIN: float = 0.02     # y-axis margin for "extended"


# ── Helper functions ──────────────────────────────────────────────────
def _dist(a: Landmark, b: Landmark) -> float:
    """Euclidean distance between two landmarks (2-D, ignores z)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _dist_3d(a: Landmark, b: Landmark) -> float:
    """Euclidean distance between two landmarks (3-D)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _lm(landmarks: list[Landmark], idx: int) -> Landmark:
    """Retrieve a landmark by its MediaPipe index."""
    return landmarks[idx]


# ── Public pure functions ─────────────────────────────────────────────
def compute_hand_scale(landmarks: list[Landmark]) -> float:
    """Distance from wrist (0) to middle-finger MCP (9).

    This serves as a scale-invariant reference length for all
    threshold comparisons.  Returns 0.001 as a floor to avoid
    division-by-zero.

    Args:
        landmarks: List of 21 ``Landmark`` objects.

    Returns:
        Positive float representing hand scale in normalised coords.
    """
    if len(landmarks) < 21:
        return 0.001
    return max(_dist(_lm(landmarks, 0), _lm(landmarks, 9)), 0.001)


def finger_is_extended(landmarks: list[Landmark], finger: Finger) -> bool:
    """Check whether a finger is extended (straightened out).

    For the four non-thumb fingers the rule is:
        tip.y < pip.y  (tip is *above* PIP joint — remember y=0 is top).

    For the thumb the rule uses the x-axis instead (thumb extends
    laterally), checking that the tip is farther from the palm centre
    than the MCP joint.

    Args:
        landmarks: 21 ``Landmark`` objects.
        finger:    Which finger to check.

    Returns:
        ``True`` if the finger appears extended.
    """
    if len(landmarks) < 21:
        return False

    tip_id, dip_id, pip_id, mcp_id = FINGER_LANDMARKS[finger]
    tip = _lm(landmarks, tip_id)
    pip_ = _lm(landmarks, pip_id)
    mcp = _lm(landmarks, mcp_id)

    if finger == Finger.THUMB:
        # Thumb: tip is farther from wrist in x-axis than MCP
        wrist = _lm(landmarks, 0)
        return abs(tip.x - wrist.x) > abs(mcp.x - wrist.x)

    # Non-thumb: tip is above PIP (lower y value = higher on screen)
    return tip.y < pip_.y - FINGER_EXTENDED_MARGIN


def pinch_distance(
    landmarks: list[Landmark], tip_a: int, tip_b: int
) -> float:
    """Normalised distance between two fingertip landmarks.

    The raw Euclidean distance is divided by ``hand_scale`` so the
    result is comparable across different camera distances.

    Args:
        landmarks: 21 ``Landmark`` objects.
        tip_a:     Landmark index of first tip (e.g. 4 for thumb).
        tip_b:     Landmark index of second tip (e.g. 8 for index).

    Returns:
        Distance as a fraction of hand scale (dimensionless).
    """
    scale = compute_hand_scale(landmarks)
    return _dist(_lm(landmarks, tip_a), _lm(landmarks, tip_b)) / scale


def all_fingers_extended(landmarks: list[Landmark]) -> bool:
    """Return ``True`` if all five fingers are extended (open palm)."""
    return all(
        finger_is_extended(landmarks, f) for f in Finger
    )


def all_fingers_curled(landmarks: list[Landmark]) -> bool:
    """Return ``True`` if all four non-thumb fingers are curled (fist).

    The thumb is excluded because it naturally stays partly visible
    even in a closed fist.
    """
    non_thumb = [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY]
    return all(
        not finger_is_extended(landmarks, f) for f in non_thumb
    )


def count_extended_fingers(landmarks: list[Landmark]) -> int:
    """Return the number of extended fingers (0 to 5)."""
    return sum(1 for f in Finger if finger_is_extended(landmarks, f))


def _scroll_velocity(
    landmarks: list[Landmark],
    prev_landmarks: Optional[list[Landmark]],
    dt: float,
) -> float:
    """Vertical velocity of the index+middle midpoint between frames.

    Positive = moving downward (scroll down), negative = upward.
    Returns 0.0 if either frame has no landmarks.
    """
    if not prev_landmarks or len(prev_landmarks) < 21 or len(landmarks) < 21:
        return 0.0

    curr_mid_y = (_lm(landmarks, 8).y + _lm(landmarks, 12).y) / 2.0
    prev_mid_y = (_lm(prev_landmarks, 8).y + _lm(prev_landmarks, 12).y) / 2.0

    if dt <= 0:
        return 0.0

    return (curr_mid_y - prev_mid_y) / dt


# ── Main classifier ──────────────────────────────────────────────────
def classify_gesture(
    landmarks: list[Landmark],
    prev_landmarks: Optional[list[Landmark]] = None,
    dt: float = 0.033,
) -> GestureResult:
    """Classify the current hand pose into a gesture label.

    Priority order (first match wins):
        1. PAUSE (closed fist)
        2. OPEN_PALM (all fingers extended)
        3. LEFT_CLICK (thumb-index pinch)
        4. RIGHT_CLICK (thumb-middle pinch)
        5. SCROLL (index+middle extended, vertical velocity)
        6. MOVE (index extended, default tracking)
        7. NONE (fallback)

    Args:
        landmarks:      Current frame's 21 ``Landmark`` objects.
        prev_landmarks: Previous frame's landmarks (for velocity).
        dt:             Time delta between frames in seconds.

    Returns:
        A ``GestureResult`` with label, confidence, and metadata.
    """
    if len(landmarks) < 21:
        return GestureResult(label=GestureLabel.NONE, confidence=0.0)

    # ── 1. Closed fist → PAUSE ─────────────────────────────────────
    if all_fingers_curled(landmarks):
        return GestureResult(label=GestureLabel.PAUSE, confidence=0.95)

    # ── 2. Open palm → calibration / OPEN_PALM ────────────────────
    if all_fingers_extended(landmarks):
        return GestureResult(label=GestureLabel.OPEN_PALM, confidence=0.90)

    # ── 3. Thumb-index pinch → LEFT_CLICK ─────────────────────────
    thumb_index_d = pinch_distance(landmarks, 4, 8)
    if thumb_index_d < PINCH_THRESHOLD:
        return GestureResult(
            label=GestureLabel.LEFT_CLICK,
            confidence=max(0.0, 1.0 - thumb_index_d / PINCH_THRESHOLD),
            metadata={"pinch_distance": thumb_index_d},
        )

    # ── 4. Thumb-middle pinch → RIGHT_CLICK ───────────────────────
    thumb_middle_d = pinch_distance(landmarks, 4, 12)
    if thumb_middle_d < PINCH_THRESHOLD:
        return GestureResult(
            label=GestureLabel.RIGHT_CLICK,
            confidence=max(0.0, 1.0 - thumb_middle_d / PINCH_THRESHOLD),
            metadata={"pinch_distance": thumb_middle_d},
        )

    # ── 5. Index + middle extended → SCROLL ───────────────────────
    index_ext = finger_is_extended(landmarks, Finger.INDEX)
    middle_ext = finger_is_extended(landmarks, Finger.MIDDLE)
    ring_curled = not finger_is_extended(landmarks, Finger.RING)
    pinky_curled = not finger_is_extended(landmarks, Finger.PINKY)

    if index_ext and middle_ext and ring_curled and pinky_curled:
        vel = _scroll_velocity(landmarks, prev_landmarks, dt)
        if abs(vel) > SCROLL_VELOCITY_THRESHOLD:
            label = GestureLabel.SCROLL_DOWN if vel > 0 else GestureLabel.SCROLL_UP
            return GestureResult(
                label=label,
                confidence=min(1.0, abs(vel) / (SCROLL_VELOCITY_THRESHOLD * 3)),
                metadata={"velocity": vel},
            )

    # ── 6. Index extended only → MOVE (cursor tracking) ───────────
    if index_ext:
        return GestureResult(label=GestureLabel.MOVE, confidence=0.80)

    # ── 7. Fallback ───────────────────────────────────────────────
    return GestureResult(label=GestureLabel.NONE, confidence=0.0)


def classify_whiteboard_gesture(
    landmarks: list[Landmark],
    prev_landmarks: Optional[list[Landmark]] = None,
    dt: float = 0.033,
) -> GestureResult:
    """Classify the current hand pose into a whiteboard gesture.
    
    Priority order (first match wins):
        1. CLEAR (closed fist)
        2. HOVER (open palm)
        3. UNDO (thumb-index pinch)
        4. SELECT_COLOR (2, 3, or 4 fingers extended)
        5. DRAW (index finger only)
        6. NONE (fallback)
    """
    if len(landmarks) < 21:
        return GestureResult(label=GestureLabel.NONE, confidence=0.0)

    # 1. CLEAR
    if all_fingers_curled(landmarks):
        return GestureResult(label=GestureLabel.CLEAR, confidence=0.95)

    # 2. HOVER
    if all_fingers_extended(landmarks):
        return GestureResult(label=GestureLabel.HOVER, confidence=0.90)

    # 3. UNDO (thumb-index pinch)
    thumb_index_d = pinch_distance(landmarks, 4, 8)
    if thumb_index_d < PINCH_THRESHOLD:
        return GestureResult(
            label=GestureLabel.UNDO,
            confidence=max(0.0, 1.0 - thumb_index_d / PINCH_THRESHOLD),
            metadata={"pinch_distance": thumb_index_d},
        )

    # 4. SELECT_COLOR (2, 3, or 4 non-thumb fingers extended, thumb NOT extended)
    non_thumb_fingers = [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY]
    extended_non_thumb = sum(1 for f in non_thumb_fingers if finger_is_extended(landmarks, f))
    index_ext = finger_is_extended(landmarks, Finger.INDEX)
    thumb_ext = finger_is_extended(landmarks, Finger.THUMB)

    if index_ext and extended_non_thumb in (2, 3, 4) and not thumb_ext:
        # Map finger count to color: 2=red, 3=green, 4=blue
        color_map = {2: "red", 3: "green", 4: "blue"}
        return GestureResult(
            label=GestureLabel.SELECT_COLOR,
            confidence=0.85,
            metadata={"finger_count": extended_non_thumb, "color": color_map[extended_non_thumb]},
        )

    # 5. DRAW (index finger only, EXCLUDING thumb)
    if index_ext and extended_non_thumb == 1:
        return GestureResult(label=GestureLabel.DRAW, confidence=0.80)

    # 6. Fallback
    return GestureResult(label=GestureLabel.NONE, confidence=0.0)
