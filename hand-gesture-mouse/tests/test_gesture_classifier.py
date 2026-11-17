"""Unit tests for gesture_classifier — synthetic landmarks only.

Every test constructs a ``list[Landmark]`` with known coordinates and
asserts the expected ``GestureLabel``.  No webcam, no OpenCV, no
side-effects — safe for CI.
"""

from __future__ import annotations

import copy
import pytest

from src.models import (
    Landmark,
    GestureLabel,
    GestureResult,
    Finger,
)
from src.gesture_classifier import (
    compute_hand_scale,
    finger_is_extended,
    pinch_distance,
    all_fingers_extended,
    all_fingers_curled,
    count_extended_fingers,
    classify_gesture,
)


# ════════════════════════════════════════════════════════════════════════
# Helper: build a synthetic 21-landmark hand
# ════════════════════════════════════════════════════════════════════════

def _make_landmarks(overrides: dict[int, tuple[float, float, float]] | None = None) -> list[Landmark]:
    """Return a baseline *open-palm* hand with optional per-landmark overrides.

    The default hand is roughly:
      - Wrist at (0.5, 0.9, 0)
      - Palm centre at (0.5, 0.7, 0)
      - Finger tips higher (lower y) than their PIPs
      - Thumb tip extended to the left

    Override specific landmarks by passing ``{id: (x, y, z), ...}``.
    """
    # Baseline open palm — all fingers extended (tip.y < pip.y)
    base: dict[int, tuple[float, float, float]] = {
        # Wrist
        0:  (0.50, 0.90, 0.0),
        # Thumb: CMC, MCP, IP, TIP
        1:  (0.42, 0.80, 0.0),
        2:  (0.35, 0.72, 0.0),
        3:  (0.28, 0.64, 0.0),
        4:  (0.22, 0.56, 0.0),   # thumb tip — extended left
        # Index: MCP, PIP, DIP, TIP
        5:  (0.40, 0.62, 0.0),
        6:  (0.40, 0.50, 0.0),
        7:  (0.40, 0.40, 0.0),
        8:  (0.40, 0.30, 0.0),   # index tip — extended up
        # Middle: MCP, PIP, DIP, TIP
        9:  (0.50, 0.60, 0.0),
        10: (0.50, 0.48, 0.0),
        11: (0.50, 0.38, 0.0),
        12: (0.50, 0.28, 0.0),   # middle tip — extended up
        # Ring: MCP, PIP, DIP, TIP
        13: (0.58, 0.62, 0.0),
        14: (0.58, 0.50, 0.0),
        15: (0.58, 0.40, 0.0),
        16: (0.58, 0.30, 0.0),   # ring tip — extended up
        # Pinky: MCP, PIP, DIP, TIP
        17: (0.65, 0.65, 0.0),
        18: (0.65, 0.55, 0.0),
        19: (0.65, 0.47, 0.0),
        20: (0.65, 0.39, 0.0),   # pinky tip — extended up
    }

    if overrides:
        base.update(overrides)

    return [Landmark(id=i, x=base[i][0], y=base[i][1], z=base[i][2]) for i in range(21)]


def _make_fist() -> list[Landmark]:
    """Return a *closed fist* hand — all non-thumb tips below their PIPs."""
    return _make_landmarks({
        # Curl index: tip below PIP
        8:  (0.40, 0.62, 0.0),   # tip at/below MCP level
        # Curl middle
        12: (0.50, 0.60, 0.0),
        # Curl ring
        16: (0.58, 0.62, 0.0),
        # Curl pinky
        20: (0.65, 0.67, 0.0),
        # Thumb tucked in (closer to wrist in x)
        4:  (0.44, 0.72, 0.0),
    })


def _make_index_only() -> list[Landmark]:
    """Index finger extended, all others curled — MOVE gesture."""
    return _make_landmarks({
        # Keep index extended (default)
        # Curl middle
        12: (0.50, 0.60, 0.0),
        # Curl ring
        16: (0.58, 0.62, 0.0),
        # Curl pinky
        20: (0.65, 0.67, 0.0),
        # Thumb tucked
        4:  (0.44, 0.72, 0.0),
    })


def _make_left_click_pinch() -> list[Landmark]:
    """Thumb tip touches index tip — LEFT_CLICK."""
    return _make_landmarks({
        4:  (0.40, 0.32, 0.0),   # thumb tip at same position as index tip
        8:  (0.40, 0.30, 0.0),   # index tip
        # Curl other fingers
        12: (0.50, 0.60, 0.0),
        16: (0.58, 0.62, 0.0),
        20: (0.65, 0.67, 0.0),
    })


def _make_right_click_pinch() -> list[Landmark]:
    """Thumb tip touches middle tip — RIGHT_CLICK."""
    return _make_landmarks({
        4:  (0.50, 0.30, 0.0),   # thumb tip near middle tip
        12: (0.50, 0.28, 0.0),   # middle tip
        # Keep index extended so it doesn't trigger left-click first
        8:  (0.40, 0.30, 0.0),   # index tip extended
        # Curl ring + pinky
        16: (0.58, 0.62, 0.0),
        20: (0.65, 0.67, 0.0),
    })


def _make_scroll_pose() -> list[Landmark]:
    """Index + middle extended, ring + pinky curled — scroll-ready."""
    return _make_landmarks({
        # Index + middle extended (default positions are fine)
        # Curl ring
        16: (0.58, 0.62, 0.0),
        # Curl pinky
        20: (0.65, 0.67, 0.0),
        # Thumb tucked
        4:  (0.44, 0.72, 0.0),
    })


# ════════════════════════════════════════════════════════════════════════
# Tests: compute_hand_scale
# ════════════════════════════════════════════════════════════════════════

class TestHandScale:
    """Tests for ``compute_hand_scale``."""

    def test_positive_scale(self) -> None:
        lms = _make_landmarks()
        scale = compute_hand_scale(lms)
        assert scale > 0.0

    def test_scale_is_wrist_to_middle_mcp(self) -> None:
        lms = _make_landmarks()
        import math
        expected = math.sqrt(
            (lms[0].x - lms[9].x) ** 2 + (lms[0].y - lms[9].y) ** 2
        )
        assert abs(compute_hand_scale(lms) - expected) < 1e-6

    def test_empty_landmarks(self) -> None:
        assert compute_hand_scale([]) == 0.001


# ════════════════════════════════════════════════════════════════════════
# Tests: finger_is_extended
# ════════════════════════════════════════════════════════════════════════

class TestFingerExtended:
    """Tests for ``finger_is_extended``."""

    def test_all_extended_in_open_palm(self) -> None:
        lms = _make_landmarks()
        for finger in Finger:
            assert finger_is_extended(lms, finger), f"{finger.name} should be extended"

    def test_all_curled_in_fist(self) -> None:
        lms = _make_fist()
        for finger in [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY]:
            assert not finger_is_extended(lms, finger), f"{finger.name} should be curled"

    def test_index_only_extended(self) -> None:
        lms = _make_index_only()
        assert finger_is_extended(lms, Finger.INDEX)
        assert not finger_is_extended(lms, Finger.MIDDLE)
        assert not finger_is_extended(lms, Finger.RING)
        assert not finger_is_extended(lms, Finger.PINKY)


# ════════════════════════════════════════════════════════════════════════
# Tests: pinch_distance
# ════════════════════════════════════════════════════════════════════════

class TestPinchDistance:
    """Tests for ``pinch_distance``."""

    def test_pinch_close(self) -> None:
        lms = _make_left_click_pinch()
        d = pinch_distance(lms, 4, 8)
        assert d < 0.30, f"Pinch distance {d} should be < 0.30 for a pinch"

    def test_no_pinch_when_far(self) -> None:
        lms = _make_landmarks()  # open palm, tips far apart
        d = pinch_distance(lms, 4, 8)
        assert d > 0.30, f"Distance {d} should be > 0.30 when fingers apart"


# ════════════════════════════════════════════════════════════════════════
# Tests: count_extended_fingers
# ════════════════════════════════════════════════════════════════════════

class TestCountExtendedFingers:
    """Tests for ``count_extended_fingers``."""

    def test_zero_fingers(self) -> None:
        lms = _make_fist()
        assert count_extended_fingers(lms) == 0

    def test_one_finger(self) -> None:
        lms = _make_index_only()
        assert count_extended_fingers(lms) == 1

    def test_two_fingers(self) -> None:
        lms = _make_scroll_pose()
        assert count_extended_fingers(lms) == 2

    def test_three_fingers(self) -> None:
        # Index, middle, ring extended. Pinky & thumb curled.
        lms = _make_landmarks({
            20: (0.65, 0.67, 0.0), # Curl pinky
            4:  (0.44, 0.72, 0.0), # Tuck thumb
        })
        assert count_extended_fingers(lms) == 3

    def test_four_fingers(self) -> None:
        # All extended except thumb.
        lms = _make_landmarks({
            4:  (0.44, 0.72, 0.0), # Tuck thumb
        })
        assert count_extended_fingers(lms) == 4

    def test_five_fingers(self) -> None:
        lms = _make_landmarks() # Default is all 5 extended
        assert count_extended_fingers(lms) == 5


# ════════════════════════════════════════════════════════════════════════
# Tests: classify_gesture — the main classifier
# ════════════════════════════════════════════════════════════════════════

class TestClassifyGesture:
    """Integration tests for the top-level ``classify_gesture`` function."""

    def test_left_click_pinch(self) -> None:
        lms = _make_left_click_pinch()
        result = classify_gesture(lms)
        assert result.label == GestureLabel.LEFT_CLICK

    def test_right_click_pinch(self) -> None:
        lms = _make_right_click_pinch()
        result = classify_gesture(lms)
        assert result.label == GestureLabel.RIGHT_CLICK

    def test_no_pinch_when_far(self) -> None:
        """Fingers apart → should NOT be a click."""
        lms = _make_index_only()
        result = classify_gesture(lms)
        assert result.label != GestureLabel.LEFT_CLICK
        assert result.label != GestureLabel.RIGHT_CLICK

    def test_closed_fist_pause(self) -> None:
        lms = _make_fist()
        result = classify_gesture(lms)
        assert result.label == GestureLabel.PAUSE

    def test_open_palm(self) -> None:
        lms = _make_landmarks()  # default is open palm
        result = classify_gesture(lms)
        assert result.label == GestureLabel.OPEN_PALM

    def test_scroll_up(self) -> None:
        """Index+middle extended, upward velocity → SCROLL_UP."""
        prev = _make_scroll_pose()
        curr_overrides = {
            8:  (0.40, 0.20, 0.0),   # index tip moved UP
            12: (0.50, 0.18, 0.0),   # middle tip moved UP
            16: (0.58, 0.62, 0.0),
            20: (0.65, 0.67, 0.0),
            4:  (0.44, 0.72, 0.0),
        }
        curr = _make_landmarks(curr_overrides)
        result = classify_gesture(curr, prev_landmarks=prev, dt=0.033)
        assert result.label == GestureLabel.SCROLL_UP

    def test_scroll_down(self) -> None:
        """Index+middle extended, downward velocity → SCROLL_DOWN."""
        prev = _make_scroll_pose()
        curr_overrides = {
            8:  (0.40, 0.45, 0.0),   # index tip moved DOWN
            12: (0.50, 0.43, 0.0),   # middle tip moved DOWN
            16: (0.58, 0.62, 0.0),
            20: (0.65, 0.67, 0.0),
            4:  (0.44, 0.72, 0.0),
        }
        curr = _make_landmarks(curr_overrides)
        result = classify_gesture(curr, prev_landmarks=prev, dt=0.033)
        assert result.label == GestureLabel.SCROLL_DOWN

    def test_move_gesture(self) -> None:
        """Only index extended → MOVE."""
        lms = _make_index_only()
        result = classify_gesture(lms)
        assert result.label == GestureLabel.MOVE

    def test_hand_scale_invariance(self) -> None:
        """Same gesture at 2× scale → same label."""
        lms_1x = _make_left_click_pinch()
        label_1x = classify_gesture(lms_1x).label

        # Scale all coordinates by 0.5 (shrink hand — simulating farther distance)
        centre_x, centre_y = 0.5, 0.6
        lms_half = [
            Landmark(
                id=lm.id,
                x=centre_x + (lm.x - centre_x) * 0.5,
                y=centre_y + (lm.y - centre_y) * 0.5,
                z=lm.z,
            )
            for lm in lms_1x
        ]
        label_half = classify_gesture(lms_half).label
        assert label_1x == label_half, (
            f"Gesture changed from {label_1x} to {label_half} under scaling"
        )

    def test_empty_landmarks_returns_none(self) -> None:
        result = classify_gesture([])
        assert result.label == GestureLabel.NONE

    def test_confidence_is_positive_for_detected_gestures(self) -> None:
        lms = _make_fist()
        result = classify_gesture(lms)
        assert result.confidence > 0.0

    def test_left_click_has_pinch_distance_metadata(self) -> None:
        lms = _make_left_click_pinch()
        result = classify_gesture(lms)
        assert "pinch_distance" in result.metadata
        assert result.metadata["pinch_distance"] >= 0.0
