"""Unit tests for classify_whiteboard_gesture — synthetic landmarks only.

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
    classify_whiteboard_gesture,
    count_extended_fingers,
)


# ════════════════════════════════════════════════════════════════════════
# Helper: build a synthetic 21-landmark hand
# ════════════════════════════════════════════════════════════════════════
# These helpers are exact copies of the ones in test_gesture_classifier.py
# to guarantee identical geometry.

def _make_landmarks(overrides: dict[int, tuple[float, float, float]] | None = None) -> list[Landmark]:
    """Return a baseline *open-palm* hand with optional per-landmark overrides."""
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
    """Return a *closed fist* hand — CLEAR gesture (all non-thumb curled)."""
    return _make_landmarks({
        8:  (0.40, 0.62, 0.0),   # index tip curled
        12: (0.50, 0.60, 0.0),   # middle tip curled
        16: (0.58, 0.62, 0.0),   # ring tip curled
        20: (0.65, 0.67, 0.0),   # pinky tip curled
        4:  (0.44, 0.72, 0.0),   # thumb tucked
    })


def _make_index_only() -> list[Landmark]:
    """Index finger extended, all others curled — DRAW gesture."""
    return _make_landmarks({
        # Keep index extended (default)
        12: (0.50, 0.60, 0.0),   # middle curled
        16: (0.58, 0.62, 0.0),   # ring curled
        20: (0.65, 0.67, 0.0),   # pinky curled
        4:  (0.44, 0.72, 0.0),   # thumb tucked
    })


def _make_pinch() -> list[Landmark]:
    """Thumb tip touches index tip — UNDO gesture."""
    return _make_landmarks({
        4:  (0.40, 0.32, 0.0),   # thumb tip near index tip
        8:  (0.40, 0.30, 0.0),   # index tip
        12: (0.50, 0.60, 0.0),   # middle curled
        16: (0.58, 0.62, 0.0),   # ring curled
        20: (0.65, 0.67, 0.0),   # pinky curled
    })


def _make_two_fingers() -> list[Landmark]:
    """Index + middle extended, rest curled — SELECT_COLOR (2)."""
    return _make_landmarks({
        # index + middle keep default (extended)
        16: (0.58, 0.62, 0.0),   # ring curled
        20: (0.65, 0.67, 0.0),   # pinky curled
        4:  (0.44, 0.72, 0.0),   # thumb tucked
    })


def _make_three_fingers() -> list[Landmark]:
    """Index + middle + ring extended, rest curled — SELECT_COLOR (3)."""
    return _make_landmarks({
        # index + middle + ring keep default (extended)
        20: (0.65, 0.67, 0.0),   # pinky curled
        4:  (0.44, 0.72, 0.0),   # thumb tucked
    })


def _make_four_fingers() -> list[Landmark]:
    """All four non-thumb fingers extended, thumb curled — SELECT_COLOR (4)."""
    return _make_landmarks({
        # index + middle + ring + pinky keep default (extended)
        4:  (0.44, 0.72, 0.0),   # thumb tucked
    })


# ════════════════════════════════════════════════════════════════════════
# Tests: classify_whiteboard_gesture — priority order
# ════════════════════════════════════════════════════════════════════════

class TestClassifyWhiteboardGesture:
    """Integration tests for ``classify_whiteboard_gesture``."""

    # ── Priority 1: CLEAR ─────────────────────────────────────────
    def test_clear_gesture(self) -> None:
        lms = _make_fist()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.CLEAR

    # ── Priority 2: HOVER ─────────────────────────────────────────
    def test_hover_gesture(self) -> None:
        lms = _make_landmarks()  # Open palm — all 5 extended
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.HOVER

    # ── Priority 3: UNDO ──────────────────────────────────────────
    def test_undo_gesture(self) -> None:
        lms = _make_pinch()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.UNDO

    # ── Priority 4: SELECT_COLOR ──────────────────────────────────
    def test_select_color_2(self) -> None:
        lms = _make_two_fingers()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.SELECT_COLOR
        assert result.metadata.get("finger_count") == 2

    def test_select_color_3(self) -> None:
        lms = _make_three_fingers()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.SELECT_COLOR
        assert result.metadata.get("finger_count") == 3

    def test_select_color_4(self) -> None:
        lms = _make_four_fingers()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.SELECT_COLOR
        assert result.metadata.get("finger_count") == 4

    # ── Priority 5: DRAW ──────────────────────────────────────────
    def test_draw_gesture(self) -> None:
        lms = _make_index_only()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.DRAW

    # ── Priority 6: NONE ──────────────────────────────────────────
    def test_none_fallback_too_few_landmarks(self) -> None:
        result = classify_whiteboard_gesture([])
        assert result.label == GestureLabel.NONE

    def test_none_fallback_short_landmarks(self) -> None:
        result = classify_whiteboard_gesture(
            [Landmark(id=i, x=0, y=0, z=0) for i in range(10)]
        )
        assert result.label == GestureLabel.NONE

    # ── Scale invariance ──────────────────────────────────────────
    def test_scale_invariance_undo(self) -> None:
        """UNDO should be detected even if the hand is very small (far away)."""
        lms_1x = _make_pinch()
        label_1x = classify_whiteboard_gesture(lms_1x).label

        # Shrink hand around a centre point (same pattern as existing tests)
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
        label_half = classify_whiteboard_gesture(lms_half).label
        assert label_1x == label_half, (
            f"Gesture changed from {label_1x} to {label_half} under scaling"
        )

    # ── Priority order enforcement ────────────────────────────────
    def test_clear_beats_draw(self) -> None:
        """A fist should be CLEAR, not DRAW."""
        lms = _make_fist()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.CLEAR

    def test_hover_beats_select_color(self) -> None:
        """Open palm (5 fingers) should be HOVER, not SELECT_COLOR."""
        lms = _make_landmarks()
        result = classify_whiteboard_gesture(lms)
        assert result.label == GestureLabel.HOVER
        assert result.label != GestureLabel.SELECT_COLOR

    def test_confidence_is_positive_for_detected_gestures(self) -> None:
        lms = _make_fist()
        result = classify_whiteboard_gesture(lms)
        assert result.confidence > 0.0

    def test_undo_has_pinch_distance_metadata(self) -> None:
        lms = _make_pinch()
        result = classify_whiteboard_gesture(lms)
        assert "pinch_distance" in result.metadata
        assert result.metadata["pinch_distance"] >= 0.0
