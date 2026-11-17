"""Shared utilities for smoothing and debouncing."""

from typing import Optional

from src.models import GestureLabel


class DebounceFilter:
    """Requires a gesture to be seen N consecutive frames before firing."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.current_gesture: GestureLabel = GestureLabel.NONE
        self.count: int = 0

    def update(self, new_gesture: GestureLabel) -> bool:
        """Returns True if the gesture has passed the debounce threshold."""
        if new_gesture == self.current_gesture:
            self.count += 1
        else:
            self.current_gesture = new_gesture
            self.count = 1

        return self.count >= self.threshold


class EMASmoother:
    """Exponential Moving Average filter for 2D points to reduce jitter."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y
            return x, y

        smoothed_x = self.alpha * x + (1 - self.alpha) * self.last_x
        smoothed_y = self.alpha * y + (1 - self.alpha) * self.last_y

        self.last_x, self.last_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y

    def reset(self):
        self.last_x, self.last_y = None, None
