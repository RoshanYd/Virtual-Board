"""Mouse Controller module.

Maps GestureLabels to OS-level mouse actions using PyAutoGUI.
Includes exponential smoothing (EMA) to reduce cursor jitter,
a debounce filter to prevent accidental rapid clicks, and strict
safety guards (kill-switch and pause gesture).
"""

import time
from typing import Optional

import pyautogui

from src.models import GestureLabel, GestureResult, Landmark
from src.utils import DebounceFilter, EMASmoother


# ── Configuration & Safety ─────────────────────────────────────────────
# PyAutoGUI safety configuration
pyautogui.FAILSAFE = True  # Moving mouse to a corner aborts the program
pyautogui.PAUSE = 0.01     # Tiny delay to ensure thread doesn't lock OS


# ── Main Controller ───────────────────────────────────────────────────
class MouseController:
    """Executes OS-level mouse commands based on detected gestures."""

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        debounce_frames: int = 3,
        active_zone: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85),
    ):
        """
        Args:
            smoothing_alpha: 0.0 to 1.0 (lower = smoother but more lag).
            debounce_frames: Number of frames a click/drag gesture must hold.
            active_zone: (x_min, y_min, x_max, y_max) in normalised coords.
                         Fingers within this rectangle map to the full screen.
        """
        self.screen_w, self.screen_h = pyautogui.size()
        
        self.smoother = EMASmoother(alpha=smoothing_alpha)
        self.click_debounce = DebounceFilter(threshold=debounce_frames)
        self.pause_debounce = DebounceFilter(threshold=30)  # ~1 sec to pause

        # Active zone bounds
        self.x_min, self.y_min, self.x_max, self.y_max = active_zone
        
        # State
        self.paused: bool = False
        self.is_dragging: bool = False
        self.last_executed_click: Optional[GestureLabel] = None

    def _map_to_screen(self, norm_x: float, norm_y: float) -> tuple[float, float]:
        """Map a normalised coordinate from the active zone to screen pixels."""
        # Clamp to active zone
        x = max(self.x_min, min(norm_x, self.x_max))
        y = max(self.y_min, min(norm_y, self.y_max))
        
        # Interpolate
        mapped_x = (x - self.x_min) / (self.x_max - self.x_min)
        mapped_y = (y - self.y_min) / (self.y_max - self.y_min)
        
        screen_x = mapped_x * self.screen_w
        screen_y = mapped_y * self.screen_h
        
        # Prevent PyAutoGUI FailSafe triggering by keeping cursor slightly inside the screen
        screen_x = max(1, min(screen_x, self.screen_w - 2))
        screen_y = max(1, min(screen_y, self.screen_h - 2))
        
        return screen_x, screen_y

    def _get_cursor_position(self, landmarks: list[Landmark]) -> tuple[float, float]:
        """Returns the smoothed screen coordinate for the index fingertip."""
        index_tip = landmarks[8]
        raw_screen_x, raw_screen_y = self._map_to_screen(index_tip.x, index_tip.y)
        return self.smoother.update(raw_screen_x, raw_screen_y)

    def execute(self, gesture_result: GestureResult, landmarks: list[Landmark]) -> None:
        """Main entry point called per-frame to dispatch actions.
        
        Args:
            gesture_result: The detected gesture.
            landmarks: Current frame's 21 landmarks.
        """
        if not landmarks:
            self.smoother.reset()
            return

        label = gesture_result.label

        # ── 1. Check for Pause/Resume Toggle ────────────────────────
        if label == GestureLabel.PAUSE:
            if self.pause_debounce.update(label):
                self.paused = not self.paused
                state_str = "PAUSED" if self.paused else "TRACKING"
                print(f"[INFO] Mode toggled: {state_str}")
                # Reset debounce so it doesn't immediately toggle back
                self.pause_debounce.count = 0
                
                # Release drag if we were dragging
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp()
                    except pyautogui.FailSafeException:
                        pass
                    self.is_dragging = False
            return
        else:
            self.pause_debounce.update(GestureLabel.NONE) # Reset pause counter

        # If paused, ignore all other gestures
        if self.paused:
            self.smoother.reset()
            return

        try:
            # ── 2. Handle Cursor Movement (Always update if available) ──
            # Movement is allowed during MOVE, DRAG, LEFT_CLICK (for aiming before click happens)
            if label in (GestureLabel.MOVE, GestureLabel.DRAG, GestureLabel.LEFT_CLICK, GestureLabel.RIGHT_CLICK):
                screen_x, screen_y = self._get_cursor_position(landmarks)
                
                # Note: We use absolute coordinates because PyAutoGUI's moveTo 
                # uses virtual screen coordinates which are stable.
                pyautogui.moveTo(screen_x, screen_y)
            else:
                self.smoother.reset()

            # ── 3. Handle Actions (Clicks, Drags, Scroll) ───────────────
            
            # Debounce the action gestures
            is_debounced = self.click_debounce.update(label)

            # Handle Dragging
            if label == GestureLabel.DRAG:
                if is_debounced and not self.is_dragging:
                    pyautogui.mouseDown()
                    self.is_dragging = True
                # Movement is handled in section 2
            else:
                # If we were dragging and the gesture changed, release the mouse
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False

            # Handle Clicks
            if is_debounced:
                if label == GestureLabel.LEFT_CLICK and self.last_executed_click != GestureLabel.LEFT_CLICK:
                    pyautogui.click()
                    self.last_executed_click = GestureLabel.LEFT_CLICK
                
                elif label == GestureLabel.RIGHT_CLICK and self.last_executed_click != GestureLabel.RIGHT_CLICK:
                    pyautogui.rightClick()
                    self.last_executed_click = GestureLabel.RIGHT_CLICK
            
            # Handle Scrolling
            # Scrolling uses continuous velocity, so we don't strict-debounce it the same way as clicks
            if label in (GestureLabel.SCROLL_UP, GestureLabel.SCROLL_DOWN):
                velocity = gesture_result.metadata.get("velocity", 0.0)
                # PyAutoGUI scroll units vary by OS. On Windows, 120 is a typical "click".
                scroll_amount = 120 if label == GestureLabel.SCROLL_UP else -120
                # Scale by velocity for dynamic scrolling
                scaled_scroll = int(scroll_amount * (abs(velocity) * 50)) 
                if scaled_scroll != 0:
                    pyautogui.scroll(scaled_scroll)
            
            # Reset last executed click if we moved away from the click gesture
            if label not in (GestureLabel.LEFT_CLICK, GestureLabel.RIGHT_CLICK):
                self.last_executed_click = None

        except pyautogui.FailSafeException:
            print("[WARNING] PyAutoGUI FailSafe triggered! Mouse in corner.")
            self.paused = True
            # Force release drag state in case it happened during drag
            self.is_dragging = False
            return
