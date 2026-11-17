"""Whiteboard Controller module.

Manages the drawing canvas and executes drawing commands based on gestures.
Includes state debounce logic for actions like clearing the canvas.
"""

import cv2
import numpy as np
from typing import Optional

from src.models import GestureLabel, GestureResult, Landmark
from src.utils import DebounceFilter, EMASmoother


class DrawingCanvas:
    """Manages the in-memory drawing surface."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        # BGR canvas initialized to black
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def draw_line(self, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int):
        cv2.line(self.canvas, p1, p2, color, thickness)

    def clear(self):
        self.canvas.fill(0)

    def get_image(self) -> np.ndarray:
        return self.canvas

    def resize(self, width: int, height: int):
        if self.width != width or self.height != height:
            self.canvas = cv2.resize(self.canvas, (width, height))
            self.width = width
            self.height = height


class WhiteboardController:
    """Translates gestures into whiteboard actions on a canvas."""

    def __init__(
        self,
        smoothing_alpha: float = 0.5,
        clear_debounce_frames: int = 45,  # ~1.5s at 30fps
        undo_debounce_frames: int = 15,   # ~0.5s at 30fps
    ):
        self.canvas = DrawingCanvas()
        self.smoother = EMASmoother(alpha=smoothing_alpha)
        
        # Debouncers for discrete actions
        self.clear_debounce = DebounceFilter(threshold=clear_debounce_frames)
        self.undo_debounce = DebounceFilter(threshold=undo_debounce_frames)
        
        # State
        self.is_drawing = False
        self.draw_grace_frames = 0
        self.color_debounce_frames = 0
        self.last_color_name = ""
        self.last_point: Optional[tuple[int, int]] = None
        self.current_color = (0, 255, 0) # Green default (BGR)
        self.current_thickness = 5

        # Palette definition (BGR format)
        self.palette = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
        }
        
    def _map_to_screen(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """Map normalized coords to canvas pixel coords."""
        x = int(norm_x * self.canvas.width)
        y = int(norm_y * self.canvas.height)
        return x, y

    def process_gesture(self, gesture: GestureResult, landmarks: list[Landmark], frame_shape: tuple[int, int]) -> dict:
        """Process the gesture and update the canvas.
        
        Args:
            gesture: The classified whiteboard gesture
            landmarks: Hand landmarks for position
            frame_shape: (height, width) of the current video frame
            
        Returns:
            dict with UI state updates (e.g. {"action": "cleared", "color": "#00FF00"})
        """
        h, w = frame_shape
        self.canvas.resize(w, h)
        
        ui_state = {}
        
        # Handle CLEAR with strict debounce
        if self.clear_debounce.update(gesture.label) and gesture.label == GestureLabel.CLEAR:
            self.canvas.clear()
            self.is_drawing = False
            self.last_point = None
            ui_state["action"] = "cleared"
            return ui_state

        # Update position for Drawing/Hovering based on index finger
        if len(landmarks) >= 21:
            index_tip = landmarks[8]
            raw_x, raw_y = self._map_to_screen(index_tip.x, index_tip.y)
            smooth_x, smooth_y = self.smoother.update(raw_x, raw_y)
            current_point = (int(smooth_x), int(smooth_y))
        else:
            current_point = None

        if gesture.label == GestureLabel.DRAW and current_point:
            self.draw_grace_frames = 3  # Set grace frames when actively drawing
            if not self.is_drawing:
                self.is_drawing = True
                self.last_point = current_point
            
            # Draw line
            if self.last_point:
                self.canvas.draw_line(self.last_point, current_point, self.current_color, self.current_thickness)
            self.last_point = current_point
            ui_state["action"] = "drawing"
            
        elif gesture.label == GestureLabel.SELECT_COLOR:
            self.is_drawing = False
            self.last_point = None
            
            color_name = gesture.metadata.get("color", "")
            if color_name == self.last_color_name:
                self.color_debounce_frames += 1
            else:
                self.last_color_name = color_name
                self.color_debounce_frames = 1
                
            # Debounce color selection: only update if stable
            if self.color_debounce_frames >= 10:
                if color_name in self.palette:
                    self.current_color = self.palette[color_name]
                    # Convert BGR to Hex for UI
                    b, g, r = self.current_color
                    ui_state["color"] = f"#{r:02x}{g:02x}{b:02x}"
                    ui_state["action"] = "color_selected"
                
        else:
            self.color_debounce_frames = 0
            self.last_color_name = ""
            # HOVER or NONE or UNDO
            # Apply grace frames before fully breaking the draw stroke
            if self.is_drawing and self.draw_grace_frames > 0:
                self.draw_grace_frames -= 1
                if current_point:
                    self.last_point = current_point
            else:
                self.is_drawing = False
                self.last_point = None
            
            self.smoother.reset() # Reset smoother when not actively drawing/hovering near canvas
            
            if self.undo_debounce.update(gesture.label) and gesture.label == GestureLabel.UNDO:
                ui_state["action"] = "undo_triggered"

        return ui_state

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Composite the drawing canvas over the video frame."""
        canvas_img = self.canvas.get_image()
        if canvas_img.shape != frame.shape:
            return frame
            
        # Create a mask of non-black pixels in the canvas
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # Black-out the area of drawing in the frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        
        # Take only region of drawing from canvas
        canvas_fg = cv2.bitwise_and(canvas_img, canvas_img, mask=mask)
        
        # Put them together
        return cv2.add(frame_bg, canvas_fg)
