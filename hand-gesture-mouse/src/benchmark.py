"""Evaluation & Benchmarking Script.

Runs a headless timed session processing webcam frames to evaluate
system performance. Computes average detection latency, sustained FPS,
and estimated false-positive rates for gestures.
Outputs the results as a Markdown table.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.hand_tracker import HandTracker
from src.gesture_classifier import classify_whiteboard_gesture
from src.models import GestureLabel

def run_benchmark(duration_seconds: int = 15) -> None:
    print(f"[INFO] Starting benchmark for {duration_seconds} seconds...")
    print("[INFO] Please move your hand and perform various gestures during this time.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    tracker = HandTracker(max_hands=1)
    
    start_time = time.time()
    frames_processed = 0
    total_latency_ms = 0.0
    
    gesture_counts = defaultdict(int)
    
    # For false positive estimation (flickering gestures)
    # If a gesture lasts only 1 frame before changing again, it's likely a false positive
    last_gesture: Optional[GestureLabel] = None
    gesture_run_length = 0
    false_positives = 0
    total_gesture_events = 0

    prev_landmarks = None
    last_frame_time = time.time()

    while (time.time() - start_time) < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now

        # Measure latency for tracking + classification
        t0 = time.perf_counter()
        
        hand_result = tracker.process_frame(frame)
        if hand_result.detected:
            gesture_result = classify_whiteboard_gesture(hand_result.landmarks, prev_landmarks, dt)
            label = gesture_result.label
            prev_landmarks = hand_result.landmarks
        else:
            label = GestureLabel.NONE
            prev_landmarks = None
            
        t1 = time.perf_counter()
        
        total_latency_ms += (t1 - t0) * 1000
        frames_processed += 1
        
        # Track gesture statistics
        if label != GestureLabel.NONE and label != GestureLabel.HOVER:
            gesture_counts[label] += 1
            
        # False positive logic
        if label != last_gesture:
            if last_gesture is not None and last_gesture != GestureLabel.NONE and last_gesture != GestureLabel.HOVER:
                total_gesture_events += 1
                if gesture_run_length < 2:  # 1 frame duration is highly likely a FP
                    false_positives += 1
            last_gesture = label
            gesture_run_length = 1
        else:
            gesture_run_length += 1
            
        # Optional: visual feedback
        # cv2.imshow("Benchmarking...", hand_result.annotated_frame)
        # cv2.waitKey(1)

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    
    # Compute final metrics
    actual_duration = time.time() - start_time
    avg_latency = total_latency_ms / frames_processed if frames_processed > 0 else 0
    sustained_fps = frames_processed / actual_duration if actual_duration > 0 else 0
    fp_rate = (false_positives / total_gesture_events * 100) if total_gesture_events > 0 else 0.0

    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    markdown_output = (
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Avg Detection Latency | {avg_latency:.2f} ms |\n"
        f"| Sustained FPS | {sustained_fps:.1f} |\n"
        f"| Draw Detections | {gesture_counts[GestureLabel.DRAW]} |\n"
        f"| Select Color Detections | {gesture_counts[GestureLabel.SELECT_COLOR]} |\n"
        f"| Undo Detections | {gesture_counts[GestureLabel.UNDO]} |\n"
        f"| Clear Detections | {gesture_counts[GestureLabel.CLEAR]} |\n"
        f"| Estimated False Positive Rate | {fp_rate:.1f}% |\n"
    )
    
    print(markdown_output)
    print("="*50)

if __name__ == "__main__":
    run_benchmark(duration_seconds=15)  # 15s is good for a quick run
