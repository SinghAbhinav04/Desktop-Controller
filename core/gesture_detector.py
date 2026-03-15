import time
import math
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Any

@dataclass
class GestureEvent:
    type: str
    confidence: float
    position: Tuple[float, float]

class GestureDetector:
    # Gesture Constants
    CURSOR_MOVE = "CURSOR_MOVE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    DRAG = "DRAG"
    CURSOR_LOCK = "CURSOR_LOCK"
    SWIPE_OUT = "SWIPE_OUT"
    SWIPE_IN = "SWIPE_IN"
    SCROLL = "SCROLL"
    NONE = "NONE"

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.config_path = os.path.join(base_dir, 'config', 'config.json')
        self.config = self._load_config()
        
        # Pull parameters from config (reduced default frames required for instant reaction)
        self.frames_required = self.config.get('gesture_frames_required', 2)
        
        # State tracking
        self.current_gesture = self.NONE
        self.gesture_frame_count = 0
        
        # Debounce/Cooldown tracking (timestamps in seconds)
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0
        self.last_drag_time = 0.0
        self.last_swipe_time = 0.0
        self.last_hands_distance = None
        
        # Double click state
        self.double_click_threshold = 0.4  # 400ms
        
        # Cooldowns (faster for a more responsive double click feeling)
        self.click_cooldown = 0.2 # 200ms
        self.drag_cooldown = 0.4  # 400ms
        self.swipe_cooldown = 0.6 # 600ms
        
        # Hardcoded pinch distance thresholds (normalized coordinates)
        self.pinch_threshold = 0.04

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def _get_distance(self, p1: Any, p2: Any) -> float:
        """Calculates 2D Euclidean distance between two landmarks."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def detect(self, multi_landmarks) -> Optional[GestureEvent]:
        """
        Analyzes hand landmarks, identifies gestures, applies debouncing,
        and returns a Event dataclass if a validated gesture occurred.
        """
        if not multi_landmarks:
            self.current_gesture = self.NONE
            self.gesture_frame_count = 0
            return None

        # Convert to list if it's a single hand from older code usage
        if not isinstance(multi_landmarks, list):
            multi_landmarks = [multi_landmarks]

        # 1. First Pass: Check if ANY hand is a "Lock" hand (fully open palm)
        lock_active = False
        dominant_hand = multi_landmarks[0]
        
        for landmarks in multi_landmarks:
            lm = landmarks.landmark
            wrist = lm[0]
            thumb_up = self._get_distance(lm[4], wrist) > self._get_distance(lm[3], wrist)
            index_up = self._get_distance(lm[8], wrist) > self._get_distance(lm[6], wrist)
            middle_up = self._get_distance(lm[12], wrist) > self._get_distance(lm[10], wrist)
            ring_up = self._get_distance(lm[16], wrist) > self._get_distance(lm[14], wrist)
            pinky_up = self._get_distance(lm[20], wrist) > self._get_distance(lm[18], wrist)
            
            # If 4 fingers extended (ignore thumb, as it's flaky), trigger lock/swipe evaluation
            if index_up and middle_up and ring_up and pinky_up:
                lock_active = True
                
        # 1.5 Evaluate Two-Hand Swiping (Horizontal Only)
        current_time = time.time()
        if len(multi_landmarks) == 2 and lock_active:
             wrist1 = multi_landmarks[0].landmark[0]
             wrist2 = multi_landmarks[1].landmark[0]
             
             # Only care about X-axis distance for swipe to prevent vertical drifting from triggering it
             current_h_distance = abs(wrist1.x - wrist2.x)
             
             if self.last_hands_distance is not None:
                 dist_delta = current_h_distance - self.last_hands_distance
                 # Velocity threshold for swipe (horizontal only)
                 if current_time - self.last_swipe_time > self.swipe_cooldown:
                     if dist_delta > 0.04: # Moving apart on X axis
                          self.last_swipe_time = current_time
                          self.last_hands_distance = current_h_distance
                          return GestureEvent(self.SWIPE_OUT, 1.0, (0.0, 0.0))
                     elif dist_delta < -0.04: # Moving together on X axis
                          self.last_swipe_time = current_time
                          self.last_hands_distance = current_h_distance
                          return GestureEvent(self.SWIPE_IN, 1.0, (0.0, 0.0))
             
             self.last_hands_distance = current_h_distance
        else:
             self.last_hands_distance = None

        # Determine dominant hand properly for single-hand tasks
        if len(multi_landmarks) == 2:
            # If two hands, find the one that ISN'T the open lock hand.
            dominant_hand = multi_landmarks[1] if multi_landmarks[0] == multi_landmarks[0] else multi_landmarks[0] # simplification, we trust lock_active is dominant
            for landmarks in multi_landmarks:
                 lm = landmarks.landmark
                 wrist = lm[0]
                 index_up = self._get_distance(lm[8], wrist) > self._get_distance(lm[6], wrist)
                 middle_up = self._get_distance(lm[12], wrist) > self._get_distance(lm[10], wrist)
                 ring_up = self._get_distance(lm[16], wrist) > self._get_distance(lm[14], wrist)
                 if not (index_up and middle_up and ring_up):
                     dominant_hand = landmarks
                     break

        # State evaluation
        detected_raw = self.NONE
        lm = dominant_hand.landmark
        
        # Key landmark indices based on MediaPipe Hands
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        # Distances
        thumb_index_dist = self._get_distance(lm[THUMB_TIP], lm[INDEX_TIP])
        index_middle_dist = self._get_distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
        
        # Check fingers up (rotation-invariant by comparing tip distance to wrist vs PIP distance to wrist)
        # Wrist is lm[0]. PIP is lm[6], [10], [14], [18]
        wrist = lm[0]
        index_up = self._get_distance(lm[INDEX_TIP], wrist) > self._get_distance(lm[6], wrist)
        middle_up = self._get_distance(lm[MIDDLE_TIP], wrist) > self._get_distance(lm[10], wrist)
        ring_up = self._get_distance(lm[RING_TIP], wrist) > self._get_distance(lm[14], wrist)
        pinky_up = self._get_distance(lm[PINKY_TIP], wrist) > self._get_distance(lm[18], wrist)
        
        if lock_active:
            # Tap-to-click: When locked, check if the tip is closer to the wrist than the PIP joint
            # This is the old, reliable method that worked well.
            index_is_curled = self._get_distance(lm[INDEX_TIP], wrist) < self._get_distance(lm[6], wrist)
            
            if index_is_curled:
                detected_raw = self.LEFT_CLICK
            else:
                detected_raw = self.CURSOR_LOCK
            
        # 1. Right Click: Removed old pinch right click to prevent conflicts.

        # 2. Left Click / Drag: Thumb and Index pinched
        elif thumb_index_dist < self.pinch_threshold and not middle_up and not ring_up and not pinky_up:
            detected_raw = self.LEFT_CLICK # We evaluate if it's a drag later based on time
            
        # 3. Scroll: Index and Middle fingers extended safely apart
        elif index_up and middle_up and not ring_up and not pinky_up:
            detected_raw = self.SCROLL

        # 4. Cursor Move: ONLY Index finger extended
        elif index_up and not middle_up and not ring_up and not pinky_up:
            detected_raw = self.CURSOR_MOVE

        # Frame counting for stability (require N consecutive frames of the same gesture)
        if detected_raw == self.current_gesture and detected_raw != self.NONE:
            self.gesture_frame_count += 1
        else:
            self.current_gesture = detected_raw
            self.gesture_frame_count = 1

        # If we haven't seen it enough times, or if it's NONE, just move the cursor/lock or do nothing
        if self.gesture_frame_count < self.frames_required:
            if detected_raw in [self.CURSOR_MOVE, self.CURSOR_LOCK]:
                 return GestureEvent(detected_raw, 1.0, (lm[INDEX_TIP].x, lm[INDEX_TIP].y))
            return None

        # Once stabilized, evaluate state limits/cooldowns
        current_time = time.time()
        pos = (lm[INDEX_TIP].x, lm[INDEX_TIP].y) # Base position off index tip

        if self.current_gesture == self.RIGHT_CLICK:
            if self.gesture_frame_count == self.frames_required:
                self.last_right_click_time = current_time
                return GestureEvent(self.RIGHT_CLICK, 1.0, pos)

        elif self.current_gesture == self.LEFT_CLICK:
            # Check for hold (DRAG)
            hold_duration = (self.gesture_frame_count - self.frames_required) * (1.0 / 30.0) # Approx
            if hold_duration > 0.3: # 300ms hold to start drag
                # DRAG requires continuous event emission so the cursor can move smoothly in real-time
                return GestureEvent(self.DRAG, 1.0, pos)
            
            # Fire discrete clicks EXACTLY when the gesture stabilizes, not repeatedly
            if self.gesture_frame_count == self.frames_required:
                if current_time - self.last_left_click_time < self.double_click_threshold:
                    # Upgrade to double click
                    self.last_left_click_time = 0 # reset to prevent triple click
                    return GestureEvent(self.DOUBLE_CLICK, 1.0, pos)
                else:
                    # Single left click
                    self.last_left_click_time = current_time
                    return GestureEvent(self.LEFT_CLICK, 1.0, pos)

        elif self.current_gesture == self.CURSOR_MOVE:
             return GestureEvent(self.CURSOR_MOVE, 1.0, pos)

        elif self.current_gesture == self.SCROLL:
             return GestureEvent(self.SCROLL, 1.0, pos)

        elif self.current_gesture == self.CURSOR_LOCK:
             return GestureEvent(self.CURSOR_LOCK, 1.0, pos)

        return None
