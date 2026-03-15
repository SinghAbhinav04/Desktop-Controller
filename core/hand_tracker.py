import cv2
import json
import os
import mediapipe as mp
from typing import Any, Optional

class HandTracker:
    def __init__(self):
        # Resolve config path statically relative to this file
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.config_path = os.path.join(base_dir, 'config', 'config.json')
        self.config = self._load_config()

        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Store current config
        self.min_conf = self.config.get('confidence_threshold', 0.8)
        self.current_complexity = 1
        
        # Initialize hands with static_image_mode=False for video stream performance
        self._init_mediapipe(self.current_complexity)

    def _init_mediapipe(self, complexity: int):
        """Initializes or re-initializes the MediaPipe Hands object with a specific model complexity."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
            
        self.current_complexity = complexity
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # Limiting to 2 hands for lock gesture
            min_detection_confidence=self.min_conf,
            min_tracking_confidence=self.min_conf,
            model_complexity=complexity
        )
        
    def set_model_complexity(self, complexity: int):
        """Dynamically adjusts the MediaPipe model complexity (0 or 1)."""
        if self.current_complexity != complexity:
            self._init_mediapipe(complexity)

    def _load_config(self) -> dict:
        """Loads configuration from config.json."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def detect(self, frame: Any, draw_landmarks: bool = False) -> Optional[Any]:
        """
        Processes a single BGR frame, detects hands, and returns the landmarks of the first hand found.
        
        Args:
            frame: A BGR image directly from cv2.VideoCapture.
            draw_landmarks: If true, draws the skeleton directly onto the input frame.
            
        Returns:
            The NormalizedLandmarkList (the landmarks) for the hand if detected, otherwise None.
        """
        # Convert the BGR image to RGB before processing.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        hand_landmarks_result = None

        # Draw the hand annotations on the image if requested.
        if results.multi_hand_landmarks:
            hand_landmarks_result = results.multi_hand_landmarks
            
            if draw_landmarks:
                for hl in hand_landmarks_result:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hl,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
        return hand_landmarks_result

    def close(self):
        """Releases underlying mediapipe resources."""
        self.hands.close()
