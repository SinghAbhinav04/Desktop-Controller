import cv2

class FrameProcessor:
    def __init__(self, target_width: int = 640, target_height: int = 480):
        self.target_size = (target_width, target_height)

    def process(self, frame):
        """Resizes the frame to a standard resolution to improve MediaPipe performance and stabilize tracking scale."""
        if frame is None:
            return None
        return cv2.resize(frame, self.target_size)
