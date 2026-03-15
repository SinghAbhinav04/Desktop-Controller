import queue
import threading
import time
import cv2
import json
import os
from utils.logger import get_logger
from core.camera_manager import CameraManager
from core.hand_tracker import HandTracker
from core.gesture_detector import GestureDetector, GestureEvent
from controllers.mouse_controller import MouseController
from core.frame_processor import FrameProcessor
from utils.smoothing import PositionSmoother

class GesturePipeline:
    def __init__(self):
        self.logger = get_logger("GesturePipeline")
        
        # Load config
        self._load_config()

        # Initialize core components
        self.camera_manager = CameraManager()
        self.hand_tracker = HandTracker()
        self.gesture_detector = GestureDetector()
        self.mouse_controller = MouseController()
        
        # 1. Performance - Deep Downscale to 256x144 (16:9 ratio) for high FPS
        # Preserving the native 16:9 Mac webcam aspect ratio is critical for MediaPipe accuracy
        self.frame_processor = FrameProcessor(target_width=256, target_height=144)
        
        # Pull smoothing parameters from config
        m_thresh = self.config.get('movement_threshold', 5)
        # Auto-correct pixel-based thresholds to normalized scale (0-1)
        if m_thresh >= 1.0:
            m_thresh = m_thresh / 1000.0
            
        self.smoother = PositionSmoother(
            smoothing_factor=self.config.get('smoothing_factor', 0.6),
            movement_threshold=m_thresh
        )

        # Queues for inter-thread communication. 
        self.frame_queue = queue.Queue(maxsize=2)
        self.landmark_queue = queue.Queue(maxsize=2)
        self.gesture_queue = queue.Queue(maxsize=2)
        
        self.latest_frame = None
        self.latest_landmarks = None
        self.latest_gesture_name = "NONE"
        self.running = False
        self.threads = []
        
        # Performance / Profiling state
        self.profiling = {'camera': 0.0, 'tracking': 0.0, 'gesture': 0.0, 'mouse': 0.0}
        self.frames_processed = 0
        self.last_log_time = time.time()

    def _load_config(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, 'config', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def _log_performance(self):
        """Logs average FPS and slowest thread every 5 seconds, dynamically reducing complexity if needed."""
        now = time.time()
        elapsed = now - self.last_log_time
        if elapsed >= 5.0:
            fps = self.frames_processed / elapsed
            slowest_thread = max(self.profiling, key=self.profiling.get)
            slowest_time = self.profiling[slowest_thread]
            
            self.logger.info(f"Performance - FPS: {fps:.1f} | Slowest Thread: {slowest_thread} ({slowest_time*1000:.1f}ms)")
            
            # 2. Performance - Dynamic tune down if FPS drops
            if fps < 20 and self.hand_tracker.current_complexity == 1:
                self.logger.warning("FPS < 20. Reducing MediaPipe model complexity to 0.")
                self.hand_tracker.set_model_complexity(0)
            elif fps > 40:
                self.hand_tracker.set_model_complexity(1)
                
            self.frames_processed = 0
            self.last_log_time = now

    def _camera_thread(self):
        """Thread 1: Camera capture -> frame_queue"""
        cap = self.camera_manager.get_camera()
        if not cap:
            self.logger.error("Camera failed to initialize. Exiting...")
            self.running = False
            return

        fps_limit = self.config.get('fps_limit', 30)
        frame_time = 1.0 / fps_limit

        self.logger.info("Camera thread started.")
        while self.running:
            t0 = time.time()
            
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            self.latest_frame = frame.copy()
            
            try:
                 self.frame_queue.put_nowait(frame)
            except queue.Full:
                 try:
                     self.frame_queue.get_nowait()
                     self.frame_queue.put_nowait(frame)
                 except (queue.Empty, queue.Full):
                     pass
            
            self.profiling['camera'] = time.time() - t0
            
            elapsed = time.time() - t0 # total elapsed including queue logic
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
            
        cap.release()
        self.logger.info("Camera thread stopped.")

    def _tracking_thread(self):
        """Thread 2: frame_queue -> landmark_queue"""
        self.logger.info("Tracking thread started.")
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                t0 = time.time()
                
                # 3. Performance - Downscale before inference
                small_frame = self.frame_processor.process(frame)
                
                # Hand detection on the smaller frame
                landmarks = self.hand_tracker.detect(small_frame, draw_landmarks=False)
                
                self.latest_landmarks = landmarks
                
                self.profiling['tracking'] = time.time() - t0
                
                # Increment FPS counter when tracking succeeds (this is our main bottleneck usually)
                self.frames_processed += 1
                self._log_performance()
                
                try:
                    self.landmark_queue.put_nowait(landmarks)
                except queue.Full:
                    try:
                        self.landmark_queue.get_nowait()
                        self.landmark_queue.put_nowait(landmarks)
                    except (queue.Empty, queue.Full):
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in tracking thread: {e}")
        self.logger.info("Tracking thread stopped.")

    def _gesture_thread(self):
        """Thread 3: landmark_queue -> gesture_queue"""
        self.logger.info("Gesture thread started.")
        while self.running:
            try:
                landmarks = self.landmark_queue.get(timeout=0.1)
                
                t0 = time.time()
                event = self.gesture_detector.detect(landmarks)
                self.profiling['gesture'] = time.time() - t0
                
                if event:
                    try:
                        self.gesture_queue.put_nowait(event)
                    except queue.Full:
                        try:
                            self.gesture_queue.get_nowait()
                            self.gesture_queue.put_nowait(event)
                        except (queue.Empty, queue.Full):
                            pass
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in gesture thread: {e}")
        self.logger.info("Gesture thread stopped.")

    def _mouse_thread(self):
        """Thread 4: gesture_queue -> MouseController"""
        self.logger.info("Mouse thread started.")
        while self.running:
            try:
                event: GestureEvent = self.gesture_queue.get(timeout=0.1)

                t0 = time.time()
                if event.type == GestureDetector.CURSOR_MOVE:
                     self.latest_gesture_name = "CURSOR_MOVE"
                     smoothed_x, smoothed_y = self.smoother.smooth(event.position[0], event.position[1])
                     self.mouse_controller.move(smoothed_x, smoothed_y)
                     if self.mouse_controller.is_dragging:
                         self.mouse_controller.end_drag()
                     if hasattr(self.mouse_controller, "last_scroll_y"):
                         delattr(self.mouse_controller, "last_scroll_y")

                elif event.type == GestureDetector.LEFT_CLICK:
                     self.latest_gesture_name = "LEFT_CLICK"
                     self.mouse_controller.click()
                     
                elif event.type == GestureDetector.RIGHT_CLICK:
                     self.latest_gesture_name = "RIGHT_CLICK"
                     self.mouse_controller.right_click()
                     
                elif event.type == GestureDetector.DOUBLE_CLICK:
                     self.latest_gesture_name = "DOUBLE_CLICK"
                     self.mouse_controller.double_click()
                     
                elif event.type == GestureDetector.DRAG:
                     self.latest_gesture_name = "DRAG"
                     self.mouse_controller.start_drag()
                     smoothed_x, smoothed_y = self.smoother.smooth(event.position[0], event.position[1])
                     self.mouse_controller.move(smoothed_x, smoothed_y)
                 
                elif event.type == GestureDetector.SCROLL:
                     self.latest_gesture_name = "SCROLL"
                     smoothed_x, smoothed_y = self.smoother.smooth(event.position[0], event.position[1])
                     self.mouse_controller.scroll(smoothed_y)
                     
                elif event.type == GestureDetector.CURSOR_LOCK:
                     self.latest_gesture_name = "LOCKED"
                     # Cursor is locked, so we Intentionally do not call move() or click()
                     if self.mouse_controller.is_dragging:
                         self.mouse_controller.end_drag()
                     if hasattr(self.mouse_controller, "last_scroll_y"):
                         delattr(self.mouse_controller, "last_scroll_y")
                     
                elif event.type == GestureDetector.SWIPE_OUT:
                     self.latest_gesture_name = "MISSION_CONTROL_OPEN"
                     self.mouse_controller.trigger_mission_control()
                     
                elif event.type == GestureDetector.SWIPE_IN:
                     self.latest_gesture_name = "MISSION_CONTROL_CLOSE"
                     self.mouse_controller.close_mission_control()
                
                self.profiling['mouse'] = time.time() - t0
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in mouse thread: {e}")
        self.logger.info("Mouse thread stopped.")

    def start(self):
        if self.running:
            return
            
        self.running = True
        
        self.threads = [
            threading.Thread(target=self._camera_thread, daemon=True, name="CameraThread"),
            threading.Thread(target=self._tracking_thread, daemon=True, name="TrackingThread"),
            threading.Thread(target=self._gesture_thread, daemon=True, name="GestureThread"),
            threading.Thread(target=self._mouse_thread, daemon=True, name="MouseThread")
        ]
        
        for t in self.threads:
            t.start()
            
        self.logger.info("Pipeline started.")

    def stop(self):
        self.logger.info("Stopping pipeline...")
        self.running = False
        for t in self.threads:
            t.join(timeout=2.0)
        self.hand_tracker.close()
        self.logger.info("Pipeline stopped.")
