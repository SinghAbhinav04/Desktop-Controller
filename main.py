import cv2
import sys
import os
import mediapipe as mp
from pipeline.gesture_pipeline import GesturePipeline
from core.camera_manager import CameraManager
from utils.logger import get_logger

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    logger = get_logger("Main")
    
    # Check if a camera is configured yet
    cam_manager = CameraManager()
    if 'camera_index' not in cam_manager.config or not cam_manager.get_camera():
         logger.info("No valid camera configured. Running setup...")
         selected = cam_manager.select_camera()
         if selected is None:
             logger.error("Setup failed. Exiting.")
             sys.exit(1)

    # Initialize and start the multi-threaded pipeline
    pipeline = GesturePipeline()
    logger.info("Starting Gesture Pipeline...")
    pipeline.start()

    logger.info("Application running. Press 'q' in the video window to exit.")
    
    try:
        # Main Thread: UI Update Loop
        while True:
            # We just pull the latest frame from the pipeline thread to render
            # We don't process it here, keeping the UI thread incredibly lightweight
            frame = pipeline.latest_frame
            
            if frame is not None:
                # Get the latest tracked landmarks
                multi_landmarks = pipeline.latest_landmarks
                if multi_landmarks:
                    # In case it's a list from the updated hand_tracker
                    if not isinstance(multi_landmarks, list):
                        multi_landmarks = [multi_landmarks]
                    
                    for landmarks in multi_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, 
                            landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                
                # Optional: Overlay simple text
                gesture_text = getattr(pipeline, 'latest_gesture_name', 'NONE')
                cv2.putText(frame, f"State: {gesture_text}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow("Gesture Control", frame)
                
            # OpenCV requires waitKey to render images and handle window events
            # Use 1ms delay. If 'q' is pressed, break the UI loop.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit requested by user.")
                break

    except KeyboardInterrupt:
        logger.info("Ctrl+C detected.")
    finally:
        # Graceful shutdown of threads and resources
        pipeline.stop()
        cv2.destroyAllWindows()
        logger.info("Application exited purely.")

if __name__ == "__main__":
    main()
