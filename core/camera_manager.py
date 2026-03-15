import cv2
import json
import os

class CameraManager:
    def __init__(self):
        # Resolve config path statically relative to this file
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.config_path = os.path.join(base_dir, 'config', 'config.json')
        self.config = self._load_config()

    def _load_config(self):
        """Loads configuration from config.json."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_config(self):
        """Saves current configuration state back to config.json."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def scan_cameras(self, max_index=10):
        """Scans camera indices 0 through max_index and returns available ones."""
        available_cameras = []
        print(f"Scanning for cameras (0-{max_index})...")
        for index in range(max_index + 1):
            # ⚠️ Note: On macOS, we might sometimes need to explicitly define backends 
            # like cv2.CAP_AVFOUNDATION if the default fails.
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Read a single test frame to confirm it's a working feed
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(index)
                cap.release()
        return available_cameras

    def select_camera(self):
        """Interactive prompt for the user to select an active camera."""
        cameras = self.scan_cameras()
        if not cameras:
            print("No cameras found!")
            return None

        print("\nAvailable Cameras:")
        for idx in cameras:
            print(f"[{idx}] Camera {idx}")
            
        if len(cameras) == 1:
            selected_idx = cameras[0]
            print(f"Auto-selecting the only available camera: {selected_idx}")
        else:
            while True:
                try:
                    selection = input(f"Select camera index {cameras}: ")
                    selected_idx = int(selection)
                    if selected_idx in cameras:
                        break
                    else:
                        print("Invalid selection. Please choose from the available indices.")
                except ValueError:
                    print("Please enter a valid number.")

        self.config['camera_index'] = selected_idx
        self._save_config()
        print(f"Selected camera {selected_idx} saved to config.")
        return selected_idx

    def get_camera(self):
        """Returns a cv2.VideoCapture object for the configured camera."""
        camera_index = self.config.get('camera_index', 0)
        
        # ⚠️ Note: If camera initialization is very slow or fails on Mac, 
        # consider modifying this to cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera at index {camera_index}")
            return None
            
        # Optional: Apply some basic settings 
        cap.set(cv2.CAP_PROP_FPS, self.config.get('fps_limit', 30))
        
        return cap
