import pyautogui
from utils.logger import get_logger

class MouseController:
    def __init__(self):
        self.logger = get_logger("MouseController")
        
        # Disable FailSafe because gesture controllers can inadvertently
        # trigger it when pushing against the edge of the screen
        pyautogui.FAILSAFE = False
        
        # Performance: Disable PyAutoGUI implicit delays (vital for macOS)
        pyautogui.PAUSE = 0
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.MINIMUM_SLEEP = 0
        
        # Get host screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Bounding box margin (5%) to easily reach screen corners
        # If the hand moves within the inner 90% of the camera view, it spans 100% of the screen.
        self.margin_x = 0.05
        self.margin_y = 0.05
        
        # Track drag state so we don't spam mousedown events
        self.is_dragging = False
        
        self.logger.info(f"Initialized. Screen resolution: {self.screen_width}x{self.screen_height}")

    def _normalize_to_screen(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """
        Converts normalized camera coordinates (0.0 - 1.0) into screen coordinates,
        applying a bounding box margin to ensure corners are reachable.
        """
        # Constrain to margin box
        bounded_x = max(self.margin_x, min(norm_x, 1.0 - self.margin_x))
        bounded_y = max(self.margin_y, min(norm_y, 1.0 - self.margin_y))

        # Map the restricted bounding box [margin, 1-margin] to [0, 1]
        mapped_x = (bounded_x - self.margin_x) / (1.0 - 2 * self.margin_x)
        mapped_y = (bounded_y - self.margin_y) / (1.0 - 2 * self.margin_y)

        # ⚠️ Note: For mirrored interaction (webcam style), you usually want to mirror the X axis
        # Assuming the camera is already flipping horizontally or we want to flip logic here
        # E.g., moving hand right (higher camera X) = moving cursor right (higher screen X).
        # We'll leave it 1:1, but the main pipeline should flip the image for natural UX.
        
        screen_x = int(mapped_x * self.screen_width)
        screen_y = int(mapped_y * self.screen_height)
        return screen_x, screen_y

    def move(self, x: float, y: float):
        """Moves cursor to the given normalized camera coordinates."""
        screen_pos = self._normalize_to_screen(x, y)
        
        # Log occasionally so the user knows code is running even if macOS blocks the actual mouse
        import time
        if getattr(self, '_last_log_time', 0) < time.time() - 2.0:
            self.logger.info(f"Moving cursor to ({screen_pos[0]}, {screen_pos[1]}). If it doesn't move, check macOS Accessibility Privacy Settings.")
            self._last_log_time = time.time()
            
        # ⚠️ Note: macOS has a known PyAutoGUI bug where moveTo takes longer than Windows.
        # Adding `_pause=False` bypasses the default 0.1s block.
        pyautogui.moveTo(screen_pos[0], screen_pos[1], _pause=False)

    def click(self):
        """Standard left click."""
        self.logger.debug("Executing Left Click")
        pyautogui.click(_pause=False)

    def right_click(self):
        """Standard right click."""
        self.logger.debug("Executing Right Click")
        pyautogui.click(button='right', _pause=False)

    def double_click(self):
        """Double click."""
        self.logger.debug("Executing Double Click")
        pyautogui.doubleClick(_pause=False)

    def start_drag(self):
        """Sends mousedown if not currently dragging."""
        if not self.is_dragging:
            self.logger.debug("Started Drag")
            pyautogui.mouseDown(_pause=False)
            self.is_dragging = True

    def end_drag(self):
        """Sends mouseup if currently dragging."""
        if self.is_dragging:
            self.logger.debug("Ended Drag")
            pyautogui.mouseUp(_pause=False)
            self.is_dragging = False

    def trigger_mission_control(self):
        """Triggers macOS Mission Control by pressing 'ctrl+up'."""
        self.logger.info("Triggering Mission Control (Open)")
        # F3 is standard, but some Macs map it to ctrl+up
        pyautogui.hotkey('ctrl', 'up', _pause=False)

    def close_mission_control(self):
        """Dismisses macOS Mission Control by pressing 'ctrl+down'."""
        self.logger.info("Triggering Mission Control (Close)")
        pyautogui.hotkey('ctrl', 'down', _pause=False)

    def scroll(self, y: float):
        """Scrolls the screen based on relative vertical movement."""
        if not hasattr(self, "last_scroll_y"):
            self.last_scroll_y = y
            return
            
        delta_y = y - self.last_scroll_y
        if abs(delta_y) > 0.005: # Threshold to prevent jitter
            # Translate normalized delta to scroll clicks
            # Mac scrolling is continuous; typical values are 1-20
            scroll_amount = int(-delta_y * 300) # Negative because y grows downward, but we scroll down when hand moves down
            if scroll_amount != 0:
                pyautogui.scroll(scroll_amount, _pause=False)
                self.last_scroll_y = y

