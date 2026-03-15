import math
import time
from typing import Tuple, Optional

class OneEuroFilter:
    """1 Euro Filter for real-time, low-latency noise reduction used in VR tracking."""
    def __init__(self, min_cutoff=0.01, beta=1.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = 0.0
        self.dy_prev = 0.0
        self.t_prev = None

    def _alpha(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t: float, x: float, y: float) -> Tuple[float, float]:
        if self.t_prev is None:
            self.x_prev, self.y_prev = x, y
            self.t_prev = t
            return x, y

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev, self.y_prev

        # Calculate filtered derivative (velocity)
        ad = self._alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dy = (y - self.y_prev) / t_e
        
        dx_hat = ad * dx + (1 - ad) * self.dx_prev
        dy_hat = ad * dy + (1 - ad) * self.dy_prev

        # Calculate cutoff frequency based on velocity
        velocity_mag = math.hypot(dx_hat, dy_hat)
        cutoff = self.min_cutoff + self.beta * velocity_mag

        # Low-pass filter the points using the dynamic cutoff
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        y_hat = a * y + (1 - a) * self.y_prev

        self.x_prev, self.y_prev = x_hat, y_hat
        self.dx_prev, self.dy_prev = dx_hat, dy_hat
        self.t_prev = t

        return x_hat, y_hat

class PositionSmoother:
    def __init__(self, smoothing_factor: float = 0.5, movement_threshold: float = 0.005):
        """
        Replaced the custom EMA with the 1 Euro Filter algorithm.
        It eliminates jitter entirely at slow speeds while remaining instantly responsive at high speeds.
        """
        # Butter-smooth constants:
        # min_cutoff=0.02: Gentle smoothing at rest to kill micro-jitter
        # beta=5.0: Aggressively ramps up responsiveness during fast motion
        #           so the cursor snaps to your hand position with near-zero lag
        self.filter = OneEuroFilter(min_cutoff=0.02, beta=5.0)

    def smooth(self, new_x: float, new_y: float) -> Tuple[float, float]:
        return self.filter(time.time(), new_x, new_y)
