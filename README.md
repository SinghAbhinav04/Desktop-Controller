# Virtual Gesture Desktop Controller (macOS)

A high-performance, multi-threaded macOS desktop controller powered by MediaPipe and OpenCV. Control your mouse, scroll, and trigger Mission Control using only hand gestures at up to 60 FPS.

## 🚀 Features

- **Multi-threaded Pipeline**: Separate threads for Camera, Tracking, Gestures, and Mouse control to ensure zero-lag performance.
- **1 Euro Filter Smoothing**: Advanced signal processing for jitter-free cursor movement that feels like a native hardware mouse.
- **Dynamic Resolution**: Automatic downscaling and model complexity adjustment to run smoothly on diverse hardware.
- **Native macOS Integration**: Direct control over Mission Control, scrolling, and clicks.

## 🖐️ Gesture Reference

### 1. Cursor Movement & Locking
- **Standard Move**: Hold up just your **Index finger** to move the cursor.
- **Lock Position**: Show your **entire palm** (4 fingers extended) to lock the cursor onto a target. The cursor will stay frozen in place until you release your palm.

### 2. Clicking & Dragging
- **Left Click**: While the cursor is **Locked**, bend your index finger inward.
- **Drag**: While the cursor is **Locked**, bend your index finger and hold it. Move your hand to drag the locked object.
- **Right Click**: Reverted (In development for higher accuracy).

### 3. Navigation & Scrolling
- **Scroll**: Hold up **Index and Middle fingers** (Peace sign) and move your hand Up or Down to natively scroll the page.
- **Mission Control (Open)**: Show **both palms** and move them **away** from each other horizontally.
- **Mission Control (Close)**: Show **both palms** and move them **towards** each other horizontally.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gesture-desktop-controller
   ```

2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Permissions (CRITICAL)**:
   - On macOS, you must grant **Accessibility Permissions** to your Terminal (iTerm2, Terminal.app, or VS Code) so `PyAutoGUI` can control the mouse.
   - Go to: `System Settings` > `Privacy & Security` > `Accessibility`.

## 🖥️ Usage

Run the main application:
```bash
python3 main.py
```
- Press **'q'** in the camera window to exit.

## ⚙️ Configuration

Tweak `config/config.json` to adjust sensitivity:
- `fps_limit`: Target framerate.
- `smoothing_factor`: Sensitivity of the movement filter.
- `confidence_threshold`: MediaPipe detection strictness.
