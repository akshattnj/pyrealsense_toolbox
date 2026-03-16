import numpy as np
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
FS = 15
CALIBRATION_TIME = 10
WINDOW_SEC = 15
BUFFER_SIZE = FS * WINDOW_SEC
MODEL_PATH = 'pose_landmarker_heavy.task'

# --- 1. MEDIAPIAPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO)

landmarker = PoseLandmarker.create_from_options(options)

# --- 2. REALSENSE HARDWARE SETUP ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, FS)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, FS) # Color needed for MediaPipe

profile = pipeline.start(config)
device = profile.get_device()
depth_sensor = device.query_sensors()[0]

if depth_sensor.supports(rs.option.visual_preset):
    depth_sensor.set_option(rs.option.visual_preset, 4) 
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

decimation = rs.decimation_filter(3)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color) # Align depth to color for MediaPipe matching

# --- 3. DATA STORAGE ---
depth_buffer = np.zeros(BUFFER_SIZE)
time_axis = np.linspace(-WINDOW_SEC, 0, BUFFER_SIZE)
start_time = time.time()

# --- 4. VISUALIZATION SETUP ---
fig, ax = plt.subplots(figsize=(9, 5))
line, = ax.plot(time_axis, depth_buffer, color='#00ff00', lw=2)
status_text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center', 
                     fontsize=14, fontweight='bold', color='white',
                     bbox=dict(facecolor='black', alpha=0.7))

ax.set_ylim(-10, 10)
ax.set_title("Live Chest Displacement (MediaPipe Auto-Tracking)")
ax.grid(True, alpha=0.2)

def update(frame):
    global depth_buffer
    
    # Capture and Align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_f = aligned_frames.get_depth_frame()
    color_f = aligned_frames.get_color_frame()
    
    if not depth_f or not color_f: return line, status_text

    # --- MEDIAPIPE POSE DETECTION ---
    color_image = np.asanyarray(color_f.get_data())
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
    
    # Process pose at current timestamp
    timestamp_ms = int((time.time() - start_time) * 1000)
    pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    # Default ROI (Center) in case MediaPipe fails
    h_img, w_img = color_image.shape[:2]
    x1, y1, x2, y2 = int(w_img/3), int(h_img/3), int(2*w_img/3), int(2*h_img/3)

    if pose_result.pose_landmarks:
        # Landmark 11 = Left Shoulder, 12 = Right Shoulder
        lm = pose_result.pose_landmarks[0]
        sh_l, sh_r = lm[11], lm[12]
        
        # Calculate chest center and width based on shoulders
        chest_center_x = int((sh_l.x + sh_r.x) / 2 * w_img)
        chest_center_y = int((sh_l.y + sh_r.y) / 2 * h_img) + 40 # Offset down to chest
        box_w = int(abs(sh_l.x - sh_r.x) * w_img * 0.6) # 60% of shoulder width
        
        x1, y1 = chest_center_x - box_w//2, chest_center_y - box_w//2
        x2, y2 = chest_center_x + box_w//2, chest_center_y + box_w//2

    # Draw the dynamic ROI box on a colored depth feed for the viewfinder
    viewfinder_img = np.asanyarray(colorizer.colorize(depth_f).get_data())
    cv2.rectangle(viewfinder_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.imshow("RealSense + MediaPipe Tracking", viewfinder_img)
    cv2.waitKey(1)

    # --- DEPTH CALCULATION ---
    depth_data = np.asanyarray(depth_f.get_data())
    # Ensure ROI is within image bounds
    y1, y2, x1, x2 = max(0, y1), min(h_img, y2), max(0, x1), min(w_img, x2)
    roi = depth_data[y1:y2, x1:x2]
    
    valid_pixels = roi[roi > 0]
    if valid_pixels.size > 0:
        depth_buffer = np.roll(depth_buffer, -1)
        depth_buffer[-1] = np.median(valid_pixels)

    # --- CALIBRATION & PLOT ---
    elapsed = time.time() - start_time
    instruction = "INHALE" if (elapsed % 5) < 2.5 else "EXHALE"
    if elapsed < CALIBRATION_TIME:
        status_text.set_text(f"CALIBRATION: {instruction} ({int(CALIBRATION_TIME-elapsed)}s)")
        if instruction == "INHALE":
            status_text.set_bbox(dict(facecolor='#27ae60'))
        else:
            status_text.set_bbox(dict(facecolor='#c0392b'))
    else:
        status_text.set_text("MONITORING LIVE")
        status_text.set_bbox(dict(facecolor='#2980b9'))

    plot_signal = (depth_buffer - np.mean(depth_buffer)) * -1
    line.set_ydata(plot_signal)
    
    if np.max(np.abs(plot_signal)) > 0.5:
        limit = np.max(np.abs(plot_signal)) + 2
        ax.set_ylim(-limit, limit)

    return line, status_text

try:
    ani = FuncAnimation(fig, update, interval=1000/FS, blit=True, cache_frame_data=False)
    plt.show()
finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    landmarker.close()