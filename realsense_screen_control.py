import pyrealsense2 as rs
import mediapipe as mp
import pyautogui
import cv2
import numpy as np
import time

# --- Config ---
BOX_SIZE = 0.25
SMOOTHING_NORMAL = 5
SMOOTHING_DRAG = 12   # Much higher smoothing while dragging to keep it steady
CLICK_THRESHOLD = 0.035 
model_path = 'hand_landmarker.task'

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

screen_w, screen_h = pyautogui.size()
box_x, box_y = 0.5, 0.5
pos_history = []
is_dragging = False

with HandLandmarker.create_from_options(options) as landmarker:
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            if latest_result and latest_result.hand_landmarks:
                landmarks = latest_result.hand_landmarks[0]
                index_tip, thumb_tip, middle_tip = landmarks[8], landmarks[4], landmarks[12]

                # 1. Box Repositioning (The Wiimote logic)
                dist_to_center = np.hypot(index_tip.x - box_x, index_tip.y - box_y)
                if dist_to_center > (BOX_SIZE * 0.8):
                    box_x += (index_tip.x - box_x) * 0.15
                    box_y += (index_tip.y - box_y) * 0.15
                else:
                    l, r = box_x - BOX_SIZE/2, box_x + BOX_SIZE/2
                    t, b = box_y - BOX_SIZE/2, box_y + BOX_SIZE/2
                    if index_tip.x < l: box_x -= (l - index_tip.x)
                    if index_tip.x > r: box_x += (index_tip.x - r)
                    if index_tip.y < t: box_y -= (t - index_tip.y)
                    if index_tip.y > b: box_y += (index_tip.y - b)
                box_x, box_y = np.clip(box_x, BOX_SIZE/2, 1-BOX_SIZE/2), np.clip(box_y, BOX_SIZE/2, 1-BOX_SIZE/2)

                # 2. Pinch Detection
                def get_dist(p1, p2): return np.hypot(p1.x - p2.x, p1.y - p2.y)
                d_thumb_index = get_dist(thumb_tip, index_tip)
                d_thumb_middle = get_dist(thumb_tip, middle_tip)

                # Left Click/Drag Logic
                if d_thumb_index < CLICK_THRESHOLD and d_thumb_middle > (CLICK_THRESHOLD * 1.5):
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                elif d_thumb_index > CLICK_THRESHOLD:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                # Right Click Logic (Quick pulse)
                if d_thumb_index < CLICK_THRESHOLD and d_thumb_middle < CLICK_THRESHOLD:
                    pyautogui.rightClick()
                    time.sleep(0.2) # Prevent multiple right clicks

                # 3. Mouse Movement (Enabled during drag!)
                rel_x = np.clip((index_tip.x - (box_x - BOX_SIZE/2)) / BOX_SIZE, 0, 1)
                rel_y = np.clip((index_tip.y - (box_y - BOX_SIZE/2)) / BOX_SIZE, 0, 1)

                target_x, target_y = rel_x * screen_w, rel_y * screen_h
                pos_history.append((target_x, target_y))
                
                # Dynamic Smoothing: Use more frames if dragging to stay steady
                current_smoothing = SMOOTHING_DRAG if is_dragging else SMOOTHING_NORMAL
                if len(pos_history) > current_smoothing: pos_history.pop(0)

                avg_x = sum([p[0] for p in pos_history]) / len(pos_history)
                avg_y = sum([p[1] for p in pos_history]) / len(pos_history)
                
                pyautogui.moveTo(avg_x, avg_y, _pause=False)

                # Visuals
                color = (0, 0, 255) if is_dragging else (0, 255, 0)
                bx1, by1 = int((box_x - BOX_SIZE/2) * w), int((box_y - BOX_SIZE/2) * h)
                bx2, by2 = int((box_x + BOX_SIZE/2) * w), int((box_y + BOX_SIZE/2) * h)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            cv2.imshow("RealSense Wiimote Drag", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()