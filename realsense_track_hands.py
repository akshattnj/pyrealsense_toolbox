import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Setup MediaPipe Task ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

# TUNING FOR ALL POSES:
# Lowering 'min_hand_presence_confidence' helps detect hands that aren't perfectly frontal.
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5, # More sensitive to finding a hand
    min_hand_presence_confidence=0.4,  # More sensitive to keeping the hand in view
    min_tracking_confidence=0.5        # Helps when hand is moving/rotating
)
detector = vision.HandLandmarker.create_from_options(options)

# --- Setup RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

cv2.namedWindow('Universal Hand Isolator')
def nothing(x): pass
cv2.createTrackbar('Min Buffer (cm)', 'Universal Hand Isolator', 8, 50, nothing)
cv2.createTrackbar('Max Buffer (cm)', 'Universal Hand Isolator', 12, 50, nothing)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Tasks API works better with RGB
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = detector.detect(mp_image)
        output_image = color_image.copy()

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            h, w = color_image.shape[:2]

            # STABILITY LOGIC: Average depth of Wrist(0), Middle MCP(9), and Pinky MCP(17)
            # This prevents the 'flicker' when the hand rotates sideways.
            key_points = [hand_lms[0], hand_lms[9], hand_lms[17]]
            distances = []

            for kp in key_points:
                kx, ky = int(kp.x * w), int(kp.y * h)
                if 0 <= kx < w and 0 <= ky < h:
                    d = depth_frame.get_distance(kx, ky)
                    if d > 0: distances.append(d)

            if distances:
                avg_dist_m = sum(distances) / len(distances)
                dist_cm = avg_dist_m * 100
                
                min_b = cv2.getTrackbarPos('Min Buffer (cm)', 'Universal Hand Isolator')
                max_b = cv2.getTrackbarPos('Max Buffer (cm)', 'Universal Hand Isolator')

                min_units = ((dist_cm - min_b) / 100.0) / depth_scale
                max_units = ((dist_cm + max_b) / 100.0) / depth_scale

                # Masking
                mask = cv2.inRange(depth_image, min_units, max_units)
                output_image = cv2.bitwise_and(color_image, color_image, mask=mask)
                
                # Visual Feedback
                cv2.putText(output_image, f"Stable Dist: {dist_cm:.1f}cm", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Universal Hand Isolator', output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()