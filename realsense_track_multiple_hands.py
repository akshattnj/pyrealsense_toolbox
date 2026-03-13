import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Setup MediaPipe Task ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=4, # Support up to 4 hands
    min_hand_detection_confidence=0.5
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

cv2.namedWindow('Multi-Depth Tracker')
def nothing(x): pass
# This slider now controls the THICKNESS of the bubble around each hand
cv2.createTrackbar('Bubble Thickness (cm)', 'Multi-Depth Tracker', 10, 30, nothing)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = detector.detect(mp_image)

        # We start with a black mask (hide everything)
        master_mask = np.zeros(depth_image.shape, dtype=np.uint8)
        output_image = color_image.copy()

        if result.hand_landmarks:
            h, w = color_image.shape[:2]
            thickness = cv2.getTrackbarPos('Bubble Thickness (cm)', 'Multi-Depth Tracker')

            for hand_lms in result.hand_landmarks:
                # 1. Get depth for THIS specific hand
                palm = hand_lms[9]
                cx, cy = int(palm.x * w), int(palm.y * h)
                
                if 0 <= cx < w and 0 <= cy < h:
                    dist_m = depth_frame.get_distance(cx, cy)
                    if dist_m > 0:
                        dist_cm = dist_m * 100
                        
                        # 2. Create a "Local Slice" for this hand
                        min_u = ((dist_cm - thickness) / 100.0) / depth_scale
                        max_u = ((dist_cm + thickness) / 100.0) / depth_scale
                        
                        # Mask for just this hand's depth
                        hand_slice = cv2.inRange(depth_image, min_u, max_u)
                        
                        # 3. Add this hand's slice to the master mask
                        master_mask = cv2.bitwise_or(master_mask, hand_slice)

                        # Feedback
                        cv2.putText(output_image, f"{dist_cm:.0f}cm", (cx, cy-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Apply the combined "bubbles" to the final image
            output_image = cv2.bitwise_and(color_image, color_image, mask=master_mask)

        cv2.imshow('Multi-Depth Tracker', output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()