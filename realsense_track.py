import pyrealsense2 as rs
import numpy as np
import cv2

# --- Initialize RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

# --- Initialize Tracker ---
# CSRT is best for accuracy; if it's too slow, use cv2.TrackerKCF_create()
tracker = cv2.TrackerCSRT_create()
tracking_active = False
bbox = None

# --- GUI Setup ---
cv2.namedWindow('Full Motion Tracker')
def nothing(x): pass
cv2.createTrackbar('Min Buffer (cm)', 'Full Motion Tracker', 10, 100, nothing)
cv2.createTrackbar('Max Buffer (cm)', 'Full Motion Tracker', 15, 100, nothing)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if tracking_active:
            # Update the tracker with the new frame
            success, bbox = tracker.update(color_image)
            
            if success:
                # 1. Calculate the center of the bounding box
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Ensure the center is within the image bounds
                center_x = max(0, min(center_x, 639))
                center_y = max(0, min(center_y, 479))

                # 2. Get LIVE depth at the box's center
                curr_dist_m = depth_frame.get_distance(center_x, center_y)
                curr_dist_cm = curr_dist_m * 100

                # 3. Adjust sliders based on buffers
                min_buf = cv2.getTrackbarPos('Min Buffer (cm)', 'Full Motion Tracker')
                max_buf = cv2.getTrackbarPos('Max Buffer (cm)', 'Full Motion Tracker')
                
                min_units = ((curr_dist_cm - min_buf) / 100.0) / depth_scale
                max_units = ((curr_dist_cm + max_buf) / 100.0) / depth_scale

                # 4. Mask the image
                mask = cv2.inRange(depth_image, min_units, max_units)
                output = cv2.bitwise_and(color_image, color_image, mask=mask)

                # Draw tracking visuals
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(output, f"Dist: {curr_dist_cm:.1f}cm", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                output = color_image.copy()
                cv2.putText(output, "Tracking Lost!", (20, 40), 1, 2, (0, 0, 255), 2)
        else:
            output = color_image.copy()
            cv2.putText(output, "Press 'S' to Select Object", (20, 40), 1, 2, (255, 255, 0), 2)

        cv2.imshow('Full Motion Tracker', output)
        key = cv2.waitKey(1) & 0xFF

        # --- User Interaction ---
        if key == ord('s'):
            # Stop the stream momentarily to let the user draw a box
            bbox = cv2.selectROI('Full Motion Tracker', color_image, fromCenter=False)
            if bbox[2] > 0 and bbox[3] > 0: # Ensure valid box
                tracker = cv2.TrackerCSRT_create() # Reset tracker
                tracker.init(color_image, bbox)
                tracking_active = True
        
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()