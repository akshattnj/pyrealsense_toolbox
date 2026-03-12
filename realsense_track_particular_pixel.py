import pyrealsense2 as rs
import numpy as np
import cv2

# --- Global Tracking Variables ---
target_coords = None # Stores the (x, y) you clicked

def select_point(event, x, y, flags, param):
    global target_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        target_coords = (x, y)
        print(f"Locked coordinates: {x}, {y}")

# --- Setup RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

# --- GUI Setup ---
cv2.namedWindow('Auto-Chasing Tracker')
cv2.setMouseCallback('Auto-Chasing Tracker', select_point)

def nothing(x): pass
cv2.createTrackbar('Min Buffer (cm)', 'Auto-Chasing Tracker', 10, 100, nothing)
cv2.createTrackbar('Max Buffer (cm)', 'Auto-Chasing Tracker', 10, 100, nothing)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        min_buffer = cv2.getTrackbarPos('Min Buffer (cm)', 'Auto-Chasing Tracker')
        max_buffer = cv2.getTrackbarPos('Max Buffer (cm)', 'Auto-Chasing Tracker')

        if target_coords is not None:
            # 1. LIVE SAMPLING: Get current distance at the clicked coordinates
            curr_dist_m = depth_frame.get_distance(target_coords[0], target_coords[1])
            
            if curr_dist_m > 0:
                curr_dist_cm = curr_dist_m * 100
                
                # 2. AUTO-ADJUST: Move the window based on the new live distance
                min_cm = max(0, curr_dist_cm - min_buffer)
                max_cm = curr_dist_cm + max_buffer
                
                min_units = (min_cm / 100.0) / depth_scale
                max_units = (max_cm / 100.0) / depth_scale

                # 3. MASKING
                mask = cv2.inRange(depth_image, min_units, max_units)
                output = cv2.bitwise_and(color_image, color_image, mask=mask)
                
                # Draw a small crosshair on the tracked point
                cv2.drawMarker(output, target_coords, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(output, f"Live Dist: {curr_dist_cm:.1f}cm", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # If the sensor loses the pixel (e.g. hand moves away), show original
                output = color_image.copy()
                cv2.putText(output, "Object Lost (Invalid Depth)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            output = color_image.copy()
            cv2.putText(output, "Click to Lock & Chase", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Auto-Chasing Tracker', output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()