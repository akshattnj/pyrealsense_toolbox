import pyrealsense2 as rs
import numpy as np
import cv2

# --- Global Variables ---
tracked_dist_cm = None

def select_point(event, x, y, flags, param):
    global tracked_dist_cm
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_f = param['depth_frame']
        dist = depth_f.get_distance(x, y) 
        if dist > 0:
            tracked_dist_cm = int(dist * 100)
            print(f"Locked onto depth: {tracked_dist_cm} cm")

# --- Setup RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

# --- GUI Setup ---
cv2.namedWindow('Dynamic Tracker')
data_packet = {'depth_frame': None}
cv2.setMouseCallback('Dynamic Tracker', select_point, data_packet)

def nothing(x): pass
# Individual sliders for front and back buffers
cv2.createTrackbar('Min Buffer (cm)', 'Dynamic Tracker', 10, 100, nothing)
cv2.createTrackbar('Max Buffer (cm)', 'Dynamic Tracker', 10, 100, nothing)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        data_packet['depth_frame'] = depth_frame
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get current buffer values from sliders
        min_buffer = cv2.getTrackbarPos('Min Buffer (cm)', 'Dynamic Tracker')
        max_buffer = cv2.getTrackbarPos('Max Buffer (cm)', 'Dynamic Tracker')

        if tracked_dist_cm is not None:
            # Apply individual buffers
            min_cm = max(0, tracked_dist_cm - min_buffer)
            max_cm = tracked_dist_cm + max_buffer
            
            # Convert to camera units
            min_units = (min_cm / 100.0) / depth_scale
            max_units = (max_cm / 100.0) / depth_scale

            # Create the isolation mask
            mask = cv2.inRange(depth_image, min_units, max_units)
            output = cv2.bitwise_and(color_image, color_image, mask=mask)
            
            # HUD Overlay
            cv2.putText(output, f"Target: {tracked_dist_cm}cm (-{min_buffer}/+{max_buffer})", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            output = color_image.copy()
            cv2.putText(output, "Click an object to lock depth range", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Dynamic Tracker', output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()