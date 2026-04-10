import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 1. Initialize YOLO26 on GPU
# 'device=0' targets your 1660 Ti. 'half=True' uses FP16 for 2x speedup on Turing GPUs.
model = YOLO("yolo26x.pt").to('cuda') 

# 2. RealSense Setup (Same as before)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # 3. GPU-Accelerated Inference
        # We pass device=0 and half=True for maximum 1660 Ti performance
        results = model.predict(color_image, device=0, half=True, stream=True, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                
                # Distance sampling
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                distance = depth_frame.get_distance(cx, cy)

                # Annotate
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(color_image, f"{label}: {distance:.2f}m", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("1660 Ti Accelerated YOLO26", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()