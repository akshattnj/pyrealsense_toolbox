import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import ollama  # pip install ollama
import threading

# 1. Load YOLO26 on GPU
yolo_model = YOLO("yolo26n.pt").to('cuda') # Using 'n' to save VRAM for the LLM

import base64

def ask_local_llm(frame):
    print("🏠 Local AI is looking at the headphones...")
    
    # 1. Convert frame to Base64 (The most reliable format for Ollama)
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    try:
        # 2. Call generate with stream=False to get the whole answer at once
        response = ollama.generate(
            model='moondream',
            prompt='does this user seem to like monster energy? if yes, what flavour? describe the placement of the can!',
            images=[base64_image],
            stream=False,  # This prevents the blank output issue
            options={
                'temperature': 0.8,        # Higher = more creative/random (0.0 to 1.0+)
                'top_p': 0.9,              # Nucleus sampling; higher = more diverse vocabulary
                'repeat_penalty': 1.2,     # Penalizes the model for repeating the same words
                'num_predict': 100,        # Limit response length to save VRAM/time
            }
        )
        
        # 3. Access the 'response' key
        answer = response.get('response', 'No response received.')
        print(f"\n🧠 Local AI says: {answer}\n")
        
    except Exception as e:
        print(f"❌ Local LLM Error: {e}")

# 2. RealSense Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_image = np.asanyarray(frames.get_color_frame().get_data())

        # 3. Fast YOLO Detection
        results = yolo_model.predict(color_image, device=0, half=True, stream=True, verbose=False)
        
        # Display the live YOLO feed
        for r in results:
            annotated_frame = r.plot()

        cv2.putText(annotated_frame, "Press 'SPACE' for Local AI description", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("YOLO26 + Local Vision LLM", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            # Run LLM in background so the 20FPS stream doesn't stutter
            threading.Thread(target=ask_local_llm, args=(color_image.copy(),)).start()
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()