import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import ollama  # pip install ollama
import threading
import base64
import tkinter as tk # Added for input window
from tkinter import simpledialog

# --- NEW: Global variable for dynamic questions ---
current_prompt = "does this user seem to like monster energy? if yes, what flavour? describe the placement of the can!"

# 1. Load YOLO26 on GPU
yolo_model = YOLO("yolo26n.pt").to('cuda') # Using 'n' to save VRAM for the LLM

# --- NEW: Function to open the input window ---
def get_user_input():
    global current_prompt
    # Create a hidden tkinter root
    root = tk.Tk()
    root.withdraw()
    # Open the input dialog
    new_q = simpledialog.askstring("AI Query", "Ask the AI a question about this frame:", initialvalue=current_prompt)
    if new_q:
        current_prompt = new_q
        print(f"🎯 Prompt updated to: {current_prompt}")
    root.destroy()

def ask_local_llm(frame):
    global current_prompt # Access the dynamic prompt
    print(f"🏠 Local AI is analyzing: {current_prompt}")
    
    # 1. Convert frame to Base64
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    try:
        # 2. Use current_prompt instead of hardcoded string
        response = ollama.generate(
            model='moondream',
            prompt=current_prompt,
            images=[base64_image],
            stream=False,
            keep_alive=-1,
            options={
                'temperature': 0.8,
                'top_p': 0.9,
                'repeat_penalty': 1.2,
                'num_predict': 100,
            }
        )
        
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

        # Added instructions for the new input key
        cv2.putText(annotated_frame, "SPACE: Analyze | 'i': Change Question", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Q: {current_prompt[:40]}...", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("YOLO26 + Local Vision LLM", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            # Run LLM in background so the 20FPS stream doesn't stutter
            threading.Thread(target=ask_local_llm, args=(color_image.copy(),)).start()
        
        # --- NEW: Press 'i' to open the input window ---
        elif key == ord('i'):
            threading.Thread(target=get_user_input).start()
            
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()