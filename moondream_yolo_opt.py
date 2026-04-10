import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import ollama
import threading
import base64
import tkinter as tk
from tkinter import simpledialog

# --- Global Configuration ---
current_prompt = "What color is the shirt the person is wearing?"
llm_busy = False
trigger_input = False
show_yolo = True  # Toggle flag for YOLO detections

# Initialize YOLO
yolo_model = YOLO("yolo26n.pt").to('cuda')

# Initialize Tkinter Root
root = tk.Tk()
root.withdraw()

def ask_local_llm(frame):
    global current_prompt, llm_busy
    if llm_busy: return
    llm_busy = True
    
    # Encode for Moondream
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    base64_image = base64.b64encode(buffer).decode('utf-8')

    try:
        response = ollama.generate(
            model='moondream',
            prompt=current_prompt,
            images=[base64_image],
            options={
                'temperature': 0.1,
                'num_predict': 100,
                'num_ctx': 2048
            }
        )
        answer = response.get('response', '').strip()
        print(f"\n🧠 AI: {answer if answer else 'No response...'}\n")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
    finally:
        llm_busy = False

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        color_image = np.asanyarray(color_frame.get_data())

        # Handle UI Input Request
        if trigger_input:
            new_q = simpledialog.askstring("AI Query", "Ask the AI:", initialvalue=current_prompt)
            if new_q: current_prompt = new_q
            trigger_input = False 

        # --- YOLO Logic with Toggle ---
        if show_yolo:
            results = yolo_model.predict(color_image, device=0, half=True, verbose=False)
            display_frame = results[0].plot()
        else:
            # If YOLO is off, just show the raw frame (saves GPU inference time)
            display_frame = color_image.copy()

        # UI Overlays
        yolo_status = "ON" if show_yolo else "OFF"
        cv2.putText(display_frame, f"YOLO [Y]: {yolo_status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Q: {current_prompt[:40]}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Vision System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            threading.Thread(target=ask_local_llm, args=(color_image.copy(),), daemon=True).start()
        elif key == ord('i'):
            trigger_input = True
        elif key == ord('y'):
            show_yolo = not show_yolo # Toggle YOLO
            print(f"Toggle: YOLO is now {'ENABLED' if show_yolo else 'DISABLED'}")
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    root.destroy()