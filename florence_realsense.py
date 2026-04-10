import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import threading

# --- Init Florence-2 (Much better for your 6GB VRAM) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = 'microsoft/Florence-2-base'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Global distance storage
target_distance = 0.0
current_label = "Scanning..."

def run_florence(image_bgr, prompt):
    global target_distance, current_label
    
    # Convert BGR to RGB for Florence
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Task: Grounding and Captioning
    # Florence-2 uses specific task prefixes
    task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{prompt}"
    
    inputs = processor(text=task_prompt, images=image_rgb, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(results, task=task_prompt, image_size=(image_bgr.shape[1], image_bgr.shape[0]))
    
    # Logic to extract coordinates and query depth
    # The output format is usually: {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2]], 'labels': ['shirt']}}
    data = parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
    if data.get('bboxes'):
        bbox = data['bboxes'][0]
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Get depth from global frame (need to handle carefully with threading)
        # We'll update a label for the main loop to handle the depth call
        current_label = f"{data['labels'][0]} at {cx}, {cy}"
        return (cx, cy)
    return None

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
        depth_frame = aligned_frames.get_depth_frame()

        cv2.putText(color_image, f"Status: {current_label} | Dist: {target_distance:.2f}m", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Florence-2 Grounding", color_image)

        key = cv2.waitKey(1)
        if key == ord(' '):
            # Example: Find the shirt
            t = threading.Thread(target=lambda: handle_depth_query(color_image.copy(), depth_frame))
            t.start()
        elif key == ord('q'):
            break
finally:
    pipeline.stop() 