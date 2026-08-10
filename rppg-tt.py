import cv2, numpy as np, mediapipe as mp, scipy.signal as sig
from mediapipe.tasks.python import vision, BaseOptions

cap, lm_det = cv2.VideoCapture(0), vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path='face_landmarker.task')))
hues, fps = [], 30
while len(hues) < fps * 10:
    _, img = cap.read()
    res = lm_det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if res.face_landmarks:
        h, w, _ = img.shape; pt = res.face_landmarks[0][151]; cx, cy = int(pt.x*w), int(pt.y*h)
        roi = img[max(0, cy-15):cy+15, max(0, cx-15):cx+15]
        if roi.size: hues.append(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:,:,0]))
cap.release()
f, p = sig.periodogram(sig.filtfilt(*sig.butter(4, [0.75/15, 2.5/15], 'band'), sig.detrend(hues)), fs=fps, nfft=2048); print(f"BPM: {f[np.argmax(p)]*60:.1f}")