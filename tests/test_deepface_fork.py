"""Test DeepFace inside a forked ProcessPoolExecutor."""
import os
import sys
import numpy as np

# Add parent (project root) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import deepface before fork to simulate the main script
from deepface import DeepFace

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from utils.face_recognition_logic import FaceRecognitionLogic

_worker_face_logic = None

def _init_worker():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    global _worker_face_logic
    _worker_face_logic = FaceRecognitionLogic()
    print("✅ Worker initialized")

def _deepface_task(img_path):
    print(f"Running DeepFace inside worker on {img_path}...")
    # This will crash if TensorFlow state was corrupted by fork
    return _worker_face_logic.recognise_face(img_path)

if __name__ == "__main__":
    # Create a dummy image
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite("dummy.jpg", dummy_img)

    print("1. Creating ProcessPoolExecutor with fork context...")
    mp_context = multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker, mp_context=mp_context) as pool:
        print("2. Submitting DeepFace task to worker process...")
        future = pool.submit(_deepface_task, "dummy.jpg")
        result = future.result(timeout=30)
        print(f"3. Worker returned: {result}")
    
    os.remove("dummy.jpg")
    print("ALL CHECKS PASSED ✅")
