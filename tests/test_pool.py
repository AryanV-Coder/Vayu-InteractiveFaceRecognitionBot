"""Test that ProcessPoolExecutor with fork context works for face recognition."""
import os
import sys

# Add parent (project root) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
    print("✅ Worker initialized in separate process!")

def _dummy_task(x):
    return x * 2

if __name__ == "__main__":
    print("1. Creating ProcessPoolExecutor with fork context...")
    mp_context = multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker, mp_context=mp_context) as pool:
        print("2. Submitting dummy task to worker process...")
        future = pool.submit(_dummy_task, 21)
        result = future.result(timeout=30)
        print(f"3. Worker returned: {result}")
        assert result == 42
    print("ALL CHECKS PASSED ✅ — ProcessPoolExecutor (fork) works!")
