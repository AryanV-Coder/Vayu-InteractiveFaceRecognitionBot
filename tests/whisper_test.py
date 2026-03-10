import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Add parent (project root) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper_stt import stt

stt("WhatsApp Audio 2026-03-11 at 02.23.26.mp3")