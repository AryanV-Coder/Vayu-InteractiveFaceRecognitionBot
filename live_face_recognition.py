import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import time
import threading
import pyaudio
import numpy as np
import wave
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from utils.conversational_llm import ConversationalLLM
from utils.main_pipeline import main_pipeline, speak_text


## Keep Threshold as 0.65, below this face is not recognised

# ============ AUDIO CONFIGURATION ============
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512  # 32ms chunks at 16kHz

# We defer loading VAD model to prevent the child process (spawn) from loading it unnecessarily
vad_model = None

def get_vad_model():
    global vad_model
    if vad_model is None:
        import torch
        print("Loading Silero VAD model...")
        vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                       model='silero_vad',
                                       force_reload=False)
        print("✅ VAD model loaded")
    return vad_model

# ============ SHARED STATE ============
shared_state = {
    'current_name': '',
    'current_description': '',
    'current_conversation': None,
    'bot_is_speaking': False,
    'running': True,
}


# ============ FACE RECOGNITION WORKER (runs in separate process) ============
_worker_face_logic = None

def _init_worker():
    """Initializer for the worker process — creates its own FaceRecognitionLogic.
    Imports are deferred here so the main process doesn't force the child to re-import
    everything during a 'spawn' start."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    from utils.face_recognition_logic import FaceRecognitionLogic

    global _worker_face_logic
    _worker_face_logic = FaceRecognitionLogic()
    print("✅ Face recognition worker initialized (separate process)")


def _recognise_face_in_worker(image_path):
    """Runs in the worker process. Performs face recognition and returns the result.
    Returns dict {"name": ..., "description": ...} or None."""
    return _worker_face_logic.recognise_face(image_path)


def _handle_recognition_result(result, current_name, current_description, current_conversation):
    """Handle the recognition result from the worker process (runs in main process).
    Creates a new conversation and triggers greeting if person changed.

    Returns:
        tuple: (name, description, conversation)
    """
    if not result:
        print("❌ No face recognized")
        return current_name, current_description, current_conversation

    recognised_name = result["name"]
    recognised_description = result["description"]

    if recognised_name == current_name and recognised_description == current_description:
        print(f"🕵️ Person is same: {recognised_name}")
        return current_name, current_description, current_conversation
    else:
        print(f'''🚧 Person changed: 
              previous - {current_name}, 
              new - {recognised_name}''')

        current_conversation = ConversationalLLM(recognised_name, recognised_description)

        # Update shared_state so the audio thread can access the conversation
        shared_state['current_conversation'] = current_conversation
        shared_state['bot_is_speaking'] = False

        # # Auto-greet in background thread (so video loop isn't blocked)
        # def greet():
        #     greeting = current_conversation.send_message("Hello")
        #     print(f"👋 Greeting: {greeting}")
        #     speak_text(greeting, shared_state)

        # threading.Thread(target=greet, daemon=True).start()

        return recognised_name, recognised_description, current_conversation


def audio_recording_thread():
    """Thread: Continuous audio recording with Silero VAD.
    When user finishes speaking (silence detected), saves audio to file
    and calls main_pipeline (STT → LLM → TTS)."""

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("🎙️  Audio recording started - listening...")

    is_recording = False
    audio_buffer = []
    silence_counter = 0
    silence_threshold = 31    # ~1.0s of silence to end recording
    speech_threshold = 0.5
    min_speech_chunks = 10    # Minimum ~320ms to consider valid speech

    try:
        while shared_state['running']:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)

            # Skip while bot is speaking
            if shared_state['bot_is_speaking']:
                if is_recording:
                    # Discard buffer if bot starts speaking mid-recording
                    is_recording = False
                    audio_buffer = []
                    silence_counter = 0
                continue

            # Run Silero VAD on chunk
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            model = get_vad_model()
            import torch
            speech_prob = model(torch.from_numpy(audio_float32), RATE).item()
            is_speech = speech_prob > speech_threshold

            if is_speech:
                if not is_recording:
                    print("\n🎤 Speech detected - recording...")
                    is_recording = True
                audio_buffer.append(audio_chunk)
                silence_counter = 0

            elif is_recording:
                audio_buffer.append(audio_chunk)
                silence_counter += 1

                if silence_counter >= silence_threshold:
                    print("🔇 Silence detected - processing...")

                    if len(audio_buffer) >= min_speech_chunks:
                        # Save recorded audio to temp WAV file
                        audio_path = os.path.join("live_recognition", "temp_audio.wav")
                        os.makedirs("live_recognition", exist_ok=True)

                        with wave.open(audio_path, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)  # 16-bit = 2 bytes
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(audio_buffer))

                        # Process: STT → LLM → TTS (blocks until done)
                        main_pipeline(audio_path, shared_state)
                    else:
                        print("⚠️  Audio too short, skipping...")

                    # Reset recording state
                    is_recording = False
                    audio_buffer = []
                    silence_counter = 0
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("🎙️  Audio recording stopped")


def start_live_recognition():
    """Start live camera feed, face recognition (in separate process), and audio pipeline."""

    # Use 'spawn' context. On macOS, 'fork' with TensorFlow/Metal causes sudden crashes.
    # We use deferred imports so 'spawn' doesn't take 30s to start.
    mp_context = multiprocessing.get_context('spawn')
    face_pool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker, mp_context=mp_context)
    print("🚀 Face recognition process pool created (spawn mode)")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        face_pool.shutdown(wait=False)
        return

    print("📹 Camera opened successfully")
    print("Press 'q' to quit")

    # Start audio recording thread
    audio_thread = threading.Thread(target=audio_recording_thread, daemon=True)
    audio_thread.start()

    current_name = "unknown"
    current_description = ""
    current_conversation = ConversationalLLM("unknown", "")
    shared_state['current_conversation'] = current_conversation

    save_dir = "live_recognition"
    os.makedirs(save_dir, exist_ok=True)

    # Submit first frame immediately
    pending_future = None
    ret, frame = cap.read()
    if ret:
        print("\n🎬 Submitting first frame for recognition...")
        save_path = os.path.join(save_dir, "current_frame.jpg")
        cv2.imwrite(save_path, frame.copy())
        pending_future = face_pool.submit(_recognise_face_in_worker, save_path)

    last_process_time = time.time()
    process_interval = 5  # seconds

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Error: Failed to capture frame")
            break

        # ---------- Check if a pending recognition result is ready (non-blocking) ----------
        if pending_future is not None and pending_future.done():
            try:
                result = pending_future.result()
                print(f"\n🔍 Recognition result received at {time.strftime('%H:%M:%S')}")
                current_name, current_description, current_conversation = _handle_recognition_result(
                    result, current_name, current_description, current_conversation)
            except Exception as e:
                print(f"❌ Face recognition error: {e}")
            pending_future = None
            
            # Reset the timer AFTER processing finishes, so we wait a full 5s before next run
            last_process_time = time.time()

        # ---------- Submit new recognition task if interval is up AND no task pending ----------
        current_time = time.time()
        if current_time - last_process_time >= process_interval and pending_future is None:
            save_path = os.path.join(save_dir, "current_frame.jpg")
            cv2.imwrite(save_path, frame.copy())
            pending_future = face_pool.submit(_recognise_face_in_worker, save_path)
            last_process_time = current_time
            print(f"\n🔍 Frame submitted for recognition at {time.strftime('%H:%M:%S')}")

        # Display current recognition on frame
        if current_name:
            cv2.putText(frame, current_name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Cleanup
    shared_state['running'] = False
    face_pool.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released, process pool shut down, and windows closed")


if __name__ == "__main__":
    start_live_recognition()
