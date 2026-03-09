import json
import wave
from vosk import Model, KaldiRecognizer

def transcribe_audio_file(audio_file_path, model_path="model"):
    """
    Transcribe a complete audio file using Vosk
    
    Args:
        audio_file_path: Path to the WAV audio file
        model_path: Path to Vosk model directory
        
    Returns:
        str: Transcribed text
    """
    
    # Load Vosk model
    print(f"Loading Vosk model from: {model_path}")
    try:
        model = Model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Open audio file
    print(f"Opening audio file: {audio_file_path}")
    try:
        wf = wave.open(audio_file_path, "rb")
    except Exception as e:
        print(f"✗ Error opening audio file: {e}")
        return None
    
    # Check audio format
    if wf.getnchannels() != 1:
        print("✗ Audio must be mono (1 channel)")
        return None
    if wf.getsampwidth() != 2:
        print("✗ Audio must be 16-bit")
        return None
    if wf.getcomptype() != "NONE":
        print("✗ Audio must be PCM format")
        return None
    
    # Create recognizer
    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)
    
    print("Processing audio...")
    
    # Read and process audio in chunks
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        recognizer.AcceptWaveform(data)
    
    # Get final result
    result = json.loads(recognizer.FinalResult())
    text = result.get('text', '')
    
    wf.close()
    
    if text:
        print("\n✓ Transcription complete!")
        return text
    else:
        print("\n✗ No speech detected")
        return ""


if __name__ == "__main__":
    import sys
    
    print("=== Vosk Speech-to-Text (File Mode) ===\n")
    
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default file path - change this to your audio file
        audio_file = "sarvam/audio/t.wav"
    
    # Optional: specify different model path
    model = "model"
    if len(sys.argv) > 2:
        model = sys.argv[2]
    
    # Transcribe
    transcription = transcribe_audio_file(audio_file, model)
    
    if transcription:
        print("\n" + "=" * 50)
        print("TRANSCRIPTION:")
        print("=" * 50)
        print(transcription)
        print("=" * 50)
