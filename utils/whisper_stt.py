from faster_whisper import WhisperModel

# Load the model (downloads automatically the first time)
# "base" is fast and accurate enough for most testing on a laptop CPU

model = WhisperModel("base", device="cpu", compute_type="int8")

def stt(auido_path):
    print("Transcribing...")

    # Pass the path to your audio file
    segments, info = model.transcribe(auido_path, beam_size=5)

    print(f"Detected language: {info.language} (Probability: {info.language_probability:.2f})")
    print("-" * 30)

    # Reconstruct the sentence from the output segments
    user_text = "".join([segment.text for segment in segments]).strip()

    print(user_text)
    return user_text