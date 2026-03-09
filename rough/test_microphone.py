import pyaudio
import numpy as np

def test_microphone():
    """Test if microphone is working and detect audio levels"""
    
    audio = pyaudio.PyAudio()
    
    print("=== Microphone Test ===\n")
    
    # List all audio devices
    print("Available audio input devices:")
    print("-" * 60)
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # Input device
            print(f"Device {i}: {info['name']}")
            print(f"  Channels: {info['maxInputChannels']}")
            print(f"  Sample Rate: {int(info['defaultSampleRate'])} Hz")
            print()
    
    # Get default input device
    default_input = audio.get_default_input_device_info()
    print(f"Default input device: {default_input['name']}\n")
    
    # Test recording
    RATE = 16000
    CHUNK = 4000
    
    print("🎤 Testing audio input for 5 seconds...")
    print("Please speak into your microphone!\n")
    
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    stream.start_stream()
    
    try:
        for i in range(int(RATE / CHUNK * 5)):  # 5 seconds
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert to numpy array to measure volume
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            
            # Visual volume meter
            bar_length = int(volume / 100)
            bar = "█" * bar_length
            
            print(f"Volume: {volume:6.0f} {bar}", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nTest stopped")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    print("\n\n✓ Microphone test complete!")
    print("\nIf you saw volume changes when speaking, your microphone works.")
    print("If volume stayed at 0, check:")
    print("  1. System Preferences > Security & Privacy > Microphone")
    print("  2. Grant Terminal/Python permission to access microphone")
    print("  3. Check if microphone is muted or volume is too low")


if __name__ == "__main__":
    test_microphone()
