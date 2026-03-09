# Vosk Speech-to-Text Setup

## What to Do:

### 1. Install Vosk
```bash
pip install vosk
```

### 2. Download a Language Model

**For English (Small - 40MB):**
```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 model
```

**OR using curl:**
```bash
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 model
```

**For better accuracy (larger model - 1.8GB):**
```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
mv vosk-model-en-us-0.22 model
```

**For Hindi:**
```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip
unzip vosk-model-small-hi-0.22.zip
mv vosk-model-small-hi-0.22 model
```

More models: https://alphacephei.com/vosk/models

### 3. Run the Script
```bash
python3.11 vosk_stt.py
```

## Features

✓ **Offline** - No internet or API key needed
✓ **Real-time** - Instant transcription as you speak
✓ **Free** - Completely open source
✓ **Fast** - Lightweight and efficient
✓ **Partial results** - See transcription while speaking

## Directory Structure
```
Vayu_InteractiveFaceRecognitionBot/
├── vosk_stt.py
└── model/              # Vosk model folder (after download)
    ├── am/
    ├── conf/
    ├── graph/
    └── ...
```

## Usage

The script will continuously listen and print transcriptions. Press **Ctrl+C** to stop.

## Custom Model Path

If your model is in a different location:
```python
continuous_speech_to_text(model_path="path/to/your/model")
```
