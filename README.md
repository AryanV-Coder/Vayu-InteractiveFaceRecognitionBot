# 🤖 Vayu - Interactive Face Recognition Bot

> An interactive AI bot that recognizes faces in real-time and holds personalized voice conversations — built for **Converge**, the annual fest of **Jaypee Institute of Information Technology (JIIT)**, by **CICR**.

---

## 🎯 What It Does

Vayu is a live, camera-powered AI assistant (codename: **J.A.R.V.I.S. Protocol**) deployed at a fest booth. It does two things simultaneously:

1. **Face Recognition** — Identifies people standing in front of the camera using a FAISS vector database of pre-registered face embeddings (powered by DeepFace + Facenet512).
2. **Interactive Voice Chatbot** — Listens to the user's voice, transcribes it, generates a witty AI response personalized to the recognized person, and speaks it back — all in real-time with streaming audio playback.

If a person is not in the database, Vayu treats them as "Unknown" and playfully asks them to come closer.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  live_face_recognition.py                │
│                    (Main Entry Point)                    │
├──────────────┬──────────────────┬────────────────────────┤
│  Camera Loop │   Audio Thread   │  Face Recognition      │
│  (OpenCV)    │   (Silero VAD)   │  (ProcessPoolExecutor) │
└──────┬───────┴────────┬─────────┴──────────┬─────────────┘
       │                │                    │
       │                ▼                    ▼
       │     ┌──────────────────┐   ┌──────────────────────┐
       │     │ main_pipeline.py │   │ face_recognition_     │
       │     │  STT → LLM → TTS│   │ logic.py              │
       │     └───┬────┬────┬───┘   │  FAISS + SQLite       │
       │         │    │    │       └──────────────────────  ┘
       │         ▼    ▼    ▼
       │   ┌─────┐ ┌────┐ ┌─────┐
       │   │ STT │ │LLM │ │ TTS │
       │   └─────┘ └────┘ └─────┘
       │   Sarvam   Groq   Sarvam
       │   saaras   llama  bulbul
       │   :v3      3.3-70b :v3
       │            versatile
       ▼
   Camera Feed
   (cv2.imshow)
```

---

## 📁 Project Structure

```
Vayu_InteractiveFaceRecognitionBot/
├── live_face_recognition.py        # 🚀 Main entry point — camera, audio, face recognition
├── self_data_upload.py             # 📸 CLI tool to register new faces into the database
├── whisper_stt.py                  # 🎙️ Local Whisper STT (alternate, offline)
│
├── utils/
│   ├── main_pipeline.py            # 🔗 Orchestrator: STT → LLM → TTS
│   ├── sarvam_stt.py               # 🎙️ Speech-to-Text via Sarvam AI (saaras:v3)
│   ├── sarvam_tts.py               # 🔊 Text-to-Speech via Sarvam AI (bulbul:v3) with streaming
│   ├── conversational_llm.py       # 🧠 Groq LLM with personalized system prompts
│   ├── face_recognition_logic.py   # 🕵️ Face recognition: FAISS lookup + similarity threshold
│   ├── faiss_db.py                 # 📦 FAISS vector database (cosine similarity, Facenet512)
│   ├── sqlite_db.py                # 🗃️ SQLite database for person metadata (name, description)
│   └── deepface_recognition.py     # 🧬 DeepFace + MTCNN face embedding extraction
│
├── READMEs/
│   └── STREAMING_PIPELINE.md       # 📖 Deep-dive: when to stream STT/LLM/TTS and why
│
├── face_database.db                # SQLite database file (auto-generated)
├── face_index.faiss                # FAISS index file (auto-generated)
├── requirements.txt                # Python dependencies
└── .env                            # API keys (SARVAM_API_KEY, GROQ_API_KEY)
```

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Face Embeddings** | DeepFace + Facenet512 + MTCNN | Generate 512-dim face vectors from images |
| **Vector Search** | FAISS (IndexFlatIP + IndexIDMap) | Cosine similarity search for face matching |
| **Metadata Store** | SQLite | Store person name & description linked to FAISS IDs |
| **Voice Activity Detection** | Silero VAD (PyTorch) | Detect when user starts/stops speaking |
| **Speech-to-Text** | Sarvam AI (saaras:v3) | Transcribe Hindi/English/Hinglish audio |
| **LLM** | Groq (LLaMA 3.3 70B Versatile) | Generate witty, personalized responses |
| **Text-to-Speech** | Sarvam AI (bulbul:v3) | Stream audio chunks via WebSocket |
| **Audio Playback** | ffplay (FFmpeg) | Real-time streaming playback via subprocess pipe |
| **Camera** | OpenCV | Live video feed and frame capture |
| **Parallelism** | ProcessPoolExecutor (spawn) | Offload face recognition to a separate process |

---

## 🚀 Setup

### 1. Clone & Install

```bash
git clone <repo-url>
cd Vayu_InteractiveFaceRecognitionBot

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install System Dependencies (macOS)

```bash
brew install ffmpeg portaudio
```

### 3. Set Up API Keys

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Register Faces

Organize face images in the following folder structure:

```
persons/
├── Aryan/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── description.txt      # "CS student, loves AI and robotics"
├── Rahul/
│   ├── photo1.jpg
│   └── description.txt      # "Music club president, plays guitar"
```

Then run:

```bash
python self_data_upload.py
# Enter folder name: persons
```

This will:
- Extract Facenet512 embeddings from each image using DeepFace + MTCNN.
- Store the normalized embeddings in `face_index.faiss`.
- Store the person's name and description in `face_database.db` (SQLite).

### 5. Run the Bot

```bash
python live_face_recognition.py
```

Press **`q`** to quit.

---

## 🎛️ Hardware Setup (Fest Booth)

The bot is designed to work with external peripherals out of the box. **No code changes are needed** — just configure your macOS Sound settings before running.

| Peripheral | What To Do |
|---|---|
| **USB Webcam** | Plug it in. It will be auto-detected as `cv2.VideoCapture(0)`. If your MacBook's built-in camera takes priority, change `0` to `1` on line 207 of `live_face_recognition.py`. |
| **Bluetooth Speaker** (audio output) | Pair it, then go to **System Settings → Sound → Output** and select it. `ffplay` routes all audio through the system default output. |
| **External Mic via 3.5mm Jack** (audio input) | Plug it in, then go to **System Settings → Sound → Input** and select "External Microphone" or "Line In". PyAudio uses the system default input. |

> **Tip:** If the speaker is Bluetooth (wireless) and the mic is wired (jack), there should be no echo/feedback loop. The bot also mutes the mic during its own speech playback as an extra safeguard.

---

## 🔊 How Audio Streaming Works

The TTS engine uses a **hybrid streaming model** for ultra-low latency:

1. The full LLM response text is sent to Sarvam's WebSocket API in one shot (for better intonation).
2. Sarvam streams back audio chunks in base64 as they are generated.
3. Each chunk is **immediately decoded and piped** into an `ffplay` subprocess buffer.
4. Audio playback begins within **~0.4 seconds** of the first chunk arriving — the bot starts talking before the full audio has been generated!

> For a detailed breakdown of streaming decisions across **STT**, **LLM**, and **TTS**, see [`READMEs/STREAMING_PIPELINE.md`](READMEs/STREAMING_PIPELINE.md).

---

## 🔑 Key Configuration

| Parameter | File | Default | Description |
|---|---|---|---|
| `similarity_score` threshold | `utils/face_recognition_logic.py` | `0.60` | Minimum cosine similarity to accept a face match |
| `silence_threshold` | `live_face_recognition.py` | `31` (~1.0s) | Consecutive silent chunks before speech is considered finished |
| `process_interval` | `live_face_recognition.py` | `5` seconds | How often the camera submits a frame for face recognition |
| `speech_threshold` | `live_face_recognition.py` | `0.5` | VAD probability threshold to classify audio as speech |

---

## 🛡️ Error Handling

All external API calls (Groq LLM, Sarvam STT, Sarvam TTS) include explicit rate limit detection:

- **429 Too Many Requests** → Prints a highly visible `🚨 RATE LIMIT EXCEEDED` banner and **immediately terminates** the program to prevent further API abuse.
- **Other API errors** → Logged with `❌` and the pipeline gracefully skips that interaction without crashing.

---

## 🔄 Switching STT Engines

The project supports two STT backends. To switch between them, edit `utils/main_pipeline.py`:

```python
# Current (Sarvam AI — cloud-based, supports Hindi/Hinglish):
from utils.sarvam_stt import stt

# Alternative (Whisper — local, offline):
# from whisper_stt import stt
```
