# Live Face Recognition System

A real-time face recognition system using webcam feed, OpenCV, DeepFace, and FAISS for instant face matching.

## Overview

This system provides:
- Real-time face detection from webcam
- Automatic face recognition against stored database
- Live similarity scoring and confidence display
- Visual feedback with match/no-match indicators
- Manual recognition trigger

## Prerequisites

You must first create a face database using `face_database.py` before running this system.

## Technologies Used

- **OpenCV**: Real-time video capture and face detection (Haar Cascade for speed)
- **DeepFace**: Face embedding generation (Facenet512 model)
- **MTCNN**: Accurate face detection for recognition
- **FAISS**: Fast similarity search
- **SQLite**: Person metadata retrieval
- **NumPy**: Vector operations and normalization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Create Face Database
First, add faces to the database:
```bash
python face_database.py
# Choose option 1: Add faces from directory
```

### Step 2: Run Live Recognition
```bash
python live_face_recognition.py
```

## Controls

- **'r'** - Force immediate recognition
- **'q'** - Quit application

## How It Works

1. **Video Capture**: Opens webcam (camera index 0)
2. **Face Detection**: Uses Haar Cascade to detect faces in real-time (fast initial detection)
3. **Periodic Recognition**: 
   - Automatically recognizes faces every 2 seconds
   - Extracts face region from frame
   - Generates embedding using Facenet512
   - Searches FAISS database for closest match
4. **Display Results**: Shows name, match status, similarity, and confidence

## On-Screen Display

### Face Detected and Recognized (MATCH)
```
┌─────────────────────────┐
│   John Smith - MATCH    │ ← Green text
│                         │
│  [Green rectangle box]  │
│                         │
│ Similarity: 0.8523      │ ← Above threshold
│ (Threshold: 0.50)       │
│ Confidence: 85.2%       │
└─────────────────────────┘
```

### Face Detected but Not Recognized (NO MATCH)
```
┌─────────────────────────┐
│  Jane Doe - NO MATCH    │ ← Orange text
│                         │
│  [Green rectangle box]  │
│                         │
│ Similarity: 0.4123      │ ← Below threshold
│ (Threshold: 0.50)       │
│ Confidence: 41.2%       │
└─────────────────────────┘
```

### Unknown Face
```
┌─────────────────────────┐
│      Unknown            │ ← Red text
│                         │
│  [Green rectangle box]  │
└─────────────────────────┘
```

## Recognition Parameters

### Threshold: 0.5
- **Above 0.5**: MATCH (recognized as stored person)
- **Below 0.5**: NO MATCH (not recognized)

### Similarity Interpretation
- **0.9 - 1.0**: Excellent match (very high confidence)
- **0.7 - 0.9**: Good match (high confidence)
- **0.5 - 0.7**: Moderate match (medium confidence)
- **0.3 - 0.5**: Poor match (low confidence)
- **0.0 - 0.3**: Very poor match (no confidence)

### Recognition Interval
- Default: 2 seconds
- Prevents excessive computation
- Adjustable in code (line 13)

## Features

### Automatic Recognition
- Runs every 2 seconds automatically
- No user interaction required
- Results persist on screen until next recognition

### Manual Recognition
- Press 'r' to force immediate recognition
- Useful for verifying results
- Prints detailed info to console

### Visual Feedback
- **Green text**: Face recognized (MATCH)
- **Orange text**: Face detected but not recognized (NO MATCH)
- **Red text**: Unknown face (no database match)
- Green rectangle around detected faces

### Console Output
When pressing 'r':
```
Recognizing face...
Closest match: John Smith (MATCH)
Similarity: 0.8523 (Threshold: 0.50)
Confidence: 85.2%
```

## Configuration

### Adjust Recognition Threshold
Edit line 42 in the code:
```python
threshold = 0.5  # Increase for stricter matching, decrease for looser
```

**Recommendations:**
- **0.4**: Looser matching (more false positives)
- **0.5**: Balanced (default, recommended)
- **0.6**: Stricter matching (fewer false positives)

### Adjust Recognition Interval
Edit line 13 in the code:
```python
self.recognition_interval = 2  # Time in seconds
```

**Recommendations:**
- **1 second**: Fast recognition (higher CPU usage)
- **2 seconds**: Balanced (default)
- **3-5 seconds**: Lower CPU usage

### Minimum Face Size
Edit line 115 in the code:
```python
faces = self.face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
)
```

Change `minSize=(100, 100)` to detect smaller/larger faces.

## Performance Tips

1. **Good Lighting**
   - Face should be well-lit
   - Avoid backlighting
   - Natural or consistent artificial light works best

2. **Camera Position**
   - Face camera directly
   - Keep face within frame
   - Maintain 2-3 feet distance

3. **Face Quality**
   - Remove glasses if stored images don't have them
   - Maintain consistent facial expression
   - Ensure face is clear and unobstructed

4. **Database Quality**
   - Add 3-5 images per person to database
   - Use various angles and lighting
   - Include different expressions

## Troubleshooting

### "WARNING: No faces found in database!"
- Run `face_database.py` first
- Add faces using option 1
- Ensure database files exist

### Face not being detected
- Improve lighting
- Move closer to camera
- Face camera directly
- Ensure face is not tilted too much

### Face detected but not recognized
- Check similarity score
- If close to threshold (0.45-0.55), add more training images
- Adjust threshold if needed
- Ensure stored images are good quality

### Low similarity scores (< 0.4)
- Person might not be in database
- Add more images of the person
- Ensure current conditions match stored images
- Check if glasses/hat are affecting detection

### High CPU usage
- Increase `recognition_interval` to 3-5 seconds
- Close other applications
- Reduce video resolution (advanced)

### Camera not opening
- Check if another application is using camera
- Verify camera permissions (System Preferences → Security & Privacy → Camera)
- Try different camera index (change 0 to 1 in code)

## Example Workflow

```bash
# 1. Create database (first time only)
python face_database.py
# Add faces from directory

# 2. Start live recognition
python live_face_recognition.py

# 3. Position yourself in front of camera
# Wait for automatic recognition (2 seconds)

# 4. Press 'r' to force recognition
# Check console for detailed results

# 5. Press 'q' to quit
```

## Technical Details

### Face Detection
- **Initial Detection**: Haar Cascade (frontal face) - fast for video processing
- **Recognition Detection**: MTCNN - accurate for embedding generation
- **Speed**: Fast initial detection, accurate recognition
- **Minimum size**: 100×100 pixels
- **Parameters**: scaleFactor=1.1, minNeighbors=5

### Face Recognition
- **Model**: Facenet512 (512-dimensional embeddings)
- **Detector**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Similarity**: Cosine similarity (normalized vectors)
- **Search**: FAISS IndexFlatIP (exact search)
- **Latency**: ~200-500ms per recognition

### Video Processing
- **Mirror effect**: Enabled (more natural viewing)
- **Frame rate**: Limited by processing time
- **Resolution**: Default camera resolution

## Database Integration

Connects to:
- `face_database.db` - SQLite database
- `face_index.faiss` - FAISS index

Must exist in the same directory.

## Limitations

- Single face recognition at a time (uses largest face)
- Requires good lighting conditions
- Face must be relatively front-facing
- Recognition latency of 200-500ms
- Requires pre-built database

## Advanced Customization

### Change Camera Source
```python
cap = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc. for other cameras
```

### Disable Mirror Effect
```python
# Comment out line 108:
# frame = cv2.flip(frame, 1)
```

### Change Face Color
```python
# Line 122 - Change rectangle color (BGR format):
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
# (0, 255, 0) = Green
# (255, 0, 0) = Blue
# (0, 0, 255) = Red
```

## Future Enhancements

- Multi-face recognition
- Face tracking across frames
- Recognition history logging
- Performance statistics
- GUI configuration panel
- Remote camera support
