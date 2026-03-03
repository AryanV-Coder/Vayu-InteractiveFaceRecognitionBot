# Face Database System

A Python-based face recognition database system using FAISS vector database, SQLite, and DeepFace library.

## Overview

This system allows you to:
- Store face embeddings in a FAISS vector database
- Store person metadata (ID and name) in SQLite
- Search for similar faces using cosine similarity
- Build a face recognition database from a directory of images

## Technologies Used

- **DeepFace**: Face detection and embedding generation using Facenet512 model
- **FAISS**: Fast similarity search using cosine similarity (IndexFlatIP)
- **SQLite**: Metadata storage (person_id and name)
- **MTCNN**: Multi-task Cascaded Convolutional Networks for face detection

## How It Works

1. **Face Detection**: Detects faces in images using MTCNN (Multi-task Cascaded Convolutional Networks)
2. **Embedding Generation**: Converts faces to 512-dimensional embeddings using Facenet512
3. **Normalization**: Normalizes embeddings to unit vectors for cosine similarity
4. **Storage**:
   - FAISS IndexIDMap stores normalized embeddings with unique IDs
   - SQLite stores person_id (from FAISS) and name (from filename)
5. **Search**: Finds similar faces using cosine similarity (higher = more similar)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the script:
```bash
python face_database.py
```

### Menu Options

#### 1. Add faces from directory
- Provide a directory path containing face images (JPG, JPEG, PNG, BMP)
- Image filename becomes the person's name (e.g., `john_doe.jpg` → "john_doe")
- Processes all images and stores embeddings + metadata
- Automatically handles multiple faces (uses largest face)

**Example:**
```
Enter path to image directory: ./faces
```

Directory structure:
```
faces/
  ├── john_smith.jpg
  ├── jane_doe.png
  └── bob_wilson.jpg
```

#### 2. Search for a face
- Provide a query image path
- Specify number of results (default: 5)
- Returns top matches ranked by similarity score

**Example:**
```
Enter path to query image: ./test/unknown_person.jpg
Number of results to return: 3
```

#### 3. List all persons
- Displays all persons in the database with their IDs

#### 4. Exit
- Closes the application

## Database Files

- `face_database.db` - SQLite database with person records
- `face_index.faiss` - FAISS vector index with embeddings

## Understanding Similarity Scores

The system uses **cosine similarity** where:
- **0.7 - 1.0**: Same person (high confidence)
- **0.5 - 0.7**: Possibly same person (medium confidence)
- **0.0 - 0.5**: Different person (low confidence)

Higher scores = better match

## Features

### Automatic Largest Face Selection
When multiple faces are detected in an image:
- Calculates bounding box area for each face
- Selects the largest (typically the front/main face)
- Shows dimensions of all detected faces

### Debugging Output
- Shows number of faces detected
- Displays face dimensions (width × height)
- Indicates which face is being used

### Persistent Storage
- Database and index files are saved to disk
- Automatically loads existing data on startup
- Incremental additions supported

## Best Practices

1. **Image Quality**
   - Use clear, well-lit photos
   - Face should be clearly visible
   - Minimum face size: 100×100 pixels recommended

2. **Multiple Images Per Person**
   - Add 3-5 images per person for better accuracy
   - Use different angles and lighting conditions
   - Different facial expressions help

3. **Naming Convention**
   - Use descriptive filenames (becomes person name)
   - Avoid special characters
   - Use underscores instead of spaces (e.g., `john_smith.jpg`)

4. **Database Maintenance**
   - Rebuild database if search accuracy decreases
   - Delete old files to start fresh:
     ```bash
     rm face_database.db face_index.faiss
     ```

## Example Workflow

```bash
# 1. Prepare images
mkdir faces
# Add your face images to the faces/ directory

# 2. Run the database system
python face_database.py

# 3. Choose option 1: Add faces from directory
# Enter path: ./faces

# 4. Choose option 3: List all persons
# Verify your faces were added

# 5. Choose option 2: Search for a face
# Provide a test image path
# View similarity scores
```

## Troubleshooting

### "Face could not be detected"
- Ensure face is clearly visible and well-lit
- Try `enforce_detection=False` in code (line 93)
- Check if image has sufficient resolution

### "No images found in directory"
- Verify directory path is correct
- Check image file extensions (.jpg, .jpeg, .png, .bmp)
- Ensure images are not in subdirectories

### Poor recognition accuracy
- Add more images per person (3-5 recommended)
- Use consistent lighting and angles
- Ensure images are high quality
- Rebuild database with better images

### Multiple faces detected (false positives)
- System automatically selects largest face
- Check debug output to see all detected faces
- Consider using higher quality images

## Technical Details

### Embedding Dimension
- Facenet512 produces 512-dimensional vectors
- Normalized to unit length for cosine similarity

### FAISS Index Type
- `IndexFlatIP`: Inner product (cosine similarity for normalized vectors)
- Exact search (no approximation)
- Wrapped with `IndexIDMap` for ID management

### SQLite Schema
```sql
CREATE TABLE persons (
    person_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
)
```

## Limitations

- Requires TensorFlow and compatible Python version (3.9-3.12)
- First run downloads Facenet512 model (~300-400 MB)
- Face detection can fail with poor lighting or extreme angles
- Single threaded (processes images sequentially)

## Future Enhancements

- Batch processing for faster image addition
- Support for video input
- Web interface
- Multiple face tracking in single image
- Face verification API endpoint
