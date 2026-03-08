# Self Data Upload - Face Database Builder

## Overview
This script processes face images from organized folders and builds a searchable face database using FAISS vector search and SQLite.

## Requirements
- **Python 3.11** (Required)
- Install dependencies: `pip install -r requirements.txt`

## How to Run
```bash
python3.11 self_data_upload.py
```

When prompted, enter the folder name containing person subfolders (e.g., `persons`).

## Folder Structure
Your input folder should be organized as:
```
persons/
├── person_name_1/
│   ├── image1.jpg
│   ├── image2.png
│   └── description.txt
├── person_name_2/
│   ├── photo1.jpg
│   └── description.txt
```

## How It Works

### 1. **Input Processing** (`self_data_upload.py`)
   - Prompts for folder name
   - Iterates through all subfolders
   - Collects image paths and reads `description.txt` from each subfolder

### 2. **Face Embedding Generation** (`utils/deepface_recognition.py`)
   - Uses DeepFace with Facenet512 model
   - Detects faces and extracts 512-dimensional embeddings
   - Normalizes embeddings for cosine similarity

### 3. **Vector Storage** (`utils/faiss_db.py`)
   - Stores face embeddings in FAISS index
   - Each embedding gets a unique `faiss_id`
   - Saves index to `face_index.faiss`

### 4. **Metadata Storage** (`utils/sqlite_db.py`)
   - Stores person information (name, description) in SQLite
   - Links FAISS IDs to person records
   - Maintains database in `face_database.db`

## Output Files
- `face_index.faiss` - Vector embeddings for face search
- `face_database.db` - Person metadata and ID mappings

## Notes
- Each subfolder name becomes the person's name in the database
- Supported image formats: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
- Multiple face images per person are supported
