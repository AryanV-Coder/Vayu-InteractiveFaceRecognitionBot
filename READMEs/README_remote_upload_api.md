# Remote Upload API

FastAPI service for remotely uploading person data (name, description, and images) to the face recognition system.

## Requirements
- Python 3.11
- Install dependencies: `pip install -r requirements.txt`

## How to Run

```bash
python3.11 remote_upload_api.py
```

API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Upload Person Data
**POST** `/upload_person`

Upload a person's data including name, description, and 5 images.

**Parameters:**
- `name` (form): Person's name (will be used as folder name)
- `description` (form): Description text
- `images` (files): Exactly 5 image files

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/upload_person" \
  -F "name=John Doe" \
  -F "description=Software engineer who loves coding" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  -F "images=@image4.jpg" \
  -F "images=@image5.jpg"
```

**Response:**
```json
{
  "status": "success",
  "message": "Person 'John Doe' uploaded successfully",
  "person_name": "John Doe",
  "saved_images": ["image_1.jpg", "image_2.jpg", "image_3.jpg", "image_4.jpg", "image_5.jpg"],
  "description_saved": true,
  "folder_path": "persons/John Doe"
}
```

### 2. List All Persons
**GET** `/persons`

Get list of all registered persons.

**Example:**
```bash
curl http://localhost:8000/persons
```

### 3. Health Check
**GET** `/`

Check if API is running.

## Using the Python Client

Example code in `api_client_example.py`:

```python
from api_client_example import upload_person

upload_person(
    name="John Doe",
    description="Software engineer who loves coding",
    image_paths=[
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
        "path/to/image4.jpg",
        "path/to/image5.jpg"
    ]
)
```

## What It Does

1. **Creates Directory**: Creates a folder in `persons/` with the person's name
2. **Saves Images**: Saves 5 images as `image_1.jpg`, `image_2.jpg`, etc.
3. **Saves Description**: Writes description to `description.txt`

## Supported Image Formats
- .jpg, .jpeg
- .png
- .gif
- .bmp
- .tiff
- .webp

## Notes
- Exactly 5 images are required per person
- Person names must be unique (no duplicates)
- Special characters in names are sanitized for folder creation
