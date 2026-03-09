from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil

app = FastAPI(title="Face Data Upload API")

PERSONS_FOLDER = "persons"

# Ensure persons folder exists
os.makedirs(PERSONS_FOLDER, exist_ok=True)


@app.post("/upload_person")
async def upload_person(
    name: str = Form(...),
    description: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Upload person data with name, description, and images.
    
    Args:
        name: Person's name (used as folder name)
        description: Description of the person
        images: List of image files (expects 5 images)
    
    Returns:
        JSON response with status and saved details
    """
    
    # Validate number of images
    if len(images) != 5:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 5 images, received {len(images)}"
        )
    
    # Validate image file types
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    for image in images:
        file_ext = os.path.splitext(image.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.filename}. Allowed: {allowed_extensions}"
            )
    
    # Sanitize name for folder creation
    safe_name = name.strip().replace("/", "_").replace("\\", "_")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    # Create person directory
    person_dir = os.path.join(PERSONS_FOLDER, safe_name)
    
    if os.path.exists(person_dir):
        raise HTTPException(
            status_code=400, 
            detail=f"Person '{safe_name}' already exists"
        )
    
    try:
        os.makedirs(person_dir, exist_ok=True)
        
        # Save images
        saved_images = []
        for idx, image in enumerate(images, start=1):
            # Get original extension
            file_ext = os.path.splitext(image.filename)[1]
            
            # Create filename: image_1.jpg, image_2.png, etc.
            image_filename = f"image_{idx}{file_ext}"
            image_path = os.path.join(person_dir, image_filename)
            
            # Save image file
            with open(image_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
            
            saved_images.append(image_filename)
        
        # Save description to description.txt
        description_file = os.path.join(person_dir, "description.txt")
        with open(description_file, "w") as f:
            f.write(description)
        
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": f"Person '{safe_name}' uploaded successfully",
                "person_name": safe_name,
                "saved_images": saved_images,
                "description_saved": True,
                "folder_path": person_dir
            }
        )
    
    except Exception as e:
        # Cleanup on error
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error saving person data: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Face Data Upload API is running",
        "endpoints": ["/upload_person", "/persons"]
    }


@app.get("/persons")
async def list_persons():
    """List all registered persons"""
    if not os.path.exists(PERSONS_FOLDER):
        return {"persons": []}
    
    persons = [
        d for d in os.listdir(PERSONS_FOLDER) 
        if os.path.isdir(os.path.join(PERSONS_FOLDER, d))
    ]
    
    return {
        "total": len(persons),
        "persons": persons
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
