from utils.faiss_db import FAISS_VectorDB
from utils.sqlite_db import SQLite_SqlDB
import os

sql_db = SQLite_SqlDB()
vector_db = FAISS_VectorDB()

# Get folder name from user
folder_name = input("Enter the folder name: ")

# Check if the folder exists
if not os.path.exists(folder_name):
    print(f"Error: Folder '{folder_name}' does not exist!")
else:
    # Get all subfolders in the given folder
    subfolders = [f for f in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, f))]
    
    # Loop through all subfolders
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_name, subfolder)
        
        # Print subfolder name
        print(f"\nProcessing subfolder: {subfolder}")
        
        # Initialize array to store image paths
        image_paths = []
        
        # Get all files in the subfolder
        files = os.listdir(subfolder_path)
        
        # Extract image paths (common image extensions)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        for file in files:
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in image_extensions):
                image_path = os.path.join(subfolder_path, file)
                image_paths.append(image_path)
        
        print(f"Found {len(image_paths)} image(s): {image_paths}")
        
        # Extract description from text file
        description = ""
        description_file = os.path.join(subfolder_path, "description.txt")
        
        if os.path.exists(description_file):
            with open(description_file, 'r') as f:
                description = f.read().strip()
            print(f"Description: {description}")
        else:
            print("No description.txt found in this subfolder")

        next_id = sql_db.maxFaissId() + 1

        faiss_ids = vector_db.insertData(image_paths,next_id)
        sql_db.insertDataIntoTable(faiss_ids,subfolder,description)

    vector_db.save_faiss_index()
