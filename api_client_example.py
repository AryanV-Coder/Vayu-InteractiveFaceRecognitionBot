"""
Example client for testing the remote upload API
"""
import requests

API_URL = "http://localhost:8000"

def upload_person(name: str, description: str, image_paths: list):
    """
    Upload person data to the API
    
    Args:
        name: Person's name
        description: Description text
        image_paths: List of 5 image file paths
    """
    
    if len(image_paths) != 5:
        print(f"Error: Expected 5 images, got {len(image_paths)}")
        return
    
    # Prepare form data
    data = {
        'name': name,
        'description': description
    }
    
    # Prepare files for upload
    files = []
    for image_path in image_paths:
        files.append(
            ('images', open(image_path, 'rb'))
        )
    
    try:
        # Send POST request
        response = requests.post(
            f"{API_URL}/upload_person",
            data=data,
            files=files
        )
        
        # Close all file handles
        for _, file_obj in files:
            file_obj.close()
        
        # Check response
        if response.status_code == 201:
            print("✓ Upload successful!")
            print(response.json())
        else:
            print(f"✗ Upload failed: {response.status_code}")
            print(response.json())
    
    except Exception as e:
        print(f"Error: {e}")


def list_persons():
    """List all registered persons"""
    try:
        response = requests.get(f"{API_URL}/persons")
        if response.status_code == 200:
            data = response.json()
            print(f"\nTotal persons: {data['total']}")
            for person in data['persons']:
                print(f"  - {person}")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage
    print("Remote Upload API Client\n")
    
    # Example: Upload a person
    # upload_person(
    #     name="John Doe",
    #     description="Software engineer who loves coding",
    #     image_paths=[
    #         "path/to/image1.jpg",
    #         "path/to/image2.jpg",
    #         "path/to/image3.jpg",
    #         "path/to/image4.jpg",
    #         "path/to/image5.jpg"
    #     ]
    # )
    
    # List all persons
    list_persons()
