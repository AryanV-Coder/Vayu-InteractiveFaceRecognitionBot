from deepface import DeepFace
import numpy as np

def front_face_embedding_value(img_path : str):
    try:
        # Generate embedding for image
        print(f"Generating embedding for {img_path}...")
        embedding_obj = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend='mtcnn'
        )
        
        # If multiple faces detected, get the largest one (front face)
        if len(embedding_obj) > 1:
            front_face = max(embedding_obj, 
                                key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
            front_face_embedding = np.array(front_face["embedding"], dtype=np.float32)
            print(f"Found {len(embedding_obj)} faces, using front one")
        else:
            front_face_embedding = np.array(embedding_obj[0]["embedding"], dtype=np.float32)
        
        # Normalize front face embedding
        front_face_embedding = front_face_embedding / np.linalg.norm(front_face_embedding)
        
        return front_face_embedding
        
    except Exception as e:
        print(f"Error in generating front face embedding using deepface: {e}")
        return []
