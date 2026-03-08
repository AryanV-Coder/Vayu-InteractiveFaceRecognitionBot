import faiss
import os
import numpy as np
from utils.deepface_recognition import front_face_embedding_value

class FAISS_VectorDB():
    def __init__(self, faiss_index_path="face_index.faiss"):
        self.faiss_index_path = faiss_index_path
        self.embedding_dim = 512  # Facenet512 produces 512-dimensional embeddings
        
        # Initialize or load FAISS index
        self.index = self.init_faiss_index()

    def init_faiss_index(self):
        """Initialize FAISS IndexIDMap with cosine similarity"""
        if os.path.exists(self.faiss_index_path):
            # Load existing index
            index = faiss.read_index(self.faiss_index_path)
            print(f" ✓ Loaded existing FAISS index: {self.faiss_index_path}")
        else:
            # Create new index with Inner Product (cosine similarity for normalized vectors)
            base_index = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIDMap(base_index)
            print(" ✓ Created new FAISS index with cosine similarity")
        
        return index
    
    def insertData(self, image_paths : list, next_id : int):
        ids = []
        embeddings = []
        for image_path in image_paths:
            embedding = front_face_embedding_value(image_path)
        
            # Skip if embedding failed
            if embedding is None or len(embedding) == 0:
                print(f"  ✗ Skipping {image_path} due to error")
                continue
                
            ids.append(next_id)
            embeddings.append(embedding)
            next_id += 1

        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            ids_array = np.array(ids, dtype=np.int64)

        self.index.add_with_ids(embeddings_array,ids_array)
        print(f"\n ✓ Added embeddings to FAISS index. Last faiss_id = {next_id-1}")
        return ids

    def retrieveData(self):
        pass

    def save_faiss_index(self):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, self.faiss_index_path)
        print(f" ✓ FAISS index saved to {self.faiss_index_path}")
