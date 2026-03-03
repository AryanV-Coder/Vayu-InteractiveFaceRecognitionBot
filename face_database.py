import os
import sqlite3
import numpy as np
import faiss
from deepface import DeepFace
from pathlib import Path


class FaceDatabase:
    def __init__(self, db_path="face_database.db", faiss_index_path="face_index.faiss"):
        """Initialize Face Database with FAISS and SQLite"""
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.embedding_dim = 512  # Facenet512 produces 512-dimensional embeddings
        
        # Initialize SQLite database
        self.init_sqlite_db()
        
        # Initialize or load FAISS index
        self.index = self.init_faiss_index()
    
    def init_sqlite_db(self):
        """Create SQLite database with person table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"SQLite database initialized: {self.db_path}")
    
    def init_faiss_index(self):
        """Initialize FAISS IndexIDMap with cosine similarity"""
        if os.path.exists(self.faiss_index_path):
            # Load existing index
            index = faiss.read_index(self.faiss_index_path)
            print(f"Loaded existing FAISS index: {self.faiss_index_path}")
        else:
            # Create new index with Inner Product (cosine similarity for normalized vectors)
            base_index = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIDMap(base_index)
            print("Created new FAISS index with cosine similarity")
        
        return index
    
    def add_faces_from_directory(self, image_dir):
        """
        Process all images in directory and add to database
        
        Args:
            image_dir: Path to directory containing face images
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            print(f"Error: Directory {image_dir} does not exist")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the current max person_id
        cursor.execute("SELECT MAX(person_id) FROM persons")
        result = cursor.fetchone()[0]
        next_id = (result + 1) if result is not None else 1
        
        embeddings_to_add = []
        ids_to_add = []
        names_to_add = []
        
        for img_file in image_files:
            try:
                print(f"Processing: {img_file.name}...")
                
                # Generate embedding using DeepFace with Facenet512
                embedding_obj = DeepFace.represent(
                    img_path=str(img_file),
                    model_name="Facenet512",
                    enforce_detection=True,
                    detector_backend='mtcnn'
                )
                
                # If multiple faces detected, get the largest one (front face)
                if len(embedding_obj) > 1:
                    largest_face = max(embedding_obj, 
                                     key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                    embedding = np.array(largest_face["embedding"], dtype=np.float32)
                    print(f"  Found {len(embedding_obj)} faces, using largest one")
                else:
                    embedding = np.array(embedding_obj[0]["embedding"], dtype=np.float32)
                
                # Normalize embedding for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                
                # Get name from filename (without extension)
                name = img_file.stem
                
                # Store for batch addition
                embeddings_to_add.append(embedding)
                ids_to_add.append(next_id)
                names_to_add.append((next_id, name))
                
                print(f"  ✓ Generated embedding for {name} (ID: {next_id})")
                next_id += 1
                
            except Exception as e:
                print(f"  ✗ Error processing {img_file.name}: {e}")
        
        # Add to FAISS index
        if embeddings_to_add:
            embeddings_array = np.array(embeddings_to_add, dtype=np.float32)
            ids_array = np.array(ids_to_add, dtype=np.int64)
            
            self.index.add_with_ids(embeddings_array, ids_array)
            print(f"\n✓ Added {len(embeddings_to_add)} embeddings to FAISS index")
            
            # Add to SQLite database
            cursor.executemany("INSERT INTO persons (person_id, name) VALUES (?, ?)", 
                             names_to_add)
            conn.commit()
            print(f"✓ Added {len(names_to_add)} persons to SQLite database")
            
            # Save FAISS index
            self.save_faiss_index()
        
        conn.close()
        print(f"\nTotal persons in database: {self.get_total_count()}")
    
    def search_face(self, image_path, k=5):
        """
        Search for similar faces in database
        
        Args:
            image_path: Path to query image
            k: Number of similar faces to return
        """
        try:
            # Generate embedding for query image
            print(f"Generating embedding for {image_path}...")
            embedding_obj = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet512",
                enforce_detection=True,
                detector_backend='mtcnn'
            )
            
            # If multiple faces detected, get the largest one (front face)
            if len(embedding_obj) > 1:
                largest_face = max(embedding_obj, 
                                 key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                query_embedding = np.array([largest_face["embedding"]], dtype=np.float32)
                print(f"Found {len(embedding_obj)} faces, using largest one")
            else:
                query_embedding = np.array([embedding_obj[0]["embedding"]], dtype=np.float32)
            
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search in FAISS (returns similarity scores, higher is better)
            similarities, ids = self.index.search(query_embedding, k)
            
            # Get names from SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            results = []
            print(f"\nTop {k} matches:")
            print("-" * 60)
            
            for i, (similarity, person_id) in enumerate(zip(similarities[0], ids[0]), 1):
                if person_id == -1:  # No match found
                    continue
                
                cursor.execute("SELECT name FROM persons WHERE person_id = ?", (int(person_id),))
                result = cursor.fetchone()
                
                if result:
                    name = result[0]
                    results.append({
                        "rank": i,
                        "person_id": int(person_id),
                        "name": name,
                        "similarity": float(similarity)
                    })
                    print(f"{i}. {name} (ID: {person_id}, Similarity: {similarity:.4f})")
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error searching face: {e}")
            return []
    
    def save_faiss_index(self):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, self.faiss_index_path)
        print(f"FAISS index saved to {self.faiss_index_path}")
    
    def get_total_count(self):
        """Get total number of persons in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM persons")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def list_all_persons(self):
        """List all persons in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT person_id, name FROM persons ORDER BY person_id")
        persons = cursor.fetchall()
        conn.close()
        
        print("\nAll persons in database:")
        print("-" * 60)
        for person_id, name in persons:
            print(f"ID: {person_id}, Name: {name}")
        
        return persons


def main():
    print("=" * 60)
    print("Face Database System (FAISS + SQLite + DeepFace)")
    print("=" * 60)
    
    # Initialize database
    face_db = FaceDatabase()
    
    while True:
        print("\nOptions:")
        print("1. Add faces from directory")
        print("2. Search for a face")
        print("3. List all persons")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            image_dir = input("Enter path to image directory: ").strip()
            face_db.add_faces_from_directory(image_dir)
        
        elif choice == "2":
            image_path = input("Enter path to query image: ").strip()
            k = input("Number of results to return (default 5): ").strip()
            k = int(k) if k.isdigit() else 5
            face_db.search_face(image_path, k)
        
        elif choice == "3":
            face_db.list_all_persons()
        
        elif choice == "4":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
