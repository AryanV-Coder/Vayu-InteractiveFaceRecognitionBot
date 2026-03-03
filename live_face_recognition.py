import cv2
import numpy as np
from deepface import DeepFace
from face_database import FaceDatabase
import time


class LiveFaceRecognition:
    def __init__(self, db_path="face_database.db", faiss_index_path="face_index.faiss"):
        """Initialize live face recognition system"""
        self.face_db = FaceDatabase(db_path, faiss_index_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.last_recognition_time = 0
        self.recognition_interval = 2  # Recognize every 2 seconds
        self.current_recognition = {}
        
    def recognize_face(self, frame):
        """Recognize face in the frame using DeepFace and FAISS"""
        try:
            # Save frame temporarily
            temp_path = "temp_recognition.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Generate embedding
            embedding_obj = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend='mtcnn'
            )
            
            if not embedding_obj:
                return None
            
            # Get largest face if multiple detected
            if len(embedding_obj) > 1:
                largest_face = max(embedding_obj, 
                                 key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                query_embedding = np.array([largest_face["embedding"]], dtype=np.float32)
            else:
                query_embedding = np.array([embedding_obj[0]["embedding"]], dtype=np.float32)
            
            # Normalize query embedding for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search in FAISS (returns similarity scores, higher is better)
            similarities, ids = self.face_db.index.search(query_embedding, 1)
            
            # Always get the closest match
            person_id = int(ids[0][0])
            similarity = float(similarities[0][0])
            
            # Get name from SQLite
            import sqlite3
            conn = sqlite3.connect(self.face_db.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM persons WHERE person_id = ?", (person_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                threshold = 0.5  # Cosine similarity threshold (higher is better)
                is_match = similarity > threshold
                
                return {
                    "name": result[0],
                    "person_id": person_id,
                    "similarity": similarity,
                    "confidence": similarity * 100,
                    "is_match": is_match,
                    "threshold": threshold
                }
            
            return None
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return None
    
    def start_recognition(self):
        """Start live face recognition from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("=" * 60)
        print("Live Face Recognition System")
        print("=" * 60)
        print("Press 'q' to quit")
        print("Press 'r' to force recognition")
        print()
        
        # Check if database has faces
        if self.face_db.get_total_count() == 0:
            print("WARNING: No faces in database!")
            print("Please add faces first using face_database.py")
            print()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            
            current_time = time.time()
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Perform recognition periodically
                if current_time - self.last_recognition_time > self.recognition_interval:
                    # Extract face region with some padding
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        result = self.recognize_face(face_roi)
                        self.current_recognition = result if result else {}
                        self.last_recognition_time = current_time
                
                # Display recognition result
                if self.current_recognition:
                    name = self.current_recognition['name']
                    confidence = self.current_recognition['confidence']
                    similarity = self.current_recognition['similarity']
                    is_match = self.current_recognition['is_match']
                    threshold = self.current_recognition['threshold']
                    
                    # Choose color based on match status
                    color = (0, 255, 0) if is_match else (0, 165, 255)  # Green if match, Orange if not
                    
                    # Display name with match status
                    match_status = "MATCH" if is_match else "NO MATCH"
                    label = f"{name} - {match_status}"
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Display similarity and threshold
                    info1 = f"Similarity: {similarity:.4f} (Threshold: {threshold:.2f})"
                    cv2.putText(frame, info1, (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Display confidence
                    info2 = f"Confidence: {confidence:.1f}%"
                    cv2.putText(frame, info2, (x, y + h + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    # Unknown face
                    cv2.putText(frame, "Unknown", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display status
            status = f"Faces detected: {len(faces)}"
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit | 'r' to recognize", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Live Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Force recognition when 'r' is pressed
            if key == ord('r'):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        print("Recognizing face...")
                        result = self.recognize_face(face_roi)
                        if result:
                            match_status = "MATCH" if result['is_match'] else "NO MATCH"
                            print(f"Closest match: {result['name']} ({match_status})")
                            print(f"Similarity: {result['similarity']:.4f} (Threshold: {result['threshold']:.2f})")
                            print(f"Confidence: {result['confidence']:.1f}%")
                            self.current_recognition = result
                        else:
                            print("Face not recognized")
                            self.current_recognition = {}
                else:
                    print("No face detected")
            
            # Quit when 'q' is pressed
            elif key == ord('q'):
                print("Exiting...")
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("Live Face Recognition System")
    print("=" * 60)
    print()
    
    # Initialize recognition system
    recognizer = LiveFaceRecognition()
    
    # Check if database has faces
    total_faces = recognizer.face_db.get_total_count()
    
    if total_faces == 0:
        print("WARNING: No faces found in database!")
        print()
        response = input("Do you want to add faces first? (y/n): ").strip().lower()
        if response == 'y':
            print("Please run: python face_database.py")
            return
    else:
        print(f"Database contains {total_faces} person(s)")
        print()
    
    # Start live recognition
    recognizer.start_recognition()


if __name__ == "__main__":
    main()
