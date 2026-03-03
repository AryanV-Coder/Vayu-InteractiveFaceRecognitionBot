import cv2
import numpy as np
from deepface import DeepFace
import os

class FaceEmbedding:
    def __init__(self, model_name="Facenet512"):
        """
        Initialize Face Embedding system
        
        Args:
            model_name: DeepFace model to use for embeddings
                       Options: VGG-Face, Facenet, Facenet512, OpenFace, 
                                DeepFace, DeepID, ArcFace, Dlib, SFace
        """
        self.model_name = model_name
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def capture_and_embed(self, save_image=True, output_dir="captured_faces"):
        """
        Capture face from webcam and generate embedding
        
        Args:
            save_image: Whether to save the captured face image
            output_dir: Directory to save captured images
        """
        # Create output directory if it doesn't exist
        if save_image and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None
        
        print("Press 'c' to capture face and generate embedding")
        print("Press 'q' to quit")
        
        embedding = None
        captured_image = None
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Face Capture - Press C to capture, Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture face when 'c' is pressed
            if key == ord('c'):
                if len(faces) > 0:
                    print("Capturing face and generating embedding...")
                    
                    # Save the captured frame
                    if save_image:
                        timestamp = cv2.getTickCount()
                        image_path = os.path.join(output_dir, f"face_{timestamp}.jpg")
                        cv2.imwrite(image_path, frame)
                        print(f"Image saved to: {image_path}")
                        captured_image = image_path
                    else:
                        # Save temporary image for embedding generation
                        temp_path = "temp_face.jpg"
                        cv2.imwrite(temp_path, frame)
                        captured_image = temp_path
                    
                    try:
                        # Generate embedding using DeepFace
                        print(f"Generating embedding using {self.model_name}...")
                        embedding_obj = DeepFace.represent(
                            img_path=captured_image,
                            model_name=self.model_name,
                            enforce_detection=True
                        )
                        
                        # Extract embedding vector
                        embedding = np.array(embedding_obj[0]["embedding"])
                        
                        print(f"Embedding generated successfully!")
                        print(f"Embedding shape: {embedding.shape}")
                        print(f"Embedding (first 10 values): {embedding[:10]}")
                        
                        # Clean up temp file if used
                        if not save_image and os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        break
                        
                    except Exception as e:
                        print(f"Error generating embedding: {e}")
                        print("Make sure your face is clearly visible and well-lit")
                else:
                    print("No face detected! Please position your face in the frame")
            
            # Quit when 'q' is pressed
            elif key == ord('q'):
                print("Exiting...")
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        return embedding, captured_image
    
    def compare_faces(self, img1_path, img2_path):
        """
        Compare two face images and return similarity
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
        """
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name
            )
            
            print(f"Faces match: {result['verified']}")
            print(f"Distance: {result['distance']}")
            print(f"Threshold: {result['threshold']}")
            
            return result
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return None


def main():
    print("=" * 60)
    print("Face Capture and Embedding System")
    print("=" * 60)
    
    # Available models
    print("\nAvailable DeepFace models:")
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", 
              "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    # Use default model or let user choose
    model_choice = input("\nEnter model number (press Enter for default Facenet512): ").strip()
    
    if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
        selected_model = models[int(model_choice) - 1]
    else:
        selected_model = "Facenet512"
    
    print(f"\nUsing model: {selected_model}")
    
    # Initialize face embedding system
    face_system = FaceEmbedding(model_name=selected_model)
    
    # Capture and generate embedding
    embedding, image_path = face_system.capture_and_embed(save_image=True)
    
    if embedding is not None:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Embedding vector dimension: {len(embedding)}")
        print(f"Image saved at: {image_path}")
        
        # Optionally save embedding to file
        save_option = input("\nDo you want to save the embedding to a file? (y/n): ").strip().lower()
        if save_option == 'y':
            embedding_file = image_path.replace('.jpg', '_embedding.npy')
            np.save(embedding_file, embedding)
            print(f"Embedding saved to: {embedding_file}")
    else:
        print("\nNo embedding was generated")


if __name__ == "__main__":
    main()
