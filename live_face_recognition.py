import cv2
import time
import os
from utils.face_recognition_logic import FaceRecognitionLogic


## Keep Threshold as 0.65, below this face is not recognised

def process_frame(frame, current_name, current_description, face_recognition_class):
    """
    Process the captured frame for face recognition.
    This function is called every 5 seconds with a frame from the camera.
    
    Args:
        frame: The captured frame from the camera feed
        face_recognition_logic: The FaceRecognitionLogic object (passed by reference)
    
    Returns:
        tuple: (name, description) of recognized person or (None, None) if no match
    """
    print(f"Processing frame at {time.strftime('%H:%M:%S')}")
    
    # Create live_recognition folder if it doesn't exist
    save_dir = "live_recognition"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the frame (overwrites each time)
    save_path = os.path.join(save_dir, "current_frame.jpg")
    cv2.imwrite(save_path, frame)
    print(f"Frame saved to {save_path}")
    
    # Call face recognition logic using the same object instance
    # The face_recognition_class object is passed by reference, not copied
    # Any state changes in the object will persist across calls
    result = face_recognition_class.recognise_face(save_path)
    
    recognised_name = result["name"]
    recognised_description = result["description"]

    if recognised_name == current_name and recognised_description == current_description:
        print(f"🕵🏻‍♂️ Person is same : \nname - {recognised_name}, \ndescription - {recognised_description}")
        return current_name,current_description
    else:
        print(f'''🚧 Person is changed : 
              previous_name - {current_name}, 
              previous_description - {current_description},
              recognised_name - {recognised_name}, 
              recognised_description - {recognised_description}
              ''')
        return recognised_name,recognised_description


def start_live_recognition():
    """
    Start live camera feed and process frames every 5 seconds.
    """
    # Initialize FaceRecognitionLogic object ONCE
    # This same object instance will be passed to process_frame each time
    face_recognition_class = FaceRecognitionLogic()
    
    # Open the default camera (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully")
    print("Press 'q' to quit")
    
    # Track time for 5-second intervals
    last_process_time = time.time()
    process_interval = 5  # seconds
    
    current_name = ""
    current_description = ""
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Check if 5 seconds have passed
        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            # Process this frame
            current_name, current_description = process_frame(frame.copy(),current_name,current_description,face_recognition_class)
            # Update the last process time
            last_process_time = current_time
            
            # Display recognition result
            if current_name:
                print(f"Recognized: {current_name} - {current_description}")
            else:
                print("No face recognized")
        
        # Display current recognition on frame (if available)
        if current_name:
            cv2.putText(frame, f"{current_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the camera feed
        cv2.imshow('Live Face Recognition', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")


if __name__ == "__main__":
    start_live_recognition()
