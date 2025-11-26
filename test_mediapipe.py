import cv2
import mediapipe as mp
import numpy as np

print("Testing MediaPipe Hand Detection Setup...")
print("=" * 50)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Test with webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot access webcam!")
    print("Please check if your webcam is connected and not being used by another app.")
    exit()

print("✅ Webcam accessed successfully!")
print("\nInstructions:")
print("- Show your hand to the camera")
print("- You should see green dots and lines tracking your hand")
print("- Press 'q' to quit")
print("=" * 50)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Show detection status
            cv2.putText(frame, f"✓ Hands Detected: {len(results.multi_hand_landmarks)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Show your hand to camera", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Instructions on frame
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('MediaPipe Hand Detection Test', frame)
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("✅ Test completed successfully!")
print("If you saw your hand with green landmarks, everything is working!")
print("=" * 50)