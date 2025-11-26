import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from datetime import datetime

class ASLDataCollector:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None  # Will initialize when needed
        
        # ASL alphabet (we'll start with letters)
        self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Storage for collected data
        self.dataset = {label: [] for label in self.labels}
        
    def extract_landmarks(self, hand_landmarks):
        """Extract 21 landmark coordinates (x, y, z) = 63 features"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def collect_sign(self, label, num_samples=100):
        """Collect samples for a specific sign"""
        print(f"\n{'='*60}")
        print(f"📸 Collecting data for sign: '{label}'")
        print(f"{'='*60}")
        print(f"Target: {num_samples} samples")
        print("\nInstructions:")
        print("1. Look up ASL sign for this letter")
        print("2. Position your hand showing the sign")
        print("3. Press SPACE to start collecting")
        print("4. Hold the sign steady (move slightly for variation)")
        print("5. Press 'q' to skip to next letter")
        print("6. Press 'r' to restart this letter")
        print("="*60)
        
        input("Press ENTER when ready...")
        
        # Initialize hands here to avoid startup issues
        if self.hands is None:
            print("Initializing hand detector...")
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                max_num_hands=1
            )
        
        cap = cv2.VideoCapture(0)
        collecting = False
        samples_collected = 0
        
        while cap.isOpened() and samples_collected < num_samples:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create info panel
            info_panel = np.zeros((150, w, 3), dtype=np.uint8)
            
            # Display sign and progress
            cv2.putText(info_panel, f"Current Sign: {label}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Progress: {samples_collected}/{num_samples}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Progress bar
            progress = int((samples_collected / num_samples) * (w - 20))
            cv2.rectangle(info_panel, (10, 90), (w - 10, 120), (50, 50, 50), -1)
            cv2.rectangle(info_panel, (10, 90), (10 + progress, 120), (0, 255, 0), -1)
            
            if not collecting:
                cv2.putText(info_panel, "Press SPACE to start collecting", 
                           (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.circle(frame, (30, 30), 15, (0, 255, 255), -1)
            else:
                cv2.putText(info_panel, "COLLECTING... (hold steady)", 
                           (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            
            # Draw hand landmarks
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Collect data if in collecting mode
                    if collecting:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.dataset[label].append(landmarks)
                        samples_collected += 1
                
                cv2.putText(frame, "Hand Detected!", 
                           (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected - Show your hand!", 
                           (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Combine frame and info panel
            combined = np.vstack([info_panel, frame])
            cv2.imshow('ASL Data Collection', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to start/stop
                if hand_detected:
                    collecting = not collecting
                    if collecting:
                        print(f"⚡ Started collecting for '{label}'")
                else:
                    print("⚠️  Show your hand first!")
            elif key == ord('q'):  # Skip to next
                print(f"⏭️  Skipping '{label}'")
                break
            elif key == ord('r'):  # Restart this letter
                print(f"🔄 Restarting '{label}'")
                self.dataset[label] = []
                samples_collected = 0
        
        cap.release()
        print(f"✅ Collected {samples_collected} samples for '{label}'")
        return samples_collected
    
    def collect_all_signs(self, samples_per_sign=50):
        """Collect data for all signs"""
        print("\n" + "="*60)
        print("      ASL ALPHABET DATA COLLECTION")
        print("="*60)
        print("\n📋 Quick Reference:")
        print("   SPACE - Start/Stop collecting")
        print("   Q     - Skip to next letter")
        print("   R     - Restart current letter")
        print("\n💡 Tips:")
        print("   - Keep good lighting")
        print("   - Hold sign steady but move hand slightly")
        print("   - Look up each ASL sign before collecting")
        print("   - Take breaks between letters!")
        print("="*60)
        
        input("\nPress ENTER to start collection...")
        
        total_collected = 0
        for i, label in enumerate(self.labels, 1):
            print(f"\n📍 Letter {i}/{len(self.labels)}")
            collected = self.collect_sign(label, samples_per_sign)
            total_collected += collected
            
            # Option to take a break
            if i % 5 == 0 and i < len(self.labels):
                choice = input(f"\n☕ Break time! Continue? (y/n): ")
                if choice.lower() != 'y':
                    print("💾 Saving progress...")
                    break
        
        cv2.destroyAllWindows()
        
        print(f"\n🎉 Collection complete! Total samples: {total_collected}")
        self.save_dataset()
    
    def save_dataset(self):
        """Save collected dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.data_dir, f'asl_dataset_{timestamp}.pkl')
        
        # Convert lists to numpy arrays (only non-empty)
        dataset_np = {
            label: np.array(samples) 
            for label, samples in self.dataset.items() 
            if len(samples) > 0
        }
        
        if not dataset_np:
            print("⚠️  No data collected!")
            return None
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset_np, f)
        
        print(f"\n{'='*60}")
        print(f"💾 Dataset saved to: {filepath}")
        print("="*60)
        print("Summary:")
        total_samples = 0
        for label, samples in dataset_np.items():
            count = len(samples)
            total_samples += count
            print(f"  {label}: {count} samples")
        print(f"\nTotal: {total_samples} samples across {len(dataset_np)} signs")
        print("="*60)
        
        return filepath

# Main execution
if __name__ == "__main__":
    print("\n🎯 SignSpeak AI - Data Collection")
    print("="*60)
    
    collector = ASLDataCollector()
    
    print("\nRecommendation: Start with 30-50 samples per sign")
    print("You can always collect more later!")
    
    samples = input("\nHow many samples per sign? (default 50): ")
    samples = int(samples) if samples else 50
    
    print(f"\n📊 Will collect {samples} samples for each letter")
    print(f"   Total target: {samples * 26} samples")
    print(f"   Estimated time: {samples * 26 // 60} - {samples * 26 // 30} minutes")
    
    ready = input("\nReady to start? (y/n): ")
    if ready.lower() == 'y':
        collector.collect_all_signs(samples_per_sign=samples)
        print("\n✨ All done! Your dataset is ready for training.")
    else:
        print("No problem! Run this script again when ready.")