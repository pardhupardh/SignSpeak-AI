import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class ASLModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = None
        self.history = None
        
    def load_data(self):
        """Load the collected dataset"""
        print("📂 Loading dataset...")
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Signs available: {list(dataset.keys())}")
        
        # Prepare data for training
        X = []
        y = []
        
        for label, samples in dataset.items():
            for sample in samples:
                X.append(sample)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   Feature dimension: {X.shape[1]} (21 landmarks × 3 coordinates)")
        print(f"   Number of classes: {len(np.unique(y))}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Print class distribution
        print("\n📈 Samples per class:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"   {label}: {count} samples")
        
        return X, y_encoded
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets"""
        print(f"\n✂️ Splitting data (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape, num_classes):
        """Build the neural network model"""
        print("\n🏗️ Building model architecture...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Dense layers with dropout for regularization
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model architecture:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the model"""
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print("="*60)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n✅ Training completed!")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\n📊 Evaluating model on test set...")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        print("\n📈 Generating training plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("   Saved plot as 'training_history.png'")
        plt.show()
    
    def save_model(self, model_dir='models'):
        """Save the trained model and label encoder"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'asl_model.h5')
        self.model.save(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"💾 Label encoder saved to: {encoder_path}")
    
    def run_full_pipeline(self, epochs=50, batch_size=32):
        """Run the complete training pipeline"""
        print("\n" + "="*60)
        print("      ASL MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load and prepare data
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(input_shape=(X.shape[1],), num_classes=num_classes)
        
        # Train model
        self.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
        
        # Evaluate
        self.evaluate(X_test, y_test)
        
        # Plot results
        self.plot_training_history()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("✨ TRAINING COMPLETE! Your model is ready to use.")
        print("="*60)


# Main execution
if __name__ == "__main__":
    print("\n🎯 SignSpeak AI - Model Training")
    print("="*60)
    
    # Find the latest dataset
    data_dir = 'data/raw'
    datasets = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not datasets:
        print("❌ No dataset found in data/raw/")
        print("Please run collect_data.py first to collect training data.")
        exit()
    
    # Use the most recent dataset
    latest_dataset = sorted(datasets)[-1]
    dataset_path = os.path.join(data_dir, latest_dataset)
    
    print(f"\n📂 Found dataset: {latest_dataset}")
    
    # Initialize trainer
    trainer = ASLModelTrainer(dataset_path)
    
    # Training parameters
    print("\n⚙️ Training Configuration:")
    epochs = input("Number of epochs (default 50, recommended 30-100): ")
    epochs = int(epochs) if epochs else 50
    
    batch_size = input("Batch size (default 32, recommended 16-64): ")
    batch_size = int(batch_size) if batch_size else 32
    
    print(f"\n🎯 Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    ready = input("\nStart training? (y/n): ")
    
    if ready.lower() == 'y':
        trainer.run_full_pipeline(epochs=epochs, batch_size=batch_size)
    else:
        print("Training cancelled. Run this script again when ready!")