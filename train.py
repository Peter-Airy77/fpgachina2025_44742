"""
Training Script for Separate Object and Action Recognition Models
Trains two independent CNN models:
1. Object Recognition: ball, bottle, empty, pen, phone, hand
2. Action Recognition: none, press, rotate, tap, touch, hold
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from cnn_model import get_model


class PressureDataset(Dataset):
    """Dataset loader for pressure sensor data"""
    
    def __init__(self, h5_files, task='object', transform=None):
        """
        Args:
            h5_files: List of HDF5 file paths
            task: 'object' or 'action' - determines which label to use
            transform: Optional data augmentation transforms
        """
        self.task = task
        self.transform = transform
        self.data = []
        
        # Load all data from HDF5 files
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                frames = f['frames'][:]
                objects = f['objects'][:]
                actions = f['actions'][:]
                
                # Decode bytes to strings
                objects = [obj.decode('utf-8') for obj in objects]
                actions = [act.decode('utf-8') for act in actions]
                
                # Load label mappings
                self.object_labels = json.loads(f.attrs['object_labels'])
                self.action_labels = json.loads(f.attrs['action_labels'])
                
                # Create label to index mappings
                self.object_to_idx = {label: idx for idx, label in enumerate(self.object_labels)}
                self.action_to_idx = {label: idx for idx, label in enumerate(self.action_labels)}
                
                # Store data
                for i in range(len(frames)):
                    self.data.append({
                        'frame': frames[i],
                        'object': self.object_to_idx[objects[i]],
                        'action': self.action_to_idx[actions[i]],
                        'timestamp': f['timestamps'][i] if 'timestamps' in f else 0
                    })
        
        print(f"Loaded {len(self.data)} samples from {len(h5_files)} files for {task} recognition")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get frame and normalize to [0, 1]
        frame = sample['frame'].astype(np.float32) / 255.0
        
        # Add channel dimension: (16, 16) -> (1, 16, 16)
        frame = np.expand_dims(frame, axis=0)
        
        # Apply transforms if any
        if self.transform:
            frame = self.transform(frame)
        
        # Convert to tensors
        frame = torch.from_numpy(frame)
        
        # Return label based on task
        if self.task == 'object':
            label = torch.tensor(sample['object'], dtype=torch.long)
        else:  # action
            label = torch.tensor(sample['action'], dtype=torch.long)
        
        return frame, label


class Trainer:
    """Training manager for single-task pressure sensor CNN"""
    
    def __init__(self, model, task='object', labels=None, device='cuda', learning_rate=0.001):
        """
        Args:
            model: PyTorch model
            task: 'object' or 'action'
            labels: List of label names
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.task = task
        self.labels = labels
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # History
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Training {self.task}')
        for frames, labels in pbar:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total_loss += loss.item()
            total_samples += frames.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': (pred == labels).float().mean().item()
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total_samples
        }
    
    def validate(self, val_loader):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum().item()
                total_loss += loss.item()
                total_samples += frames.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total_samples
        }
    
    def train(self, train_loader, val_loader, epochs=50, save_dir='models'):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Save history
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['accuracy'].append(val_metrics['accuracy'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                model_path = os.path.join(save_dir, f'best_{self.task}_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                    'labels': self.labels,
                    'task': self.task
                }, model_path)
                print(f"Saved best {self.task} model")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{self.task}_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                    'labels': self.labels,
                    'task': self.task
                }, checkpoint_path)
        
        # Plot training curves
        self.plot_training_curves(save_dir)
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        axes[0].plot(self.train_history['loss'], label='Train')
        axes[0].plot(self.val_history['loss'], label='Val')
        axes[0].set_title(f'{self.task.capitalize()} Recognition - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.train_history['accuracy'], label='Train')
        axes[1].plot(self.val_history['accuracy'], label='Val')
        axes[1].set_title(f'{self.task.capitalize()} Recognition - Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_curves_{self.task}.png'), dpi=150)
        plt.close()


def train_single_task(task='object', data_dir='data/collected', model_dir='models',
                      batch_size=32, epochs=50, learning_rate=0.001, train_split=0.8):
    """
    Train a single model for either object or action recognition
    
    Args:
        task: 'object' or 'action'
        data_dir: Directory containing HDF5 data files
        model_dir: Directory to save trained models
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        train_split: Train/validation split ratio
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training {task.upper()} Recognition Model")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    
    # Find all HDF5 files matching the task
    # IMPORTANT: Only load files for the specific task to avoid data imbalance
    # Object files: pressure_object_*.h5
    # Action files: pressure_action_*.h5
    all_files = os.listdir(data_dir)
    if task == 'object':
        h5_files = [os.path.join(data_dir, f) for f in all_files 
                    if f.endswith('.h5') and 'object' in f.lower()]
    else:  # action
        h5_files = [os.path.join(data_dir, f) for f in all_files 
                    if f.endswith('.h5') and 'action' in f.lower()]
    
    if len(h5_files) == 0:
        print(f"❌ No {task} data files found in {data_dir}!")
        print(f"   Please collect {task} data first:")
        print(f"   python data_collector.py {task}")
        return
    
    print(f"Found {len(h5_files)} {task} data files")
    for f in h5_files:
        print(f"  - {os.path.basename(f)}")
    
    # Create dataset
    dataset = PressureDataset(h5_files, task=task)
    
    # Get labels
    if task == 'object':
        labels = dataset.object_labels
    else:
        labels = dataset.action_labels
    
    print(f"Classes ({len(labels)}): {labels}")
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    # Create model
    model = get_model('advanced', task=task, num_classes=len(labels))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, task=task, labels=labels, device=device, learning_rate=learning_rate)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=epochs, save_dir=model_dir)
    
    print(f"\n{task.capitalize()} model training complete!")
    print(f"Model saved to: {model_dir}/best_{task}_model.pth")


def main():
    """Main training function - trains both models"""
    # Configuration
    DATA_DIR = 'data/collected'
    MODEL_DIR = 'models'
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 70)
    print("PRESSURE SENSOR CNN TRAINING")
    print("=" * 70)
    print("\nTraining Configuration:")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Model Directory: {MODEL_DIR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Train/Val Split: {TRAIN_SPLIT:.0%}")
    
    # Train Object Recognition Model
    print("\n" + "=" * 70)
    print("STEP 1/2: Training Object Recognition Model")
    print("=" * 70)
    train_single_task(
        task='object',
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        train_split=TRAIN_SPLIT
    )
    
    # Train Action Recognition Model
    print("\n" + "=" * 70)
    print("STEP 2/2: Training Action Recognition Model")
    print("=" * 70)
    train_single_task(
        task='action',
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        train_split=TRAIN_SPLIT
    )
    
    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTrained models saved in: {MODEL_DIR}/")
    print(f"  - best_object_model.pth")
    print(f"  - best_action_model.pth")
    print(f"\nTraining curves saved:")
    print(f"  - training_curves_object.png")
    print(f"  - training_curves_action.png")


if __name__ == '__main__':
    main()
