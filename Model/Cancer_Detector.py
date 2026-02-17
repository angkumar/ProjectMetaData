# Cancer_Detector.py
# Complete cancer detection system with training and research evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)
import seaborn as sns
import json
from datetime import datetime
import argparse

# ==================== CHECKPOINT MANAGEMENT ====================

def save_checkpoint(model, optimizer, epoch, loss, path="Cancter_Detector.pt"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

def load_checkpoint(model, optimizer, path="Cancter_Detector.pt"):
    if not os.path.exists(path):
        print("No checkpoint found, starting fresh")
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}, last loss: {checkpoint['loss']:.4f}")
    return start_epoch

# ==================== DATASET ====================

class H5CancerDataset(Dataset):
    def __init__(self, images_h5_path, labels_h5_path, transform=None,
                 image_key="x", label_key="y"):
        self.transform = transform
        self.images_h5_path = images_h5_path
        self.labels_h5_path = labels_h5_path
        self.image_key = image_key
        self.label_key = label_key
        self._images_h5 = None
        self._labels_h5 = None

        with h5py.File(self.images_h5_path, "r") as f:
            self._length = int(f[self.image_key].shape[0])
        with h5py.File(self.labels_h5_path, "r") as f:
            lbl_len = int(f[self.label_key].shape[0])
        if lbl_len != self._length:
            raise ValueError(f"Image/label length mismatch: {self._length} vs {lbl_len}")

    def _ensure_open(self):
        if self._images_h5 is None:
            self._images_h5 = h5py.File(self.images_h5_path, "r", swmr=True)
            self._images = self._images_h5[self.image_key]
        if self._labels_h5 is None:
            self._labels_h5 = h5py.File(self.labels_h5_path, "r", swmr=True)
            self._labels = self._labels_h5[self.label_key]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()
        img = torch.tensor(self._images[idx], dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(self._labels[idx].squeeze()), dtype=torch.long)
        return img, label

    def close(self):
        for attr in ("_images_h5", "_labels_h5"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            setattr(self, attr, None)

    def __del__(self):
        self.close()

# ==================== MODEL ====================

class CancerCNN(nn.Module):
    def __init__(self, num_classes, metadata_features=0):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.3)

        self.flattened_size = 128 * 12 * 12

        if metadata_features > 0:
            self.meta_fc = nn.Sequential(
                nn.Linear(metadata_features, 32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            combined_size = self.flattened_size + 32
        else:
            combined_size = self.flattened_size

        self.fc = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.metadata_features = metadata_features

    def forward(self, x, meta=None):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        if self.metadata_features and meta is not None:
            meta_features = self.meta_fc(meta)
            x = torch.cat([x, meta_features], dim=1)

        out = self.fc(x)
        return out

# ==================== TRAINING ====================

def train_model(model, train_loader, val_loader=None, num_epochs=50, lr=1e-3, device="mps"):
    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # resume if checkpoint exists
    start_epoch = load_checkpoint(model, optimizer)

    # Track history for plotting
    train_losses = []
    val_accs = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 3:
                images, meta, labels = batch
                images, meta, labels = images.to(device), meta.to(device), labels.to(device)
                outputs = model(images, meta)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}] Loss: {avg_loss:.4f}")

        # save after every epoch
        save_checkpoint(model, optimizer, epoch, avg_loss)
        print(f"Checkpoint saved at epoch {epoch+1}")

        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        images, meta, labels = batch
                        images, meta, labels = images.to(device), meta.to(device), labels.to(device)
                        outputs = model(images, meta)
                    else:
                        images, labels = batch
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            val_accs.append(val_acc)
            print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save training history
    if len(train_losses) > 0:
        history = {
            'train_losses': train_losses,
            'val_accs': val_accs if len(val_accs) > 0 else None
        }
        with open('training_history.json', 'w') as f:
            json.dump(history, f)
        print("Training history saved to training_history.json")

# ==================== RESEARCH EVALUATION ====================

def evaluate_model_comprehensive(model, test_loader, device, output_dir='research_outputs'):
    """
    Comprehensive evaluation for research paper
    Creates all necessary figures and metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION FOR RESEARCH")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Collecting predictions on test set...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(test_loader)} batches")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print(f"\nTotal test samples: {len(all_labels)}")
    
    # 1. Confusion Matrix
    print("\n1. Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Malignant'],
                yticklabels=['Healthy', 'Malignant'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Cancer Detection', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/confusion_matrix.png")
    
    # 2. Classification Report
    print("\n2. Generating Classification Report...")
    report = classification_report(all_labels, all_preds, 
                                   target_names=['Healthy', 'Malignant'],
                                   output_dict=True)
    
    report_text = classification_report(all_labels, all_preds,
                                       target_names=['Healthy', 'Malignant'])
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
    print(f"   Saved: {output_dir}/classification_report.txt")
    
    # 3. ROC Curve
    print("\n3. Generating ROC Curve...")
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/roc_curve.png")
    
    # 4. Precision-Recall Curve
    print("\n4. Generating Precision-Recall Curve...")
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=3,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Sensitivity)', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/precision_recall_curve.png")
    
    # 5. Calculate comprehensive metrics
    print("\n5. Calculating Metrics...")
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'test_samples': len(all_labels),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'f1_score': float(report['Malignant']['f1-score']),
        'auc_roc': float(roc_auc),
        'auc_pr': float(pr_auc),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
        'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"   Saved: {output_dir}/metrics.json")
    
    # 6. Create summary report
    print("\n6. Creating Summary Report...")
    with open(f'{output_dir}/RESEARCH_SUMMARY.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("CANCER DETECTION MODEL - RESEARCH SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TEST SET PERFORMANCE\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Test Samples: {metrics['test_samples']:,}\n\n")
        
        f.write("PRIMARY METRICS:\n")
        f.write(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)\n")
        f.write(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)\n")
        f.write(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
        f.write(f"  F1-Score:    {metrics['f1_score']:.4f}\n")
        f.write(f"  AUC-ROC:     {metrics['auc_roc']:.4f}\n")
        f.write(f"  AUC-PR:      {metrics['auc_pr']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True Positives:  {metrics['true_positives']:,}\n")
        f.write(f"  True Negatives:  {metrics['true_negatives']:,}\n")
        f.write(f"  False Positives: {metrics['false_positives']:,}\n")
        f.write(f"  False Negatives: {metrics['false_negatives']:,}\n\n")
        
        f.write("ERROR RATES:\n")
        f.write(f"  False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)\n")
        f.write(f"  False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)\n\n")
        
        f.write("CLINICAL INTERPRETATION:\n")
        f.write("-"*60 + "\n")
        f.write(f"Sensitivity of {metrics['sensitivity']*100:.1f}% means the model correctly\n")
        f.write(f"identifies {metrics['sensitivity']*100:.1f}% of cancer cases.\n\n")
        f.write(f"Specificity of {metrics['specificity']*100:.1f}% means the model correctly\n")
        f.write(f"identifies {metrics['specificity']*100:.1f}% of healthy tissue samples.\n\n")
        f.write(f"False negative rate of {metrics['false_negative_rate']*100:.1f}% means\n")
        f.write(f"{metrics['false_negatives']:,} cancer cases were missed.\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"   Saved: {output_dir}/RESEARCH_SUMMARY.txt")
    
    # Print summary to console
    print("\n" + "="*60)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*60)
    print(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity: {metrics['sensitivity']*100:.2f}%")
    print(f"Specificity: {metrics['specificity']*100:.2f}%")
    print(f"AUC-ROC:     {metrics['auc_roc']:.3f}")
    print(f"\nAll outputs saved to: {output_dir}/")
    print("="*60 + "\n")
    
    return metrics, cm, report


def plot_training_history(train_losses, val_accs, output_dir='research_outputs'):
    """
    Plot training history for research paper
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if val_accs and len(val_accs) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_title('Training Loss', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Accuracy (%)', fontsize=14)
        ax2.set_title('Validation Accuracy', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        # Just loss if no validation
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to: {output_dir}/training_history.png")


# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cancer Detector - Train or Evaluate')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--checkpoint', type=str, default='Cancter_Detector.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_images', type=str, default=None,
                        help='Path to test images H5 file (for evaluation)')
    parser.add_argument('--test_labels', type=str, default=None,
                        help='Path to test labels H5 file (for evaluation)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='research_outputs',
                        help='Directory for research outputs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # ==================== TRAINING MODE ====================
        print("\n" + "="*60)
        print("CANCER DETECTOR - TRAINING MODE")
        print("="*60 + "\n")
        
        train_dataset = H5CancerDataset(
            images_h5_path="Training_Data/pcam/training_split.h5",
            labels_h5_path="Training_Data/Labels/camelyonpatch_level_2_split_train_y.h5",
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]),
        )

        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        print(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}\n")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2
        )

        model = CancerCNN(num_classes=2, metadata_features=0)

        train_model(
            model,
            train_loader,
            val_loader=None,
            num_epochs=args.epochs,
            lr=1e-3,
            device="mps",
        )

        torch.save(model.state_dict(), args.checkpoint)
        print(f"Model saved to {args.checkpoint}")
        
        # Plot training history if available
        if os.path.exists('training_history.json'):
            with open('training_history.json', 'r') as f:
                history = json.load(f)
            plot_training_history(history['train_losses'], history.get('val_accs'), args.output_dir)
    
    elif args.mode == 'evaluate':
        # ==================== EVALUATION MODE ====================
        print("\n" + "="*60)
        print("CANCER DETECTOR - EVALUATION MODE")
        print("="*60)
        
        if not args.test_images or not args.test_labels:
            print("ERROR: Must specify --test_images and --test_labels for evaluation")
            exit(1)
        
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Test images: {args.test_images}")
        print(f"Test labels: {args.test_labels}")
        print(f"Output directory: {args.output_dir}")
        print("="*60 + "\n")
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}\n")
        
        # Load test dataset
        print("Loading test dataset...")
        test_dataset = H5CancerDataset(
            images_h5_path=args.test_images,
            labels_h5_path=args.test_labels
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        print(f"Test dataset: {len(test_dataset)} samples\n")
        
        # Load model
        print("Loading model...")
        model = CancerCNN(num_classes=2)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        print("Model loaded successfully!\n")
        
        # Run evaluation
        metrics, cm, report = evaluate_model_comprehensive(
            model, test_loader, device, output_dir=args.output_dir
        )
        
        print("\n✅ Research evaluation complete!")
        print(f"📁 All outputs saved to: {args.output_dir}/")