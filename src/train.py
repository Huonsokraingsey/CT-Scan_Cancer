import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Import components
from .dataset import get_dataloaders
from .model import ChestCTClassifier

def train():

    DATA_DIR = "/Users/jayden/Desktop/CT-Scan_Disease/Data/train"    
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    NUM_EPOCHS = 20
    BATCH_SIZE = 16
    PATIENCE = 4  # Early stopping patience
    
    print(f"Training on device: {DEVICE}")

    # 2. LOAD DATA
    train_loader, val_loader, _, classes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Found {len(classes)} classes: {classes}")

    # 3. INITIALIZE MODEL
    model = ChestCTClassifier(num_classes=len(classes)).to(DEVICE)
    
    # 4. LOSS & OPTIMIZER
    class_counts = [30, 26, 35, 31]
    class_weights = torch.tensor([max(class_counts)/c for c in class_counts], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 5. TRAINING LOOP
    best_acc = 0.0
    patience_counter = 0
    Path("models").mkdir(exist_ok=True)
    
    print(f"Starting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(NUM_EPOCHS):
        # --- Train Mode ---
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
        avg_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        
        # --- Validation Mode ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        
        # --- Print Progress ---
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train | Acc: {train_acc:.2%} | Prec: {train_precision:.2%} | Recall: {train_recall:.2%} | F1: {train_f1:.2%} | Loss: {avg_loss:.4f}")
        print(f"  Val   | Acc: {val_acc:.2%} | Prec: {val_precision:.2%} | Recall: {val_recall:.2%} | F1: {val_f1:.2%}")

        # --- Save Best Model ---
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")
            print("  -> Saved best model!")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
            
        # Update learning rate
        scheduler.step(val_acc)
            
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    # --- Final Report ---
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    print(f"\n{'='*60}")
    print("Final Classification Report (Best Model):")
    print(f"{'='*60}")
    print(classification_report(final_labels, final_preds, target_names=classes, zero_division=0))
    print(f"Confusion Matrix:\n{confusion_matrix(final_labels, final_preds)}")
    print(f"\nBest Accuracy: {best_acc:.2%}")
    print(f"Model saved to: models/best_model.pth")

if __name__ == "__main__":
    train()