import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.audio.load_features import load_features
from src.audio.model import CNNEmotionClassifier
from src.utils import setup_logger

# --- CONFIG & LOGGER ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger('training', log_file='logs/training.log')

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    logger.info("Starting training pipeline...")
    MODELS_DIR = os.path.join(PROJECT_ROOT, CONFIG['model']['dir'])
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, CONFIG['model']['name'])
    RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results.yaml')
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        X, y = load_features()
    except FileNotFoundError:
        logger.critical("Data not found! Run save_features.py first.")
        exit(1)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    np.save(os.path.join(MODELS_DIR, 'classes.npy'), classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=CONFIG['training']['val_split'], 
        random_state=CONFIG['training']['seed'], stratify=y_encoded
    )

    # Convert to tensors (N, C, H, W)
    X_train_tensor = torch.tensor(X_train).unsqueeze(1).float()
    X_val_tensor = torch.tensor(X_val).unsqueeze(1).float()
    y_train_tensor = torch.tensor(y_train).long()
    y_val_tensor = torch.tensor(y_val).long()

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                              batch_size=CONFIG['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), 
                            batch_size=CONFIG['training']['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Input shape: (1, 65, 174) based on standard config
    input_shape = (1, X_train.shape[1], X_train.shape[2])
    model = CNNEmotionClassifier(len(classes), input_shape).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['training']['learning_rate'])
    writer = SummaryWriter('runs/emotitune_experiment')

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Starting training loop...")
    for epoch in range(CONFIG['training']['num_epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['training']['patience']:
                logger.info("Early stopping triggered.")
                break

    writer.close()

    # --- FINAL EVALUATION ---
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    final_acc = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average='weighted')

    # Save results to yaml for run.py to read
    results = {
        'best_val_loss': float(best_val_loss),
        'final_accuracy': float(final_acc),
        'final_f1_score': float(final_f1)
    }
    with open(RESULTS_PATH, 'w') as f:
        yaml.dump(results, f)

    # Save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes, 
                          os.path.join(PROJECT_ROOT, 'confusion_matrix.png'))
    logger.info(f"Training complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()