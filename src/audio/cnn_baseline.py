import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml

from src.audio.load_features import load_features

# --- CONFIG SETUP ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(CNNEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.fc1_input_size = self._get_conv_output_shape(input_shape)
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output_shape(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # Setup paths from config
    MODELS_DIR = os.path.join(PROJECT_ROOT, CONFIG['model']['dir'])
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, CONFIG['model']['name'])
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load Data
    X, y = load_features()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split using config seed and size
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, 
        test_size=CONFIG['training']['val_split'], 
        random_state=CONFIG['training']['seed']
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train).unsqueeze(1).float()
    X_val_tensor = torch.tensor(X_val).unsqueeze(1).float()
    y_train_tensor = torch.tensor(y_train).long()
    y_val_tensor = torch.tensor(y_val).long()

    # Create dataloaders with config batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['training']['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    input_shape = (1, X_train.shape[1], X_train.shape[2])
    model = CNNEmotionClassifier(len(np.unique(y_encoded)), input_shape).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['training']['learning_rate'])
    writer = SummaryWriter('runs/emotitune_cnn_v1')

    # Training loop using config epochs
    num_epochs = CONFIG['training']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f} - Val Acc: {val_accuracy:.2f}%")
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    writer.close()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()