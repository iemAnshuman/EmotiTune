import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.audio.load_features import load_features

# Define CNN model
class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(CNNEmotionClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Dynamically calculate the input size for the fully connected layer
        self.fc1_input_size = self._get_conv_output_shape(input_shape)
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Add a dropout layer
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
        # Apply dropout
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    X, y = load_features()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train).unsqueeze(1).float()
    X_val_tensor = torch.tensor(X_val).unsqueeze(1).float()
    y_train_tensor = torch.tensor(y_train).long()
    y_val_tensor = torch.tensor(y_val).long()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the input shape from the data
    input_shape = (1, X_train.shape[1], X_train.shape[2])
    model = CNNEmotionClassifier(num_classes=len(np.unique(y_encoded)), input_shape=input_shape).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setup TensorBoard writer
    writer = SummaryWriter('runs/emotitune_cnn_dropout') # Changed log directory

    num_epochs = 30
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}")

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
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    writer.close()

if __name__ == "__main__":
    main()
