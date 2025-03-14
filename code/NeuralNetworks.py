import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the data
path = "/Users/alissadomenig/repositories/ML_final/Merged_Data.csv"
data = pd.read_csv(path)

# Assuming the first 44 columns are explanatory variables
X = data.iloc[:, :44].values
# Assuming the response variables start from column 44
Y = data.iloc[:, 44:].values

# Standardize the explanatory variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network with Dropout
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)  # Add dropout layer with 50% drop probability
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first hidden layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after the second hidden layer
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

response_variable_names = data.columns[44:]

overall_train_losses = {}
overall_val_losses = {}
prediction_accuracies = {}

epochs = 50

for i, response_name in enumerate(response_variable_names):
    y = Y[:, i].reshape(-1, 1)
    
    # Filter out missing values (0)
    valid_indices = (y != 0).flatten()
    X_valid = X[valid_indices]
    y_valid = y[valid_indices]
    
    # Convert 2 to 0 (for binary classification)
    y_valid = (y_valid == 1).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SimpleNN(input_size=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_losses = []
    val_losses = []
    
    # Train the model
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y.squeeze())  # Ensure y has the correct shape
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        # Store average training loss for this epoch
        train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(train_loss)
        
        # Evaluate validation loss
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test_tensor).squeeze()
            val_loss = criterion(val_predictions, y_test_tensor.squeeze()).item()
        val_losses.append(val_loss)
    
    overall_train_losses[response_name] = train_losses
    overall_val_losses[response_name] = val_losses
    
    # Compute prediction accuracy
    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train_tensor).cpu().numpy().flatten() > 0.5).astype(int)
        test_preds = (model(X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    prediction_accuracies[response_name] = (train_acc, test_acc)

# Define colors for each plot
train_loss_color = 'blue'
val_loss_color = 'orange'
train_acc_color = 'green'
test_acc_color = 'red'

# Plot training and validation losses separately for each response variable
for response_name in response_variable_names:
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), overall_train_losses[response_name], label=f"Train Loss - {response_name}", linestyle='--', color=train_loss_color)
    plt.plot(range(epochs), overall_val_losses[response_name], label=f"Validation Loss - {response_name}", color=val_loss_color)
    
    # Check for overfitting
    min_val_loss_epoch = overall_val_losses[response_name].index(min(overall_val_losses[response_name]))
    if min_val_loss_epoch < epochs - 10:
        plt.axvline(x=min_val_loss_epoch, color='red', linestyle=':', label=f"Min Validation Loss Epoch ({min_val_loss_epoch+1})")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Binary Cross-Entropy)")
    plt.title(f"Training and Validation Loss - {response_name}")
    plt.legend()
    plt.show()

# Plot prediction accuracy
plt.figure(figsize=(12, 6))
names = list(prediction_accuracies.keys())
train_acc_values = [prediction_accuracies[name][0] for name in names]
test_acc_values = [prediction_accuracies[name][1] for name in names]

x = range(len(names))
width = 0.4

plt.bar(x, train_acc_values, width=width, label='Train Accuracy', color=train_acc_color, alpha=0.6)
plt.bar([p + width for p in x], test_acc_values, width=width, label='Test Accuracy', color=test_acc_color, alpha=0.6)
plt.xticks([p + width/2 for p in x], names, rotation=90)
plt.ylabel("Accuracy")
plt.title("Prediction Accuracy for Each Response Variable")
plt.legend()
plt.show()
