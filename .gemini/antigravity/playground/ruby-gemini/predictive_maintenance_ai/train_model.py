import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 50
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 200
LEARNING_RATE = 0.001

# Check for device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. DATA INGESTION ---
cols = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3', 
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

print("⏳ Loading Data...")
# Use the correct path relative to where the script is run or absolute path
try:
    train_df = pd.read_csv('predictive_maintenance_ai/train_FD001.txt', sep=r'\s+', header=None, names=cols)
except FileNotFoundError:
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=cols)

# --- 2. FEATURE ENGINEERING ---
def calculate_rul(df):
    max_cycle = df.groupby('unit_number')['time_in_cycles'].max()
    result_df = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_number', right_index=True)
    result_df['RUL'] = result_df['max_cycle'] - result_df['time_in_cycles']
    return result_df.drop('max_cycle', axis=1)

train_df = calculate_rul(train_df)

# --- 3. NORMALIZATION ---
features = ['setting_1', 'setting_2', 'setting_3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 
            's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])
print("✅ Data Normalized.")

# --- 4. SLIDING WINDOW GENERATOR ---
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

seq_gen = (list(gen_sequence(train_df[train_df['unit_number']==id], SEQUENCE_LENGTH, features)) 
           for id in train_df['unit_number'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

label_gen = (gen_labels(train_df[train_df['unit_number']==id], SEQUENCE_LENGTH, ['RUL']) 
             for id in train_df['unit_number'].unique())
label_array = np.concatenate(list(label_gen)).astype(np.float32)

print(f"🧩 Input Shape: {seq_array.shape}")

# Convert to PyTorch Tensors
X_train = torch.tensor(seq_array)
y_train = torch.tensor(label_array)

# Create DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 5. MODEL ARCHITECTURE ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :] # Take the last time step
        out = self.dropout2(out)
        out = self.fc(out)
        return out

model = LSTMModel(input_size=len(features), hidden_size1=100, hidden_size2=50, output_size=1, dropout_rate=DROPOUT_RATE)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. TRAINING ---
print("🚀 Starting Training...")
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'nasa_engine_model.pth')
print("💾 Model Saved: nasa_engine_model.pth")

# Plot the training curve
plt.plot(loss_history, label='Train Loss')
plt.title('Model Learning Curve')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss_curve.png')
print("📊 Chart Saved: loss_curve.png")
