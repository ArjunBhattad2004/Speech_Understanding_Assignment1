import torch
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Define dataset paths
AUDIO_PATH = "UrbanSound8K/audio/"
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"

# UrbanSound Dataset Class
class UrbanSoundDataset(Dataset):
    def __init__(self, metadata_path, audio_path):
        self.metadata = pd.read_csv(metadata_path)
        self.audio_path = audio_path

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = f"{self.audio_path}/fold{row['fold']}/{row['slice_file_name']}"
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
        label = torch.tensor(row['classID'], dtype=torch.long)
        return waveform, sample_rate, label

# Apply STFT with different windowing functions
def apply_stft(waveform, sample_rate, window_fn):
    transform = transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
    return transform(waveform)

# Load Dataset and Apply Windowing
dataset = UrbanSoundDataset(METADATA_PATH, AUDIO_PATH)
waveform, sample_rate, _ = dataset[0]

windows = {
    "Hann": torch.hann_window,
    "Hamming": torch.hamming_window,
    "Rectangular": lambda size: torch.ones(size)
}

plt.figure(figsize=(12, 4))
for i, (name, window_fn) in enumerate(windows.items()):
    spec = apply_stft(waveform, sample_rate, window_fn)
    plt.subplot(1, 3, i + 1)
    plt.imshow(spec.log2().numpy()[0], cmap="viridis", aspect="auto")
    plt.title(f"{name} Window")
plt.show()

# CNN Model for Audio Classification
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 128 * 4, 10)  # Assuming input size
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Train CNN Model
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(5):
    for batch in dataloader:
        if batch is None:
            continue
        waveform, sample_rate, labels = batch
        optimizer.zero_grad()
        spectrograms = apply_stft(waveform, sample_rate, torch.hann_window)
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")

print("Training Completed!")
