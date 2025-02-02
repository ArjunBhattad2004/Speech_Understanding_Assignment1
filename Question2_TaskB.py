import torch
import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt

# Define Paths for Genre Audio Files
GENRE_PATHS = {
    "Rock": "Gothic-Dark_Rock.mp3",
    "Jazz": "Morning-Routine_Jazz.mp3",
    "Classical": "Mozart-Serenade-in-G-major_Classical.mp3",
    "Hip-Hop": "mixkit-cbpd-400_HipHop.mp3"
}

# Function to Apply STFT
def apply_stft(waveform, sample_rate, window_fn):
    transform = transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
    return transform(waveform)

# Load and Process Genre Audio Files
genre_waveforms = {}
for genre, path in GENRE_PATHS.items():
    try:
        waveform, sample_rate = torchaudio.load(path)
        genre_waveforms[genre] = (waveform, sample_rate)
    except Exception as e:
        print(f"Error loading {genre}: {e}")

# Generate and Plot Spectrograms
plt.figure(figsize=(10, 8))
for i, (genre, (waveform, sample_rate)) in enumerate(genre_waveforms.items()):
    spec = apply_stft(waveform, sample_rate, torch.hann_window)
    plt.subplot(2, 2, i + 1)
    plt.imshow(spec.log2().numpy()[0], cmap="inferno", aspect="auto")
    plt.title(f"{genre} Spectrogram")

plt.show()
print("Genre Spectrogram Analysis Completed!")
