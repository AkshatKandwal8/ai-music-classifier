"""
COMPLETE MUSIC GENRE CLASSIFIER
================================

This is a COMPLETE, WORKING music classification program.
Just run it and it will train on synthetic audio data!

Run: python music_classifier_complete.py

What it does:
1. Creates synthetic "music" data (3 different genres)
2. Converts to spectrograms
3. Trains a CNN
4. Classifies music genres
5. Shows results

NO EXTERNAL DATA NEEDED - Works immediately!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print("="*70)
print("🎵 MUSIC GENRE CLASSIFIER - COMPLETE VERSION")
print("="*70)

# ============================================================================
# STEP 1: CREATE SYNTHETIC AUDIO DATA
# ============================================================================

print("\n📊 STEP 1: Creating synthetic audio data...")

def create_genre_audio(genre_type, num_samples=30, duration=3, sample_rate=22050):
    """
    Create synthetic audio for different genres.
    
    In real version, you'd load actual MP3 files here.
    For demo, we create simple waveforms that differ by frequency.
    """
    samples = []
    
    for i in range(num_samples):
        # Create time array
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if genre_type == 'rock':
            # Rock: high frequency + noise (energetic)
            freq = 400 + np.random.randint(-20, 20)
            audio = np.sin(2 * np.pi * freq * t)
            audio += 0.3 * np.random.randn(len(t))  # Add noise
            
        elif genre_type == 'classical':
            # Classical: medium frequency, clean
            freq = 250 + np.random.randint(-20, 20)
            audio = np.sin(2 * np.pi * freq * t)
            # Add harmonics (more musical)
            audio += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            
        elif genre_type == 'jazz':
            # Jazz: low frequency, complex
            freq = 150 + np.random.randint(-20, 20)
            audio = np.sin(2 * np.pi * freq * t)
            # Add swing feeling (irregular rhythms)
            audio += 0.2 * np.sin(2 * np.pi * freq * 1.5 * t)
            
        samples.append(audio)
    
    return samples

# Create datasets for 3 genres
print("Creating Rock samples (high frequency)...")
rock_samples = create_genre_audio('rock', num_samples=30)

print("Creating Classical samples (medium frequency)...")
classical_samples = create_genre_audio('classical', num_samples=30)

print("Creating Jazz samples (low frequency)...")
jazz_samples = create_genre_audio('jazz', num_samples=30)

print(f"✓ Created {len(rock_samples) + len(classical_samples) + len(jazz_samples)} audio samples")

# ============================================================================
# STEP 2: CONVERT TO SPECTROGRAMS
# ============================================================================

print("\n🎨 STEP 2: Converting audio to spectrograms...")

def audio_to_spectrogram(audio, n_fft=512, hop_length=256):
    """
    Convert audio to spectrogram using FFT.
    This is a simplified version of what librosa does.
    """
    # Compute Short-Time Fourier Transform
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    spectrogram = np.zeros((n_fft // 2, num_frames))
    
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        
        # Apply window
        window = np.hanning(n_fft)
        frame = frame * window
        
        # FFT
        fft = np.fft.rfft(frame)
        magnitude = np.abs(fft)[:-1]  # Remove last value
        spectrogram[:, i] = magnitude
    
    # Resize to fixed size (128x128)
    from scipy import ndimage
    spectrogram_resized = ndimage.zoom(spectrogram, (128/spectrogram.shape[0], 128/spectrogram.shape[1]))
    
    # Normalize
    if spectrogram_resized.max() > 0:
        spectrogram_resized = spectrogram_resized / spectrogram_resized.max()
    
    return spectrogram_resized

# Convert all samples to spectrograms
all_spectrograms = []
all_labels = []

print("Processing Rock...")
for audio in rock_samples:
    spec = audio_to_spectrogram(audio)
    all_spectrograms.append(spec)
    all_labels.append(0)  # Label 0 = Rock

print("Processing Classical...")
for audio in classical_samples:
    spec = audio_to_spectrogram(audio)
    all_spectrograms.append(spec)
    all_labels.append(1)  # Label 1 = Classical

print("Processing Jazz...")
for audio in jazz_samples:
    spec = audio_to_spectrogram(audio)
    all_spectrograms.append(spec)
    all_labels.append(2)  # Label 2 = Jazz

# Convert to arrays
X = np.array(all_spectrograms)
y = np.array(all_labels)

print(f"✓ Created {len(X)} spectrograms of shape {X[0].shape}")

# Visualize examples
print("\n📸 Creating visualization...")
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].imshow(X[0], cmap='viridis', aspect='auto')
axes[0].set_title('Rock (High Freq)')
axes[1].imshow(X[30], cmap='viridis', aspect='auto')
axes[1].set_title('Classical (Med Freq)')
axes[2].imshow(X[60], cmap='viridis', aspect='auto')
axes[2].set_title('Jazz (Low Freq)')
for ax in axes:
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('genre_spectrograms.png', dpi=150, bbox_inches='tight')
print("✓ Saved: genre_spectrograms.png")
plt.close()

# ============================================================================
# STEP 3: CREATE NEURAL NETWORK
# ============================================================================

print("\n🧠 STEP 3: Building neural network...")

class GenreClassifier(nn.Module):
    """
    CNN for music genre classification.
    
    Input: Spectrogram [batch, 1, 128, 128]
    Output: Genre probabilities [batch, 3]
    """
    
    def __init__(self):
        super(GenreClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 3)  # 3 genres
    
    def forward(self, x):
        # Conv blocks
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create model
model = GenreClassifier()
num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created with {num_params:,} parameters")

# ============================================================================
# STEP 4: PREPARE DATA
# ============================================================================

print("\n📦 STEP 4: Preparing data...")

# Split into train/test
train_size = int(0.8 * len(X))
indices = np.random.permutation(len(X))

X_train = X[indices[:train_size]]
y_train = y[indices[:train_size]]
X_test = X[indices[train_size:]]
y_test = y[indices[train_size:]]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# ============================================================================
# STEP 5: TRAIN THE MODEL
# ============================================================================

print("\n🎯 STEP 5: Training the model...")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = 100 * (predicted == y_test).sum().item() / len(y_test)
    
    train_losses.append(loss.item())
    test_accuracies.append(accuracy)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | Test Acc: {accuracy:.1f}%")

print(f"\n✓ Training complete!")
print(f"✓ Final accuracy: {test_accuracies[-1]:.1f}%")

# ============================================================================
# STEP 6: EVALUATE AND VISUALIZE
# ============================================================================

print("\n📊 STEP 6: Evaluating results...")

# Get predictions on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    probabilities = torch.softmax(test_outputs, dim=1)
    _, predictions = torch.max(test_outputs, 1)

# Show some predictions
genre_names = ['Rock', 'Classical', 'Jazz']
print("\nSample predictions:")
print("-" * 60)
for i in range(min(10, len(predictions))):
    true_label = y_test[i].item()
    pred_label = predictions[i].item()
    confidence = probabilities[i][pred_label].item()
    
    status = "✓" if true_label == pred_label else "✗"
    print(f"{status} True: {genre_names[true_label]:10} | "
          f"Predicted: {genre_names[pred_label]:10} | "
          f"Confidence: {confidence:.1%}")

# Plot training curves
print("\n📈 Creating training curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(test_accuracies)
ax2.set_title('Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("✓ Saved: training_curves.png")
plt.close()

# ============================================================================
# STEP 7: CONFUSION MATRIX
# ============================================================================

print("\n🎯 STEP 7: Creating confusion matrix...")

# Calculate confusion matrix
confusion_matrix = np.zeros((3, 3), dtype=int)
for true, pred in zip(y_test, predictions):
    confusion_matrix[true][pred] += 1

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(confusion_matrix, cmap='Blues')

# Labels
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(genre_names)
ax.set_yticklabels(genre_names)

# Text annotations
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, confusion_matrix[i, j],
                      ha="center", va="center", color="black", fontsize=16)

ax.set_title('Confusion Matrix', fontsize=14, pad=20)
ax.set_xlabel('Predicted Genre', fontsize=12)
ax.set_ylabel('True Genre', fontsize=12)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎉 SUMMARY")
print("="*70)
print(f"""
✓ Trained on {len(X_train)} spectrograms
✓ Tested on {len(X_test)} spectrograms
✓ Final accuracy: {test_accuracies[-1]:.1f}%
✓ Model has {num_params:,} parameters

Generated files:
  1. genre_spectrograms.png - Example spectrograms
  2. training_curves.png - Loss and accuracy over time
  3. confusion_matrix.png - Classification results

What you learned:
  1. How to process audio data
  2. Converting audio to spectrograms
  3. Building CNNs for classification
  4. Training and evaluation
  5. Visualizing results

Next steps:
  1. Try with real MP3 files
  2. Add more genres
  3. Experiment with model architecture
  4. Try data augmentation
""")

print("="*70)
print("🎵 Music genre classification complete!")
print("="*70)

# Save the model
torch.save(model.state_dict(), 'genre_classifier.pth')
print("\n✓ Model saved to: genre_classifier.pth")
print("\nTo use the saved model:")
print("  model = GenreClassifier()")
print("  model.load_state_dict(torch.load('genre_classifier.pth'))")
print("  model.eval()")
