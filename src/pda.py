# Import required libraries
import numpy as np
from scipy.signal import butter, filtfilt, stft
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # For resizing images

# Preprocessing the dataset
file_path = '/content/s01.mat'
mat_data = loadmat(file_path)

# Extracting all EEG channels (eeg1 to eeg24)
eeg_signals = [mat_data[f'eeg{i}'].flatten() for i in range(1, 25)]

# Bandpass filter function
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=250, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Normalize function
def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1

# Resize image to a fixed shape
def resize_image(image, target_shape=(128, 128)):
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)

# Preprocessing EEG signals
preprocessed_signals = [normalize(bandpass_filter(signal)) for signal in eeg_signals]
min_length = min(len(signal) for signal in preprocessed_signals)
preprocessed_signals = [signal[:min_length] for signal in preprocessed_signals]  # Truncate
preprocessed_signals = np.array(preprocessed_signals)

# STFT feature extraction
def generate_stft_image(signal, fs=250, nperseg=128, noverlap=64):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)

# Extract STFT features and resize
stft_images = [resize_image(generate_stft_image(signal)) for signal in preprocessed_signals]
stft_images = np.array(stft_images)[..., np.newaxis]  # Adding channel axis for CNN compatibility

# VMD + STFT feature extraction
def perform_vmd(signal, num_modes=4):
    return [signal[i::num_modes] for i in range(num_modes)]

def generate_vmd_stft_image(modes, fs=250, nperseg=128, noverlap=64):
    spectrum_images = [generate_stft_image(mode) for mode in modes]
    return np.stack(spectrum_images, axis=-1)

# Extract VMD+STFT features and resize
vmd_stft_images = []
for signal in preprocessed_signals:
    modes = perform_vmd(signal, num_modes=4)  # Decompose signal into modes
    spectrum_image = generate_vmd_stft_image(modes)
    resized_spectrum = np.stack([resize_image(mode) for mode in spectrum_image], axis=-1)
    vmd_stft_images.append(resized_spectrum)

vmd_stft_images = np.array(vmd_stft_images)

# Split data into training and testing sets
labels = np.array([0, 1, 2, 3] * 6)  # Labels for emotions (0: Neutral, 1: Sad, 2: Fear, 3: Happy)
X_train_stft, X_test_stft, y_train, y_test = train_test_split(stft_images, labels, test_size=0.3, random_state=42)
X_train_vmd_stft, X_test_vmd_stft, _, _ = train_test_split(vmd_stft_images, labels, test_size=0.3, random_state=42)

# Defining all the CNN models to be used
def EEGNet(input_shape):
    model = Sequential([
        Conv2D(8, (1, 64), padding='same', input_shape=input_shape),
        BatchNormalization(),
        DepthwiseConv2D((1, 8), depth_multiplier=2, padding='same'),
        BatchNormalization(),
        Activation('elu'),
        AveragePooling2D((1, 4)),
        Dropout(0.25),
        SeparableConv2D(16, (1, 16), padding='same'),
        BatchNormalization(),
        Activation('elu'),
        AveragePooling2D((1, 8)),
        Dropout(0.5),
        Flatten(),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def DeepConvNet(input_shape):
    model = Sequential([
        Conv2D(25, (1, 10), padding='same', input_shape=input_shape),
        Conv2D(25, (1, 10), padding='same'),
        BatchNormalization(),
        Activation('elu'),
        MaxPooling2D((1, 3)),
        Conv2D(50, (1, 10), padding='same'),
        BatchNormalization(),
        Activation('elu'),
        MaxPooling2D((1, 3)),
        Conv2D(100, (1, 10), padding='same'),
        BatchNormalization(),
        Activation('elu'),
        MaxPooling2D((1, 3)),
        Flatten(),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def AlexNet(input_shape):
    model = Sequential([
        Conv2D(96, (11, 11), strides=(4, 4), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((3, 3), strides=(2, 2)),
        Conv2D(256, (5, 5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((3, 3), strides=(2, 2)),
        Conv2D(384, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(384, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def LeNet(input_shape):
    model = Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Evaluating all CNN models
models = {
    "EEGNet": EEGNet,
    "DeepConvNet": DeepConvNet,
    "AlexNet": AlexNet,
    "LeNet": LeNet
}

# Store the  accuracy for each model
accuracy_results = {}

for model_name, model_fn in models.items():
    _accuracy = 0.0  # Track the  accuracy for this model

    # Train and evaluate with STFT
    input_shape_stft = (X_train_stft.shape[1], X_train_stft.shape[2], 1)
    model = model_fn(input_shape_stft)
    model.fit(X_train_stft, y_train, validation_data=(X_test_stft, y_test), epochs=10, batch_size=32)
    y_pred_stft = np.argmax(model.predict(X_test_stft), axis=1)
    acc_stft = accuracy_score(y_test, y_pred_stft)
    _accuracy = max(_accuracy, acc_stft)  # Update with  STFT accuracy

    # Train and evaluate with VMD+STFT
    input_shape_vmd_stft = (X_train_vmd_stft.shape[1], X_train_vmd_stft.shape[2], X_train_vmd_stft.shape[3])
    model = model_fn(input_shape_vmd_stft)
    model.fit(X_train_vmd_stft, y_train, validation_data=(X_test_vmd_stft, y_test), epochs=10, batch_size=32)
    y_pred_vmd_stft = np.argmax(model.predict(X_test_vmd_stft), axis=1)
    acc_vmd_stft = accuracy_score(y_test, y_pred_vmd_stft)
    _accuracy = max(_accuracy, acc_vmd_stft)  # Update with  VMD+STFT accuracy

    # Store the best accuracy for this model
    accuracy_results[model_name] = _accuracy

# Print the  accuracy for each model
for model_name, accuracy in accuracy_results.items():
    print(f" accuracy of {model_name}: {accuracy:.2f}")
    
# Find the model with the overall  accuracy
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model_accuracy = accuracy_results[best_model_name]
print(f"\nFinal best model: {best_model_name} with accuracy: {best_model_accuracy:.2f}")
