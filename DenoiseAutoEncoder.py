import os
from tensorflow.keras.models import save_model, load_model

import cv2
import numpy as np
from keras.layers import Conv3D, MaxPooling3D, Reshape, Lambda, Flatten, Dense
from tensorflow.keras.layers import Input, Conv3DTranspose, ConvLSTM2D
from tensorflow.keras.models import Model

frame_height = 0
frame_width = 0
model_filename = 'autoencoder.sav'


def calculate_optical_flow(video_path, max_height, max_width, max_frames):
    frame_files = sorted(file for file in os.listdir(video_path) if file.endswith('.tif'))
    frame_count = len(frame_files)

    # Truncate or pad frames to match the maximum frame count
    if frame_count > max_frames:
        frame_files = frame_files[:max_frames]
    elif frame_count < max_frames:
        padding_frames = max_frames - frame_count
        last_frame = frame_files[-1]
        frame_files.extend([last_frame] * padding_frames)

    # Create an empty array to store the optical flow map volume
    optical_flow_volume = np.zeros((max_frames, max_height, max_width, 2), dtype=np.float32)

    prev_gray = cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_files[0])), cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (max_width, max_height))

    for frame_index in range(1, max_frames):
        frame = cv2.imread(os.path.join(video_path, frame_files[frame_index]))
        frame = cv2.resize(frame, (max_width, max_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Store the optical flow in the optical flow map volume
        optical_flow_volume[frame_index - 1, :, :, 0] = flow[..., 0]
        optical_flow_volume[frame_index - 1, :, :, 1] = flow[..., 1]

        prev_gray = gray

    return optical_flow_volume


def calculate_optical_flow_datasets(paths):
    optical_flow_dataset = []
    max_frames = 0
    max_height = 0
    max_width = 0
    channels = 0

    for dataset_path in paths:
        for video_directory in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video_directory)

            # Calculate the maximum height and width of frames in the dataset
            max_frame_height = 0
            max_frame_width = 0
            for frame_file in os.listdir(video_path):
                frame = cv2.imread(os.path.join(video_path, frame_file))
                if frame is None:
                    continue  # Skip frames that couldn't be read

                frame_height, frame_width, _ = frame.shape
                max_frame_height = max(max_frame_height, frame_height)
                max_frame_width = max(max_frame_width, frame_width)

            optical_flow_video = calculate_optical_flow(video_path, max_frame_height, max_frame_width, 200)
            optical_flow_dataset.append(optical_flow_video)

            # Update the maximum dimensions
            max_frames = max(max_frames, optical_flow_video.shape[0])
            print("Max frames: {} ".format(max_frames))
            max_height = max(max_height, optical_flow_video.shape[1])
            max_width = max(max_width, optical_flow_video.shape[2])
            channels = optical_flow_video.shape[3]

    # Create a new list to store resized and padded optical flow datasets
    resized_optical_flow_dataset = []

    # Resize and pad the optical flow datasets to have the same shape
    for opt_flow in optical_flow_dataset:
        frames, height, width, _ = opt_flow.shape
        if frames < max_frames or height < max_height or width < max_width:
            # Pad the optical flow dataset with zeros
            padding_frames = max_frames - frames
            padding_height = max_height - height
            padding_width = max_width - width
            padded_opt_flow = np.pad(
                opt_flow,
                [(0, padding_frames), (0, padding_height), (0, padding_width), (0, 0)],
                mode='constant'
            )
            resized_optical_flow_dataset.append(padded_opt_flow)
        else:
            resized_optical_flow_dataset.append(opt_flow)

    return np.array(resized_optical_flow_dataset), max_height, max_width, channels, max_frames

def create_AE(windows):

    # Encoder
    input_data = Input(shape=windows[0].shape)

    encoded = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_data)
    saved_shape=tuple(encoded.shape.as_list()[1:])
    print("Encoded shape : {}".format(saved_shape))

    # Apply MaxPooling3D

    encoded = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense(32, activation='relu')(encoded)  # Introduce a Dense layer to reduce the size
    encoded = Reshape(saved_shape)(encoded)
    encoded = MaxPooling3D((2, 2, 1), padding='same')(encoded)
    encoded = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(encoded)
    encoded = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(encoded)
    encoded = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(encoded)

    # Decoder
    decoded = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(encoded)
    decoded = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(decoded)
    decoded = Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 1), padding='same', activation='relu')(decoded)
    decoded = Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 1), padding='same', activation='relu')(decoded)
    decoded = Conv3D(2, (3, 3, 3), activation='sigmoid', padding='same')(decoded)
    print("Expected input shape:", windows[0].shape)
    # Autoencoder model
    return Model(input_data, decoded)



def train_model(autoencoder, dataset, batch_size, num_epochs):
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    for i, window in enumerate(dataset):
        print("Shape of window", i + 1, ":", window.shape)

    autoencoder.fit(
        dataset,
        dataset,
        batch_size=batch_size,
        epochs=num_epochs
    )
    save_model(autoencoder, model_filename)
    return autoencoder


optical_flow_dataset, height, width, channels, frames = calculate_optical_flow_datasets(
    ["UCSDped1/Train/", "UCSDped2/Train/"])  # sciezka do podfolderow zawierajacych film juz podzielony na klatki


print("Size of optical flow dataset: {}".format(optical_flow_dataset.shape))
num_frames = optical_flow_dataset.shape[0]
window_size = 8
# Calculate the minimum and maximum pixel values in the image dataset
min_value = np.min(optical_flow_dataset)
max_value = np.max(optical_flow_dataset)

# Normalize the image dataset
normalized_optical_flow_dataset = (optical_flow_dataset - min_value) / (max_value - min_value)

# Calculate the maximum height, width, and frames in the normalized dataset
max_height = normalized_optical_flow_dataset.shape[1]
max_width = normalized_optical_flow_dataset.shape[2]
max_frames = normalized_optical_flow_dataset.shape[0]

# Create windows
windows = []
for i in range(max_frames - window_size + 1):
    window = normalized_optical_flow_dataset[i: i + window_size]
    windows.append(window)
    print("Window shape: {}".format(window.shape))

autoencoder = create_AE(windows)
batch_size = 2
num_epochs = 3

trained_autoencoder = train_model(autoencoder, windows, 2, num_epochs)
# Load the model
loaded_autoencoder = load_model(model_filename)