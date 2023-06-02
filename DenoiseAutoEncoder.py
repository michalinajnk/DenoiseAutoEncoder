import os
import cv2
import numpy as np
from keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.layers import Input, Conv3DTranspose, ConvLSTM2D
from tensorflow.keras.models import Model

frame_height = 0
frame_width = 0


def calculate_optical_flow(frame_directory):
    frame_files = sorted(file for file in os.listdir(frame_directory) if file.endswith('.tif'))
    print(frame_files)  # Add this line to check the frame files

    frame_count = len(frame_files)
    print(frame_count)  # Add this line to check the frame count

    # Read the first frame to get its dimensions
    first_frame = cv2.imread(os.path.join(frame_directory, frame_files[0]))
    if first_frame is None:
        print("Error: Failed to read the first frame")
        return None

    frame_height, frame_width, channels = first_frame.shape

    # Create an empty array to store the optical flow map volume
    optical_flow_volume = np.zeros((frame_count - 1, frame_height, frame_width, 2), dtype=np.float32)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    for frame_index in range(1, frame_count):
        frame = cv2.imread(os.path.join(frame_directory, frame_files[frame_index]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Store the optical flow in the optical flow map volume
        optical_flow_volume[frame_index - 1, :, :, 0] = flow[..., 0]
        optical_flow_volume[frame_index - 1, :, :, 1] = flow[..., 1]

        prev_gray = gray

    return optical_flow_volume, frame_height, frame_width, channels


def calculate_optical_flow_datasets(paths):
    optical_flow_dataset = []
    max_frames = 0
    max_height = 0
    max_width = 0
    channels = 0

    for dataset_path in paths:
        # Loop through the directories in the dataset directory
        for video_directory in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video_directory)
            optical_flow_video, height, width, channels = calculate_optical_flow(video_path)
            optical_flow_dataset.append(optical_flow_video)

            # Update the maximum dimensions
            max_frames = max(max_frames, optical_flow_video.shape[0])
            max_height = max(max_height, optical_flow_video.shape[1])
            max_width = max(max_width, optical_flow_video.shape[2])

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


def create_AE(frames, height, width, channels):
    input_shape = (frames, height, width, channels)  # Define the input shape based on your optical flow data

    # Encoder
    input_data = Input(shape=input_shape)
    encoded = Conv3D(128, (3, 3, channels), activation='relu', padding='same')(input_data)
    encoded = MaxPooling3D((2, 2, 1), padding='same')(encoded)
    encoded = Conv3D(64, (3, 3, channels), activation='relu', padding='same')(encoded)
    encoded = MaxPooling3D((2, 2, 1), padding='same')(encoded)
    encoded = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(encoded)
    encoded = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(encoded)
    encoded = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(encoded)

    # Decoder
    decoded = Conv3DTranspose(64, (3, 3, channels), strides=(2, 2, 1), padding='same', activation='relu')(encoded)
    decoded = Conv3DTranspose(128, (3, 3, channels), strides=(2, 2, 1), padding='same', activation='relu')(decoded)
    decoded = Conv3D(2, (3, 3, channels), activation='sigmoid', padding='same')(decoded)

    # Autoencoder model
    return Model(input_data, decoded)





def resize_optical_flow_dataset(optical_flow_dataset):
    desired_frames = 199
    desired_height = 240
    desired_width = 360
    desired_channels = 2

    resized_optical_flow_dataset = []

    for opt_flow in optical_flow_dataset:
        frames, height, width, channels = opt_flow.shape
        if frames != desired_frames or height != desired_height or width != desired_width or channels != desired_channels:
            resized_opt_flow = cv2.resize(opt_flow, (desired_width, desired_height))
            resized_opt_flow = np.expand_dims(resized_opt_flow, axis=0)  # Add an extra dimension for frames
            resized_optical_flow_dataset.append(resized_opt_flow)
        else:
            resized_optical_flow_dataset.append(opt_flow)

    resized_optical_flow_dataset = np.concatenate(resized_optical_flow_dataset, axis=0)

    return resized_optical_flow_dataset


def train_model(autoencoder, optical_flow_dataset, batch_size, num_epochs):

    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.fit(
        optical_flow_dataset,
        optical_flow_dataset,
        batch_size=batch_size,
        epochs=num_epochs
    )
    return autoencoder


optical_flow_dataset, height, width, channels, frames = calculate_optical_flow_datasets(["UCSDped1/Train/", "UCSDped2/Train/"])
print("{}, {}, {}, {}".format(height, width, channels, frames))

autoencoder = create_AE(frames, height, width, channels)

batch_size = 16
num_epochs = 10
trained_autoencoder = train_model(autoencoder, optical_flow_dataset, batch_size, num_epochs)
