import os
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Conv3DTranspose, ConvLSTM2D
from tensorflow.keras.models import Model

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

    frame_height, frame_width, _ = first_frame.shape

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

    return optical_flow_volume


def calculate_optical_flow_datasets(paths):
    optical_flow_dataset = []

    for dataset_path in paths:
        # Loop through the directories in the dataset directory
        for video_directory in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video_directory)
            optical_flow_video = calculate_optical_flow(video_path)
            optical_flow_dataset.append(optical_flow_video)

    # Convert the list of optical flow arrays to a numpy array
    optical_flow_dataset = np.array(optical_flow_dataset)

    return optical_flow_dataset



def create_AE(frames, height, width, channels):
    input_shape = (frames, height, width, channels)  # Define the input shape based on your video data

    # Encoder
    input_data = Input(shape=input_shape)
    encoded = Conv3DTranspose(128, (10, 10, 10), strides=(2, 2, 2), padding='same', activation='relu')(input_data)
    encoded = Conv3DTranspose(64, (6, 6, 6), strides=(2, 2, 2), padding='same', activation='relu')(encoded)
    encoded = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(encoded)
    encoded = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(encoded)
    encoded = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(encoded)

    # Decoder
    decoded = Conv3DTranspose(128, (6, 6, 6), strides=(2, 2, 2), padding='same', activation='relu')(encoded)
    decoded = Conv3DTranspose(1, (10, 10, 10), strides=(2, 2, 2), padding='same', activation='sigmoid')(decoded)

    # Autoencoder model
    return Model(input_data, decoded)


def train_model(autoencoder, optical_flow_dataset, batch_size, num_epochs):
    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.fit(optical_flow_dataset, optical_flow_dataset, batch_size=batch_size, epochs=num_epochs)
    return autoencoder



# Example usage
dataset_path = 'path/to/optical_flow_dataset'
optical_flow_dataset = calculate_optical_flow_datasets(["UCSDped1/Train/", "UCSDped2/Train/"])

frames, height, width, channels = optical_flow_dataset.shape[1:]
autoencoder = create_AE(frames, height, width, channels)

batch_size = 16
num_epochs = 10
trained_autoencoder = train_model(autoencoder, optical_flow_dataset, batch_size, num_epochs)
