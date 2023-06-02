import os
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Conv3DTranspose, ConvLSTM2D
from tensorflow.keras.models import Model

def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an empty array to store the optical flow map volume
    optical_flow_volume = np.zeros((8, frame_height, frame_width, 3), dtype=np.float32)

    # Read the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_norm = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)

        # Store the normalized optical flow in the optical flow map volume
        if frame_index < 8:
            optical_flow_volume[frame_index] = flow_norm

        prev_gray = gray
        frame_index += 1

    cap.release()
    return optical_flow_volume


def calculate_optical_flow_dataset(dataset_path):
    optical_flow_dataset = []

    # Loop through the video files in the dataset directory
    for video_file in os.listdir(dataset_path):
        video_path = os.path.join(dataset_path, video_file)
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
optical_flow_dataset = calculate_optical_flow_dataset(dataset_path)

frames, height, width, channels = optical_flow_dataset.shape[1:]
autoencoder = create_AE(frames, height, width, channels)

batch_size = 16
num_epochs = 10
trained_autoencoder = train_model(autoencoder, optical_flow_dataset, batch_size, num_epochs)
