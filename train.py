
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, ConvLSTM3D, Reshape
from tensorflow.python.framework.ops import convert_to_tensor

from Autoencoder import Autoencoder
import os
import cv2
import numpy as np

optical_flows = []
height, width = 180, 320
desired_no_channels = 3

def resize_frames(frames, target_size):
    resized_frames = []
    for frame in frames:
        frame_data = cv2.imread(frame)
        if frame_data is None:
            print(f"Error loading frame: {frame}")
            continue
        resized_frame = cv2.resize(frame_data, target_size, interpolation=cv2.INTER_LINEAR)
        resized_frames.append(resized_frame)
    return resized_frames


def get_movie(path, max_frames):
    frame_files = sorted(file for file in os.listdir(path) if file.endswith('.tif'))
    num_frames = len(frame_files)

    # Duplicate frames if the number of frames is smaller than max_frames
    if num_frames < max_frames:
        duplication_factor = max_frames // num_frames
        remaining_frames = max_frames % num_frames
        duplicated_frames = frame_files * duplication_factor + frame_files[:remaining_frames]
        frame_files += duplicated_frames
    elif num_frames > max_frames:
        frame_files = frame_files[:max_frames]
    else:
        frame_files = frame_files

    resized_frames = resize_frames([os.path.join(path, file) for file in frame_files], (320, 180))

    return np.array(resized_frames)

def get_data(paths, max_frames):
    dataset = []
    for dataset_path in paths:
        for video_directory in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video_directory)
            video = get_movie(video_path, max_frames)
            dataset.append(video)
            print(video.shape)
    dataset = [video for video in dataset if
                          video.shape == (max_frames, height, width, desired_no_channels)]

    dataset = np.stack(dataset)
    return dataset


def calculate_max_dimensions(paths):
    max_height, max_width, max_frames = 0, 0, 0
    for dataset_path in paths:
        for video_directory in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video_directory)
            frame_files = [file for file in os.listdir(video_path) if file.endswith('.tif')]

            max_frames = max(max_frames, len(frame_files))

            for frame_file in frame_files:
                frame = cv2.imread(os.path.join(video_path, frame_file), cv2.IMREAD_UNCHANGED)

                if frame is None:
                    print(f"Error loading frame: {frame_file}")
                    continue

                max_height, max_width = max(max_height, frame.shape[0]), max(max_width, frame.shape[1])

    return max_height, max_width, max_frames


def generate_optical_flow(frame_files, max_height, max_width):
    # Create an empty array to store the optical flow map volume
    optical_flow_volume = np.zeros((len(frame_files), max_height, max_width, desired_no_channels), dtype=np.float32)
    print("Shape of a frame: {}".format(frame_files[0].shape))

    prev_gray = cv2.cvtColor(frame_files[0], cv2.COLOR_BGR2GRAY)

    for frame_index in range(1, len(frame_files)):
        frame = frame_files[frame_index]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Store the optical flow in the optical flow map volume
        optical_flow_volume[frame_index - 1, :, :, 0] = flow[..., 0]
        optical_flow_volume[frame_index - 1, :, :, 1] = flow[..., 1]

        # Set the third channel !TO BE CHANGED!
        optical_flow_volume[frame_index - 1, :, :, 2] = flow[..., 1]

        prev_gray = gray

    # return flow_volume_map_for_a_video
    print(f"Optical flow volume shape: {optical_flow_volume.shape}")
    return optical_flow_volume


"""
Calculate optical flow datasets for multiple video datasets.

Args:
    paths (List[str]): List of paths to the directories containing video datasets.

Returns:
    optical_flow_dataset (np.ndarray): Resized and padded optical flow datasets.
    max_height (int): Maximum height of frames in the datasets.
    max_width (int): Maximum width of frames in the datasets.
    channels (int): Number of channels in the optical flow datasets.
    max_frames (int): Maximum number of frames in the datasets.

"""
def calculate_optical_flows(data, max_frame_height, max_frame_width):
    for video in data:
        optical_flow_video = generate_optical_flow(video, max_frame_height, max_frame_width)
        optical_flows.append(optical_flow_video)

    print(f"dataset shape: {np.array(optical_flows).shape}")
    return np.array(optical_flows)


def update_maximum_dimensions(max_frames, max_height, max_width, optical_flow_video):
    # Update the maximum dimensions
    max_frames = max(max_frames, optical_flow_video.shape[0])
    print("Max frames: {} ".format(max_frames))
    max_height = max(max_height, optical_flow_video.shape[1])
    max_width = max(max_width, optical_flow_video.shape[2])
    channels = optical_flow_video.shape[3]
    return channels, max_frames, max_height, max_width


def divide_into_windows(optical_flow_dataset, window_size):
    windowed_dataset = []
    for optical_flow_video in optical_flow_dataset:
        windows = divide_into_windows_video(optical_flow_video, window_size)
        print(f"Window shape: {windows.shape}")
        windowed_dataset.append(windows)
    return windowed_dataset

def divide_into_windows_video(video, window_size):
    num_frames = video.shape[0]
    windows = [video[i:i + window_size] for i in range(0, num_frames, window_size)]
    compatible_windows = [window for window in windows if window.shape == (window_size, height, width, desired_no_channels)]
    np_windows = np.stack(compatible_windows)
    return np_windows


# Define the model architecture
from keras.layers import TimeDistributed

def build_lstm_autoencoder(shape):
    print("Shape in the input_data {}".format(shape))
    input_data = Input(shape=shape)
    encoded = Conv3D(filters=128, kernel_size=(10, 10, 3), activation='relu', padding='same', strides=(2, 2, 1))(input_data)
    print("Shape in the after first conv3D layer {}".format(encoded.shape))
    encoded = Conv3D(filters=64, kernel_size=(6, 6, 3), activation='relu', padding='same', strides=(2, 2, 1))(encoded)
    print("Shape in the after second conv3D layer {}".format(encoded.shape))

    encoded = ConvLSTM3D(filters=64, kernel_size=(3, 3, 3), padding='same', return_sequences=True)(encoded)
    print("Shape in the after first ConvLSTM layer {}".format(encoded.shape))
    encoded = ConvLSTM3D(filters=32, kernel_size=(3, 3, 3), padding='same', return_sequences=True)(encoded)
    print("Shape in the after second ConvLSTM layer {}".format(encoded.shape))
    encoded = ConvLSTM3D(filters=64, kernel_size=(3, 3, 3), padding='same', return_sequences=True)(encoded)
    print("Shape in the after third ConvLSTM layer {}".format(encoded.shape))
    # Decoder
    decoded = Conv3DTranspose(filters=128, kernel_size=(6, 6, 3), strides=(2, 2, 1), padding='same', activation='relu')(encoded[:, 0, ...])
    print("Shape in the after first Conv3DTranspose layer {}".format(decoded.shape))
    decoded = Conv3DTranspose(filters=1, kernel_size=(10, 10, 3), strides=(2, 2, 1), padding='same', activation='relu')(decoded)
    print("Shape in the after second Conv3DTranspose layer {}".format(decoded.shape))
    decoded = Conv3D(filters=3, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(decoded)
    print("Shape in the after third Conv3DTranspose layer {}".format(decoded.shape))
    model = Model(input_data, decoded)
    model.compile(optimizer="adam", loss="mse")
    # Autoencoder model
    return model

def data_generator(windowed_optical_flows, batch_size):
    num_samples = len(windowed_optical_flows)
    num_batches = num_samples // batch_size

    while True:
        np.random.shuffle(windowed_optical_flows)  # Shuffle the data before each epoch
        for batch_index in range(num_batches):
            batch_start = batch_index * batch_size
            batch_end = (batch_index + 1) * batch_size
            batch_data = windowed_optical_flows[batch_start:batch_end]

            yield batch_data, batch_data

def train_model(model, filename, windowed_optical_flows, optimizer):
    x_train = np.array(windowed_optical_flows)
    model.fit(x_train, x_train, epochs=10, batch_size=32)
    # Save the trained model
    model.save(f"{filename}.h5")
    return model


if __name__ == "__main__":

    paths = ["UCSDped1/Train/", "UCSDped2/Train/"]
    #max_height, max_width, max_frames = calculate_max_dimensions(paths)
    #x_train = get_movie("Train001/", 200)
    x_train = get_data(paths, 200)
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of one video: {np.array(x_train[0]).shape}")

    optical_flows = calculate_optical_flows(x_train, 180, 320)
    #optical_flows = generate_optical_flow(x_train, 180, 320) #one video
    print(optical_flows.shape)
    # print(optical_flows.shape)
    #print(np.array(optical_flows[0]).shape)

    # input shape of one video -> 200,240,260,2

    #####################################################################

    window_size = 8
    windowed_optical_flows = divide_into_windows(optical_flows, window_size)
    #windowed_optical_flows = divide_into_windows_video(optical_flows, window_size)


    print("Windowed optical flow data, shape: {}".format(np.array(windowed_optical_flows).shape))  # (num_windows, )

    # Update input_shape to match the new shape
    input_shape = (window_size, height, width, desired_no_channels)
    stored_input_shape = windowed_optical_flows[0].shape  # (window_size, height, width, channels)
    print("The input shape is correct: {} expected shape: {}  real shape: {}".format((input_shape == stored_input_shape), input_shape, stored_input_shape))

    # Build the LSTM autoencoder
    lstm_autoencoder = build_lstm_autoencoder(windowed_optical_flows[0].shape)

    # Train the model
    trained_model = train_model(lstm_autoencoder, "lstm_autoencoder", windowed_optical_flows)