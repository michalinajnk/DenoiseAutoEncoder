import os
import json
import re


def rename_frames(directory, ranges, postfix, extensions):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name, file_ext = os.path.splitext(file)

            # Extract the three-digit number from the file name
            match = re.search(r"\d{3}", file_name)
            if match:
                frame_number = match.group()
            else:
                # Skip files that don't have a three-digit number
                continue

            pairs = list(zip(ranges[::2], ranges[1::2]))
            for start_range, end_range in pairs:
                if start_range <= int(frame_number) <= end_range:
                    new_file_name = f"{frame_number}_{postfix}{file_ext}"
                    new_file_path = os.path.join(root, new_file_name)
                    os.rename(os.path.join(root, file), new_file_path)
                    break
            else:
                new_file_name = f"{frame_number}{file_ext}"
                new_file_path = os.path.join(root, new_file_name)
                os.rename(os.path.join(root, file), new_file_path)

    # Remove files with extensions not in the specified list
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext.lower() not in extensions:
                file_path = os.path.join(root, file)
                os.remove(file_path)


if __name__ == "__main__":
    json_file_path = "test_labels.json"
    postfix = "anomaly"
    extensions = ['.tif', '.bmp']

    with open(json_file_path) as json_file:
        data = json.load(json_file)

        for section_name, section_data in data.items():
            for test_name, ranges in section_data.items():
                test_directory = os.path.join(section_name, "Test", test_name)
                print(f"Renaming frames from {test_directory}")
                rename_frames(test_directory, ranges, postfix, extensions)
