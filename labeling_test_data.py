import os
import json

def rename_frames(directory, ranges, postfix, extensions):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name, file_ext = os.path.splitext(file)

            if file_ext.lower() not in extensions:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                continue
            frame_number = int(file_name)
            pairs = list(zip(ranges[::2], ranges[1::2]))
            for start_range, end_range in pairs:
                if start_range <= frame_number <= end_range:
                    new_file_name = f"{file_name}_{postfix}{file_ext}"
                    new_file_path = os.path.join(root, new_file_name)
                    os.rename(os.path.join(root, file), new_file_path)
                    break

if __name__ == "__main__":
    json_file_path = "test_labels.json"
    postfix = "anomaly"
    extensions = ['.tif', '.bmp']

    with open(json_file_path) as json_file:
        data = json.load(json_file)

        for section_name, section_data in data.items():
            for test_name, ranges in section_data.items():
                test_directory = os.path.join(section_name, "Test", test_name)
                rename_frames(test_directory, ranges, postfix, extensions)
