import os
import numpy as np


if __name__ == '__main__':
    # Directory where the files are stored
    base_directory = "./dataset/data/modelnet_ts/train/"

    # List all the files in the directory
    all_files = os.listdir(base_directory)

    # Filter out the relevant files
    relevant_files = [f for f in all_files if "label_40.npy" in f]

    for file in relevant_files:
        file_path = os.path.join(base_directory, file)

        # Load the file
        data = np.load(file_path)

        # Convert to float32
        data = data.astype(np.float32)

        # Save the data back
        np.save(file_path, data)

    print(f"Processed {len(relevant_files)} files.")
