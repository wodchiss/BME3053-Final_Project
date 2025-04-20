
######
import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

# === CONFIGURATION ===
ZIP_PATH = "../data/BBBC005_v1_ground_truth.zip"
CSV_PATH = "../data/BBBC005_results_bray.csv"
EXTRACT_DIR = "../data/BBBC005_v1"
IMAGE_SIZE = (64, 64)


def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted images to {extract_to}")
    else:
        print(f"Images already extracted to {extract_to}")


def load_dataset():
    # Extract image data if needed
    extract_zip(ZIP_PATH, EXTRACT_DIR)

    # Load labels CSV
    labels_df = pd.read_csv(CSV_PATH)

    # Build mapping of image filename to full path
    image_paths = {}
    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith(".TIF"):
                image_paths[file] = os.path.join(root, file)

    # Match filenames to labels
    X, y = [], []
    for _, row in labels_df.iterrows():
        filename = row["Image_FileName_Nuclei"]
        label = row["Image_Metadata_ActualCellCount"]
        if filename in image_paths:
            img = Image.open(image_paths[filename]).convert("L")
            img_resized = resize(np.array(img), IMAGE_SIZE, anti_aliasing=True)
            X.append(img_resized.flatten())
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"Processed {len(X)} images. Feature shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


if __name__ == "__main__":
    X, y = load_dataset()

