def parse_image_metadata(image_dir):
    import os
    import re

    pattern = re.compile(r'C(\d+)_F(\d+)_s(\d+)_w(\d+)\.TIF$', re.IGNORECASE)
    image_data = []

    for filename in os.listdir(image_dir):
        match = pattern.search(filename)
        if match:
            cell_count = int(match.group(1))
            image_path = os.path.join(image_dir, filename)
            image_data.append((image_path, cell_count))

    return image_data
######
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Constants
image_folder = "/mnt/data/BBBC005_v1/synthetic_2_ground_truth"
target_size = (64, 64)  # Resize images to 64x64 for classical ML

# Filter and map image filenames to full paths
image_paths = {
    os.path.basename(path): path
    for path in extracted_files
    if path.endswith(".TIF")
}

# Match images with labels from CSV
valid_data = []
for _, row in labels_df.iterrows():
    filename = row["Image_FileName_Nuclei"]
    label = row["Image_Metadata_ActualCellCount"]
    if filename in image_paths:
        valid_data.append((image_paths[filename], label))

# Prepare feature matrix X and label vector y
X, y = [], []
for img_path, label in valid_data:
    img = Image.open(img_path).convert("L")
    img_resized = resize(np.array(img), target_size, anti_aliasing=True)
    X.append(img_resized.flatten())
    y.append(label)

X = np.array(X)
y = np.array(y)

X.shape, y.shape

