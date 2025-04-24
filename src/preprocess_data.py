from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load the dataset
df = pd.read_csv("/workspaces/BME3053-Final_Project/data/BBBC005_results_bray.csv")  # Adjust the file path as needed

# Separate features and target
X = df.drop("Image_Metadata_ActualCellCount", axis=1)  # Replace with the actual target column name
y = df["Image_Metadata_ActualCellCount"]  # Replace with the actual target column name

# Handle categorical data (if any) - encoding categorical columns
# Example: Encoding 'Image_FileName_CellBody' and 'Image_FileName_Nuclei'
# If you have categorical columns, encode them using LabelEncoder
label_encoder = LabelEncoder()

categorical_columns = ['Image_FileName_CellBody', 'Image_FileName_Nuclei']  # Add more categorical columns if needed
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Normalize/scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the preprocessed data (optional)
pd.DataFrame(X_train).to_csv("data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test).to_csv("data/X_test_scaled.csv", index=False)
pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

print("Preprocessing completed and data saved.")
# Print data shape
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
