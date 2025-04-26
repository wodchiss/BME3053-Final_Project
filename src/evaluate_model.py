import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from model import load_model
import glob
from skimage.io import imread

def evaluate_model(model_path, X_test, y_test, images=None, save_dir="plots"):
    """
    Evaluate the trained model and visualize results.

    Args:
        model_path (str): Path to the saved model.
        X_test (array-like): Test features.
        y_test (array-like): True labels for the test set.
        images (array-like, optional): Images corresponding to the test set for visualization.
        save_dir (str): Directory to save the plots.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load the trained model
    model = load_model(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("📊 Evaluation Metrics:")
    print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
    print(f"📈 R-squared (R²): {r2:.2f}")

    # Visualize predicted vs actual values (scatter plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs. Actual Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "predicted_vs_actual.png"))
    plt.show()

    # Create a confusion matrix (for regression, bin predictions into ranges)
    bins = np.linspace(y_test.min(), y_test.max(), 11)  # Create 10 intervals
    y_test_binned = np.digitize(y_test, bins[:-1])
    y_pred_binned = np.digitize(y_pred, bins[:-1])

    cm = confusion_matrix(y_test_binned, y_pred_binned)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Bin {i}" for i in range(1, len(bins))])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Binned Predictions)")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

    # Calculate and print Accuracy, Precision, Recall
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples

    recalls = []
    precisions = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recalls.append(recall)
        precisions.append(precision)
        print(f"🔹 Bin {i+1}: Recall = {recall:.4f}, Precision = {precision:.4f}")

    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)

    print("\n✅ Overall Evaluation (Binned Predictions):")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Average Recall: {avg_recall:.4f}")
    print(f"  - Average Precision: {avg_precision:.4f}")

    # Visualize images with predicted and actual values (if images are provided)
    if images is None or len(images) != len(y_test):
        print("⚠️ Images are not provided or do not match the number of test samples. Skipping image visualization.")
        return

    num_images = min(5, len(images))  # Show up to 5 images
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "image_predictions.png"))
    plt.show()

    # Print some images with their predicted and actual values
    print("\nSample Images with Predicted and Actual Values:")
    for i in range(num_images):
        print(f"Image {i + 1}:")
        print(f"  Actual Value: {y_test[i]}")
        print(f"  Predicted Value: {y_pred[i]:.2f}")
        print(f"  Image Shape: {images[i].shape}")
        print("-" * 30)

# Example usage
if __name__ == "__main__":
    # Load test data
    X_test = pd.read_csv("data/X_test_scaled.csv").values
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    # Load corresponding images (if available)
    image_paths = glob.glob("/workspaces/BME3053-Final_Project/data/BBBC005_v1_ground_truth/*.TIF")
    print(len(image_paths))
    if not image_paths:
        raise FileNotFoundError("No .tif files found in the specified folder: /workspaces/BME3053-Final_Project/data/BBBC005_v1_ground_truth/")

    np.random.seed(42)
    selected_image_paths = sorted(image_paths)
    images = [imread(path) for path in selected_image_paths]

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("NaN in X_test:", np.isnan(X_test).any())
    print("NaN in y_test:", np.isnan(y_test).any())

    # Path to the saved model
    model_path = "model/random_forest_model.pkl"

    # Evaluate the model
    evaluate_model(model_path, X_test, y_test, images=images)
