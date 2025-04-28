# BME3053C Final Project

This project focuses on predicting cell counts using machine learning models (Logistic Regression) based on microscopy image data from the BBBC005 dataset.

---

## Dataset

The dataset used is from the Broad Bioimage Benchmark Collection:

- Dataset link: https://bbbc.broadinstitute.org/BBBC005 (copy and paste me into browser)

Download the dataset, unzip it, and organize it as follows inside the project:

```
BME3053C_Final_Project/
├── data/
│   ├── BBBC005_v1_ground_truth.zip
│   ├── synthetic_2_ground_truth.zip
│   ├── BBBC005_results_bray.csv
├── README.md
├── requirements.txt
```

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/wodchiss/BME3053C_Final_Project.git
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   After downloading `BBBC005_v1_ground_truth.zip`, unzip it into a folder called `check`:
   ```bash
   unzip BBBC005_v1_ground_truth.zip -d check/
   ```

---

## Running the Code

1. **Preprocess the Data**  
   Preprocess the raw images and create train/test datasets:
   ```bash
   python src/preprocess_data.py
   ```
   You should see the following files saved inside the `data/` directory:
   ```
   X_train_scaled.npy, X_test_scaled.npy, y_train.npy, y_test.npy
   ```

2. **Train the Model**  
   Train a Logistic Regression model:
   ```bash
   python src/train.py
   ```
   Training metrics and a model performance summary will be printed.

3. **Test the Model**  
   After training, you can test the model and generate an Actual vs. Predicted plot:
   ```bash
   python src/test_model.py
   ```
   This script loads the trained model and test data. It predicts on the test set and creates a scatter plot comparing actual vs. predicted cell counts. The plot will be saved inside the `plots/` directory.

4. **Evaluate the Model**  
   To compute evaluation metrics (accuracy, precision, recall, F1 score):
   ```bash
   python src/evaluate_model.py
   ```
   This script loads the test data and the trained model. It prints the performance metrics to the terminal and saves the confusion matrix to the `plots/` directory. These metrics help assess how well the model generalizes to unseen data.

---

## Repository Structure

```
BME3053C_Final_Project/
├── data/
│   ├── BBBC005_v1_ground_truth.zip
│   ├── synthetic_2_ground_truth.zip
│   ├── BBBC005_results_bray.csv
│   ├── X_train_scaled.npy
│   ├── X_test_scaled.npy
│   ├── y_train.npy
│   ├── y_test.npy
├── model/
│   ├── random_forest_model.pkl
├── src/
│   ├── preprocess_data.py
│   ├── train.py
│   ├── test_model.py
│   └── evaluate_model.py
├── plots/
│   ├── actual_vs_predicted_plot.png
│   └── confusion_matrix.png
├── README.md
├── requirements.txt
```

---

## References
"We used the image set BBBC005v1 from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]."
- [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC005)
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

## this is the link to the YouTube video
https://youtu.be/SLg7avwkop0

## this is the powerpoint presentation
https://uflorida-my.sharepoint.com/:p:/g/personal/howecooper_ufl_edu/EZXZQATQw0RLp1eRgSkCEzIB1hvWE4yL2aJoOkTtKAnz-A?e=1lrLvr&wdLOR=c65658CCB-11F0-413C-A96D-25F4E5A03CA9
