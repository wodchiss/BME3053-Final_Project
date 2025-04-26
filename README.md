_# BME3053C_Final_Project

This project focuses on predicting cell counts using machine learning models (Logistic Regression) based on microscopy image data from the BBBC005 dataset.

---

## **Dataset**
The dataset used is from the Broad Bioimage Benchmark Collection:

- [BBBC005 Dataset](https://bbbc.broadinstitute.org/BBBC005)

Download the dataset, unzip it, and organize it as follows inside the project:

```
BME3053C_Final_Project/
└── BBBC005_v1/
    ├── train_images/
    ├── train_masks/
    ├── test_images/
    ├── test_masks/
    └── preprocessed/ (created after preprocessing)
```

---

## **Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/wodchiss/BME3053C_Final_Project.git
   cd BME3053C-Final_Project
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

## **Running the Code**

1. **Preprocess the Data**
   Preprocess the raw images and create train/test datasets:
   ```bash
   python src/preprocess_data.py
   ```
   You should see:
   ```
   X_train_scaled.npy, X_test_scaled.npy, y_train.npy, y_test.npy
   ```
   saved inside `data`.

2. **Train the Model**
   Train a Logistic Regression model:
   ```bash
   python src/train.py
   ```
   Training metrics and a model performance summary will be printed.
3. **Testing the Model**
   After training, you can test the model and generate an Actual vs. Predicted plot:
   ***bash
   python src/test_model.py
   ***
   This script loads the trained model and test data.

    It predicts on the test set and creates a scatter plot comparing actual vs. predicted cell     counts.

    The plot will be saved inside the plots/ directory.
4. ***Evaluate the Model***\
   To compute evaluation metrics (accuracy, precision, recall, F1 score):
   *** bash
   python src/evaluate_data.py
   ***
   This script loads the test data and the trained model.

    It prints the performance metrics to the terminal and the confusion matrix to the plots/        directory.

    Metrics help you assess how well the model generalizes to unseen data.
   
   

---

## **Repository Structure**

```
BME3053C_Final_Project/
├── BBBC005_v1/
│   ├── train_images/
│   ├── train_masks/
│   ├── test_images/
│   ├── test_masks/
│   └── preprocessed/
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       ├── y_train.npy
│       └── y_test.npy
├── src/
│   ├── preprocess.py
│   └── train_logistic_regression.py
├── README.md
├── requirements.txt
```

---

## **References**
- [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC005)
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---_
