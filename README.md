# BME3053C_Final_Project

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
   git clone https://github.com/your-username/BME3053C_Final_Project.git
   cd BME3053C_Final_Project
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
   python src/preprocess.py
   ```
   You should see:
   ```
   X_train_scaled.npy, X_test_scaled.npy, y_train.npy, y_test.npy
   ```
   saved inside `BBBC005_v1/preprocessed/`.

2. **Train the Model**
   Train a Logistic Regression model:
   ```bash
   python src/train_logistic_regression.py
   ```
   Training metrics and a model performance summary will be printed.

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

---
