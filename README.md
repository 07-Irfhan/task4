# Iris Flower Classification using Logistic Regression

This project implements a machine learning model using **Logistic Regression** to classify iris flowers into three species based on their physical characteristics. The project uses the built-in Iris dataset from `scikit-learn`.

---

## 📊 Dataset

The Iris dataset contains 150 samples of iris flowers with the following features:

- `sepal length (cm)`
- `sepal width (cm)`
- `petal length (cm)`
- `petal width (cm)`
- `target` (species encoded as 0, 1, or 2)

No external dataset is needed; the dataset is loaded from `sklearn.datasets`.

---

## 🚀 Getting Started

### 1. Requirements

Install the required Python libraries using pip:

```bash
pip install pandas scikit-learn
Run the Code
You can run the script using:

bash
Copy
Edit
python iris_classifier.py
🔍 What the Code Does
Loads the Iris dataset from sklearn.datasets

Prepares the features and target labels

Splits the dataset into training and testing sets

Standardizes the features using StandardScaler

Trains a Logistic Regression model on the scaled data

Evaluates the model using:

Accuracy Score

Confusion Matrix

Classification Report

📈 Sample Output
lua
Copy
Edit
Accuracy: 1.0

Confusion Matrix:
 [[10  0  0]
  [ 0  9  1]
  [ 0  0 10]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      0.90      0.95        10
           2       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
📁 Optional: Export Dataset to CSV
If you'd like to save the dataset as a CSV:

python
Copy
Edit
df.to_csv("iris_dataset.csv", index=False)

📚 References
scikit-learn documentation

Iris dataset info

