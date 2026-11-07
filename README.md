# BITS M.Tech DSE - Machine Learning Assignment 2
**Student ID:** 2025NS04048  
**Course:** Machine Learning 

## a. Problem statement
The objective of this assignment is to build, evaluate, compare, and deploy a robust multi-model machine learning classification pipeline using a dataset comprising at least 12 features and at least 500 instances. 
The solution includes an interactive Streamlit web application hosted on Streamlit Community Cloud and fully version-controlled via GitHub.

## b. Dataset description
- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (sourced via scikit-learn)
- **Instances (Samples):** 569
- **Features:** 30 numerical attributes (computed from digitized images of fine needle aspirate of breast masses)
- **Target Variable:** Binary classification (Malignant = 0, Benign = 1)

## c. Github Repository Link
https://github.com/2025ns04048/ml-assignment-2

## d. Models used & Comparison Tables

### Evaluation Metrics Table:
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9877 | 0.9971 | 0.9861 | 0.9944 | 0.9902 | 0.9737 |
| **Decision Tree** | 0.9824 | 0.9831 | 0.9915 | 0.9804 | 0.9859 | 0.9627 |
| **KNN** | 0.9701 | 0.9948 | 0.9670 | 0.9860 | 0.9764 | 0.9360 |
| **Naive Bayes** | 0.9402 | 0.9889 | 0.9425 | 0.9636 | 0.9529 | 0.8716 |
| **Support Vector Machine** | 0.9824 | 0.9976 | 0.9806 | 0.9916 | 0.9861 | 0.9624 |

### Model Performance Observations Table:
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall accuracy (0.9877) and F1-score (0.9902) on the scaled dataset, demonstrating exceptional balance across precision and recall. |
| **Decision Tree** | Strong baseline performance (0.9824 accuracy, 0.9915 precision), though decision boundaries are axis-aligned and slightly prone to variance compared to linear methods. |
| **KNN** | High recall (0.9860) and AUC (0.9948), showing effective local neighborhood classification following feature standardization. |
| **Naive Bayes** | Solid baseline performance (0.9402 accuracy, 0.9889 AUC), though feature independence assumptions slightly limit precision compared to discriminative models. |
| **Support Vector Machine** | Highest AUC score overall (0.9976) with high recall (0.9916), demonstrating optimal class separation boundary on high-dimensional data. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the overall winner for this dataset, achieving the highest overall accuracy (98.77%), F1-score (0.9902), and MCC (0.9737), closely followed by SVM for superior class separation (AUC 0.9976). |

## Project Structure in Repository:
- `app.py`: Main Streamlit web application code.
- `requirements.txt`: Python package dependencies.
- `test_data.csv`: Evaluation dataset containing test samples.
- `*.pkl`: Serialized trained model files and standard scaler (`scaler.pkl`, `logistic_regression.pkl`, `decision_tree.pkl`, `knn.pkl`, `naive_bayes.pkl`, `support_vector_machine.pkl`).
