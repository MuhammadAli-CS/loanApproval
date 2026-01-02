# Loan Approval Prediction Using Machine Learning

## Overview

This project explores the use of supervised machine learning models to predict whether a loan application will be approved or rejected. Loan approval is a critical decision-making process for financial institutions, and accurate predictive models can help automate evaluations while reducing risk.

Using historical loan application data, we implement and compare multiple classification models to analyze how applicant financial and demographic features influence loan approval outcomes.

## Dataset

We use the Loan Approval Classification Data dataset from Kaggle, which contains information about applicants’ demographics, financial history, credit profiles, and loan details.

### Dataset link:
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data

## Models Implemented

The following machine learning models were trained and evaluated:

Support Vector Machine (SVM) with linear kernel

Random Forest Classifier

k-Nearest Neighbors (kNN)

Each model was evaluated using train/test splits and accuracy metrics to compare performance.

## Methodology

### 1- Data Preprocessing

        Handled categorical variables using one-hot encoding

Prevented data leakage by separating features and target early

Sampled 1% of the dataset to reduce computational cost

Feature Selection

Generated correlation heatmaps and ranked feature importance

Selected top features with the highest correlation to loan status:

Previous loan defaults

Loan percent of income

Loan interest rate

Home ownership (rent)

Applicant income

Model Training & Scaling

Trained models on both raw and normalized data

Applied MinMax scaling to improve SVM performance

Used consistent random seeds for reproducibility

Evaluation & Visualization

Compared training and testing accuracy across models

Visualized decision trees in the Random Forest

Used pair plots to analyze feature relationships

Results

SVM showed significant performance improvement after feature scaling.

Random Forest achieved high accuracy but showed signs of overfitting.

kNN performed competitively with strong training and testing accuracy.

Feature selection played a major role in improving model efficiency and interpretability.

Conclusion

This project demonstrates how preprocessing choices, feature selection, and model selection affect classification performance in financial decision-making tasks. While a reduced dataset was used for efficiency, expanding training to a larger portion of the data could further improve generalization and accuracy.

Technologies Used

Python

NumPy, Pandas

scikit-learn

Matplotlib, Seaborn

Jupyter / Google Colab

Contributors
