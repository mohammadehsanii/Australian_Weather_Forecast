---

# Australian Weather Forecast

For more detailed information, please refer to the .ipynb file available [here](https://www.kaggle.com/code/mohammadehsani/australian-weather-forecast)

## Table of Contents

1. **Case Study Challenge**
    - Introduction
    - Data Understanding
    - Data Structure

2. **Exploratory Data Analysis**
    - Heatmap for Bivariate Analysis
    - Data Visualization
    - Data Pre-Processing and Feature Engineering
        - Engineering Outliers
    - Handling Missing Values

3. **Confirmatory Data Analysis**
    - Encoding Numerical and Categorical Variables
    - Scaling Data
    - Model Training
    - Predict Results
    - Accuracy
    - Overfitting and Underfitting
    - K-Fold Cross Validation

4. **Outcome and Concluding Remarks**
    - Confusion Matrix
    - Classification Report
    - Probabilities
    - ROC - AUC
    - Conclusion

5. **Resources & Libraries**

## Introduction

In this project, the aim is to determine if there will be rainfall in Australia tomorrow. Python and Scikit-Learn are utilized to apply Logistic Regression. The approach involves constructing a classifier to anticipate rainfall occurrences for the following day using the Rain in Australia dataset.

## Data Set Understanding

The dataset comprises 23 columns and 145,460 rows with both categorical and numerical variables. The dataset contains some missing values, spanning approximately 10 years of daily weather observations from numerous locations across Australia.

## Data Set Structure

There are 16 numerical variables and 7 categorical variables. The pie chart illustrates the distribution, with categorical data representing 30.4% and numerical data 69.6% of the dataset.

## Correlation for Bivariate Analysis

The correlation heatmap reveals significant correlations between variables, particularly noting high positive correlations among temperature variables and wind speed.

## Data Visualization

Several visualizations provide insights into rainfall distribution, temperature patterns, wind directions, and humidity levels, aiding in understanding regional climate dynamics.

## Data Pre-Processing and Feature Engineering

The date column is divided into separate components to analyze temporal patterns more effectively. Outliers in certain variables are addressed using a top-coding approach.

## Handling Missing Values

Missing values are imputed by filling categorical variables with mode and numerical variables with median values.

## Encoding Numerical and Categorical Variables

One-hot encoding is applied to convert categorical variables into numerical format.

## Scaling Data

Numerical variables are scaled using the MinMaxScaler library to ensure optimal performance of machine learning algorithms.

## Model Training

The dataset is divided into training and testing sets for model evaluation. Logistic Regression is chosen for its suitability in binary classification tasks.

## Predict Results

The model demonstrates excellent performance, with an accuracy score of approximately 83%.

## Overfitting and Underfitting

The model exhibits no signs of overfitting or underfitting, with comparable training and testing set accuracies.

## K-Fold Cross Validation

Cross-validation with 5 folds ensures reliable model evaluation and robustness.

## Confusion Matrix

The confusion matrix provides insights into the model's performance, distinguishing between true positives, true negatives, false positives, and false negatives.

## Classification Report

Precision, recall, specificity, and F1-score metrics are evaluated to assess model performance.

## Probabilities

Probabilities are analyzed to predict the likelihood of rainfall tomorrow.

## ROC - AUC

The ROC Curve and AUC score visualize the classifier's performance, with a score of 0.85 indicating effective prediction capabilities.

## Conclusion

The Logistic Regression model performs well in forecasting rainfall, with no discernible signs of overfitting. The ROC AUC score validates the classifier's effectiveness in predicting rainfall occurrence.

## Resources & Libraries

The project utilizes various libraries for data manipulation, visualization, and modeling, including Pandas, NumPy, Matplotlib, Seaborn, Plotly Express, Category Encoders, and Scikit-learn.

---
