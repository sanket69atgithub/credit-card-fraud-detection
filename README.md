# Credit Card Fraud Detection

This project builds a machine learning model to detect fraudulent credit card transactions. Credit card fraud datasets are typically highly imbalanced, and this project explores how to handle that data and train classifiers to identify fraud effectively.

## Overview

The dataset (`creditcard.csv`) contains transactions made by European cardholders in September 2013. Out of 284,807 transactions, only 492 are fraudulent, which means the positive class (fraud) accounts for roughly 0.172% of all transactions. 

To tackle this, the project does some basic exploratory data analysis (EDA), scales the features, and then compares a couple of different classification algorithms to see what works best.

## What's in the repo?

- `credit_card_fraud.ipynb`: The main Jupyter Notebook containing all the code for data processing, EDA, model training, and evaluation.
- `creditcard.csv`: The dataset used for training (Note: this file is quite large, ~150MB).

## Libraries Used

You'll need the following Python libraries installed to run the notebook:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install them using pip if you don't have them already:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Approach

1. **Data Loading & EDA**: Checked for missing values and looked at the distribution of the classes to see just how imbalanced the dataset is.
2. **Preprocessing**: The `Amount` feature is scaled using `StandardScaler` to bring it into a similar range as the PCA-transformed features (V1 to V28). The `Time` column is dropped since it's not strictly necessary for this baseline approach.
3. **Modeling**: The data is split into 70% training and 30% testing sets. Two models were trained and evaluated:
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
4. **Evaluation**: Both models achieved over 99.9% accuracy. A confusion matrix is also plotted to give a better breakdown of true positives vs false positives, which is a lot more useful than just looking at accuracy for an imbalanced dataset like this.

## How to run

1. Clone this repository.
2. Make sure you have the required libraries installed.
3. Open `credit_card_fraud.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells from top to bottom. Make sure `creditcard.csv` is in the same directory as the notebook.
