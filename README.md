# SC1015_Mini_Project_ECDS5 Diabetes Prediction Using Machine Learning

## Description
This repository contains all the Jupyter Notebooks, datasets, images, video presentations, and source materials used and created as part of the Mini Project for **SC1015: Introduction to Data Science and AI**. The project focuses on predicting diabetes risk using machine learning techniques.

---
## Contributors

### Xie Xiaomei  
- Data Cleaning  
- Univariate Analysis (categorical)  
- Bivariate Analysis (categorical)  
- Resampling  
- Machine Learning (Decision Tree, Random Forest)

### Xie Xiaotian  
- Data Cleaning  
- Univariate Analysis (numerical)  
- Bivariate Analysis (numerical)  
- Multivariate Analysis  
- Machine Learning (Random Forest, Logistic Regression, Hyperparameter Tuning)

---

## Table of Contents
- [Problem Formulation](#problem-formulation)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - Univariate Analysis
  - Bivariate Analysis
  - Multivariate Analysis
- [Data Preparation](#data-preparation)
  - Train-Test Split
  - SMOTE
  - RandomUndersampler
- [Machine Learning](#machine-learning)
  - Default Hyperparameters
    - Multivariate Classification Tree
    - Random Forest
    - Logistic Regression
  - Hyperparameter Tuning (GridSearchCV)
    - Random Forest
- [Ranking Predictors](#ranking-predictors)
- [Insights and Conclusions](#insights-and-conclusions)
- [What we learnt](#What-We-Learnt)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Problem Formulation
**Dataset:** [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)  
**Problem Statement:** Can we use machine learning to predict diabetes risk using routine clinical and lifestyle data?

**Rationale:**  
Diabetes is a growing global health concern, with millions affected annually. It can lead to severe complications such as cardiovascular disease, nerve damage, kidney failure, and vision loss if not detected and managed early. Early prediction and intervention are essential to reducing both the personal and economic burdens of the disease.
Our analysis and predictive modeling offers a promising approach to identifying individuals at risk before symptoms become critical. By analysing features like age, BMI, glucose levels, and blood pressure, machine learning models can support healthcare professionals in making faster and more accurate risk assessments. This not only helps in early detection but also plays a key role in developing personalized treatment plans tailored to individual patient profiles. Moreover, the insights gained can contribute to public health strategies, prevention programs, and advancements in clinical practices.


---

## Data Cleaning
Steps taken:
- **Dropped duplicate rows** to ensure model performance and accuracy.
- **Selected relevant columns**, excluding features like "smoking history".
- **Removed rows** where gender was labeled as "Other" to focus on binary gender classification.
- **One-hot encoded categorical variables** to convert them into numerical format without imposing ordinality.

---

## Exploratory Data Analysis
We explored the relationship between predictors and the target variable (diabetes):

### Univariate Analysis:
- Categorical and numerical distributions

### Bivariate Analysis:
- Comparisons between individual predictors and diabetes outcome

### Multivariate Analysis:
- Interactions among multiple predictors

---

## Data Preparation
The dataset was imbalanced:  
**8.82% Positive Cases** vs. **91.18% Negative Cases**

To address this, we used:
- **SMOTE (Synthetic Minority Over-sampling Technique)** to increase minority class
- **RandomUndersampler** to reduce majority class

This balanced the dataset, improving model generalization.

---

## 🤖 Machine Learning
Models applied:
- **Multivariate Decision Tree**
- **Random Forest**
- **Logistic Regression**

### Hyperparameter Tuning:
Used `GridSearchCV` for:
- Fine-tuning the **Random Forest** model
- Improving performance on validation data

### Evaluation Metrics:

---

## Ranking Predictors
Ranking predictors helps:
- Improve model interpretability
- Guide further analysis or policy decisions

We visualized feature importances using a **bar plot**.

---

## Insights and Conclusions
- **HbA1c and glucose levels** were the most important predictors
- **Age** was moderately important
- **Hypertension and heart disease** had low predictive power

### Implications:
- Prioritize **HbA1c/glucose tests**
- Adjust screening thresholds based on **age**
- Place less emphasis on comorbidities for initial screening

### Surprising Findings
Despite their clinical relevance, **hypertension and heart disease showed low importance**—possibly due to underrepresentation in the dataset. Future work should further investigate this.

### Recommendations
- Screen all patients **over 40** with HbA1c tests
- Focus confirmatory tests on:
  - HbA1c > 5.7 & glucose > 100
  - Age > 50 with BMI > 25

### Limitations 
- Data Limitations: Underrepresentation of gestational diabetes cases
- Temporal Gaps: Lacks longitudinal progression data
---

## Future Improvements
- Incorporate **time-series biomarker measurements**
- Test with **more diverse population groups**
- Develop **dynamic risk scores** that update with new inputs

---

## 📚 References
- [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)  
- [RandomForestClassifier - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
- [Handling Imbalanced Data - GeeksForGeeks](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/)  
- [RandomUndersampler - imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html)  
- [GridSearchCV - scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html)  
- [WHO - Diabetes](https://www.who.int/health-topics/diabetes#tab=tab_1)
