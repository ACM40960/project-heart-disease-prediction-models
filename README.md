[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/GhwTNp6x)
# Comparative Study of Heart Disease Prediction Models
## Overview
This project aims to focuses on studying different machine learning models to predict heart diseases. A comprehensive evaluation is conducted using a range of performance metrics, including accuracy, precision, recall, area under the receiver operating characteristic curve (AUROC), and area under the precision-recall curve (AUPRC). By analyzing these metrics, the research aims to provide an in-depth comparison of model effectiveness, robustness, and potential clinical applicability in heart disease prediction.

## Objective
The objectives of this project are:  
- To preprocess and clean the dataset, ensuring data quality and consistency for accurate analysis.
- To implement different machine learning models for predictive analysis.
- To identify the best model and try to increase the performance metrics.

## Libraries Used


## Data
The dataset used in this project is the UCI Heart dataset.  

The UCI Heart dataset is sourced from Cleveland, Hungary, Switzerland, and VA Long Beach. It has medical records of around 726 male patients and 194 female patients. The dataset contains 14 attributes.
- age: age in years
- sex: sex (1= male, 0 = female)
- cp: chest pain type
  - Value 1: typical angina
  - Value 2: atypical angina
  - Value 3: non- anginal pain
  - Value 4: asymptotic
- Trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholesterol in mg/dl
- fbs: fasting blood sugar > 120 mg/dl (1=true, 0 = false)
- restecg: resting electrocardiographic results
  - Value 0: normal
  - Value 1: having ST-T wave abnormality
  - Value 2: showing probable or definite left ventricular hypertrophy by Estes’ criteria
- thalach: maximum heart rate achieved
- exang: exercise induced angina (1 = yes, 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
  - Value 1: upsloping
  - Value 2: flat
  - Value 3: downsloping
- Ca: number of major vessels (0-3) colored by fluoroscopy
- thal: Thalassemia
  - 3: normal
  - 6: fixed defect
  - 7: reversable defect
- num: Target variable
  - Value 0: no heart disease
  - Value 1, 2, 3, 4: presence of heart disease with increasing severity
