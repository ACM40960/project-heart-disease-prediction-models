[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/GhwTNp6x)
# Comparative Study of Heart Disease Prediction Models

<div align="center"><img width="650" height="400" alt="image" src="https://github.com/user-attachments/assets/0d4bb3a6-65ec-4018-ac2d-b66cfd6dbfe2" /></div>

## Overview
This project focuses on studying different machine learning models to predict heart diseases. A comprehensive evaluation is conducted using a range of performance metrics, including accuracy, precision, recall, area under the receiver operating characteristic curve (AUROC), and area under the precision-recall curve (AUPRC). By analyzing these metrics, the research aims to provide an in-depth comparison of model effectiveness, robustness, potential clinical applicability in heart disease prediction and find an optimal ML model for predicting heart disease.

## Objective
The objectives of this project are:  
- To preprocess and clean the dataset, ensuring data quality and consistency for accurate analysis.
- To implement different machine learning models for predictive analysis.
- To identify the best model with the best performance metrics.




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

## Exploratory Data Analysis
- The distribution of target variable is given below:
  <div align="center"><img width="576" height="432" alt="01_Target_Distribution" src="https://github.com/user-attachments/assets/da9874ee-5643-491e-85f6-6f49af4d8301" /></div>
- Correlation matrix between the variables
  <div align="center"><img width="1584" height="1296" alt="02_Correlation_Matrix" src="https://github.com/user-attachments/assets/d37e366b-ba0c-4ea7-8be6-78816326876d" /></div>
  - A strong correlation can be seen between the number of major vessels and age
- Age Distribution by Heart Disease Status
 <diiv align="center"> <img width="864" height="504" alt="03_Age_Distribution_by_Target" src="https://github.com/user-attachments/assets/d40450ca-08cf-427d-802a-f8cfca56e510" /></div>
  - It can be seen that there are more patients between the ages 50 and 60 in this data.

## Pre Processing Steps 
<div align="center"><img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/bbba66dc-eff9-4d18-af04-ecaad489b178" /></div>

1. **Rename Columns**  
   - The `num` column was renamed to **`target`** for clarity.  

2. **Binarize Target Variable**  
   - Converted the original target values (`0–4`) into a binary classification:  
     - `0` → No disease  
     - `1` → Disease present  

3. **Drop Non-Predictive Columns**  
   - Removed `id` and `dataset` columns as they carry no predictive power.  

4. **Encode Categorical Columns for Imputation**  
   - Converted categorical text columns into numeric codes temporarily.  

5. **Handle Missing Values with Iterative Imputer**
   
   <div align="center"><img width="350" height="300" alt="image" src="https://github.com/user-attachments/assets/4b22448b-ba95-48cf-a4e2-0db781af2c81" /></div>

   - Used **IterativeImputer** (10 iterations, random state 42).  
   - Each missing value is modeled as a function of the other features.  

7. **Restore Categorical Features**  
   - After imputation, categorical codes were mapped back to their original categories.  

8. **One-Hot Encoding**  
   - Converted categorical variables into dummy variables.  
   - Used `drop_first=True` to avoid multicollinearity.  

---  

## Machine Learning Models Used

- Logistic Regression: This method models the log odds of an event as a linear combination of the predictor variables. It is used for binary classification problems.

- Random Forest Classifier: This method combines individual decision trees that are trained on different random subsets of the training data. The predictions from each of these trees are independent of each other. The final decision is the majority prediction.

- XgBoost Classifier: This stands for Extreme Gradient Boosting. It makes a simple prediction on the training data, calculates the residuals and builds a decision tree. 

- Support Vector Machine: This method works by finding a hyperplane that optimizes the separation between classes based on the variables. 

- Naive Base Classifier: This method is used for text based classification. ‘Naive’ is used as it assumes that all features are independent, and Bayes refers to the use of Bayes’ Theorem that uses probability to predict the class of the target variable.

- Autoencoder + DenseNet:
  - Autoencoder: It is a neural network used for unsupervised learning to compress data and then reconstruct it. It consists of an encoder (compresses the number of features), latent space(space capturing the essential features) and a decoder (reconstructing the input code back to the orginal).  It is used for dimensionality reduction, noise reduction, feature extraction, and image reconstruction.
  - DenseNet: It is a convolutional neural network where each layer is connected to every preceding layer directly. 

## Model Training and Individual Evaluation
The main steps are highlighted below:
1. A dictionary for different classifiers has been made
  ```python
  models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(probability=True, random_state=42), # `probability=True` is needed for ROC curves.
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # XGBoost with common settings to avoid warnings.
   }
  ```
2. Each model gets a search space of hyperparameters for GridSearchCV. Naive Bayes does not have any tunable parameters here.
   ```python
   param_grids = {
    "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['liblinear']},
    "Support Vector Machine": {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']},
    "Naive Bayes": {},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    "XGBoost": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.05]}
   }
    ```
3. For each model, a grid search with a 5-fold cross validation is run on the training data, the best hyperparameters are identified and the best model is stored.
   ```python
   for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids.get(name, {}), cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
   ```
4. Autoencoder + DenseNet block
   This was the state of the art model. \\
   Autoencoder: It compresses into 8 features (bottleneck) and is trained to reconstruct the input. After proper training, the decoder is discarded and the encoder is used to get the new compressed features.
   ```python
   def create_autoencoder():
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation='relu')(input_layer)
    encoder = Dense(8, activation='relu')(encoder)   # bottleneck
    decoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    return autoencoder, encoder_model
   ```
   DenseNet: A small feedforward neural network is constructed. The input is the size of the encoded features. The output is a single probability. 
   ```python
   def create_dense_net(dense_input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(dense_input_dim,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
   ```

   The autoencoder is trained to compress and reconstruct the training data. The orginal data is transformed into encoded representations.

## Results
1. **Confusion Matrix**
   Confusion matrix gives a detailed breakdown of the model perfomance. The confusion matrix obtained for each of the models is given below.
<p align="center">
  <figure style="display:inline-block; margin:10px;text-align:center;">
    <img src="https://github.com/user-attachments/assets/0f38d7ee-35cb-40c1-bc2a-d89ec5d10d1e" width="30%">
    <figcaption>Logistic Regression</figcaption>
  </figure>

  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/da0d1d5a-8220-4057-93a0-deba12704667" width="30%">
    <figcaption>Autoencoder DenseNet</figcaption>
  </figure>
</p>

<p align="center">
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/36f1f315-bcf7-4d54-9534-525ff9043dbc" width="30%">
    <figcaption>Naive Bayes</figcaption>
  </figure>

  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/20b4c4fc-7613-4f55-8f5d-c39b9baa71cd" width="30%">
    <figcaption>Random Forest</figcaption>
  </figure>
</p>

<p align="center">
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/56690d63-e84b-4e77-aa06-ce6c46190abe" width="30%">
    <figcaption>Support Vector Machine</figcaption>
  </figure>

  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/347ec86b-e487-491d-8114-671d8ded5377" width="30%">
    <figcaption>XGBoost</figcaption>
  </figure>
</p>
The main focus here is to have a high number of True Positives and True Negatives. In addition, since this is a diagnostic tool, we also need to focus on the false negatives. //
It can be seen from the above confusion matrices that Random Forest classifier has the least number of false negatives. //
Perfomance metrics like accuracy, precision, recall and F1 score can be found from the confusion matrices. These are as summarized below://


| Model                   | Accuracy | F1-Score | Sensitivity | Specificity |
|--------------------------|----------|----------|-------------|-------------|
| Random Forest            | 0.897    | 0.909    | 0.872       | 0.933       |
| Support Vector Machine   | 0.886    | 0.899    | 0.853       | 0.933       |
| XGBoost                  | 0.886    | 0.899    | 0.853       | 0.933       |
| Logistic Regression      | 0.870    | 0.886    | 0.853       | 0.893       |
| Autoencoder + DenseNet   | 0.848    | 0.868    | 0.844       | 0.853       |
| Naive Bayes              | 0.832    | 0.853    | 0.826       | 0.840       |

2. **Sensitivity vs Specificity**
   High sensitivity is crucial, as it implies fewer false negatives. High specificity is aslo important as it implies fewer false positives
   <div align="center"><img width="700" height="432" alt="09_Sensitivity_Specificity_Comparison (2)" src="https://github.com/user-attachments/assets/c9116d6c-b4c2-4908-99c5-c7144e007578" /></div>
   

- Random Forest, SVM, and XGBoost achieved the highest specificity (~0.93) with good sensitivity (~0.85–0.87).  
- Logistic Regression is more balanced across both metrics.  
- Naive Bayes and Autoencoder + DenseNet underperformed.
  Random Forest achieves the best balance, demonstrating its effectiveness in correctly identifying both patients with
and without heart disease.

3. **ROC Curve**
The ROC curve shows the trade-off between True Positive Rate (Sensitivity) and False Positive Rate across thresholds.  It demonstrates the ability of the classifier to distinguish between the classes.
The AUC (Area Under Curve) summarizes performance — closer to 1.0 means a better classifier.
<div align="center"><img width="576" height="432" alt="08_Combined_ROC_Curves (1)" src="https://github.com/user-attachments/assets/fdfea033-86e5-42c8-986b-c82ff303c9ca" /></div>


Random Forest has the highest Area Under the Curve (AUC) of 0.961,
indicating superior discriminative performance over other models.

4. **Accuracy and F1 score**
Accuracy give sthe overall prediction of correct predictions and F1 score is the harmonic mean of precsion and recall and is mostly used for imbalanced data. 
 <div align="center"><img width="800" height="432" alt="image" src="https://github.com/user-attachments/assets/c557eaa0-83ca-45cf-84b8-7d66fd91cdec" /></div>

Random Forest again leads in both metrics, showing it balances precision and recall effectively, which is critical for reliable clinical predictions.

6. **PR Curves**
The Precision-Recall curve shows the trade-off between Precision (positive prediction accuracy) and Recall (sensitivity).  
It is most useful when dealing with imbalanced datasets.
<div align="center"><img width="576" height="432" alt="image" src="https://github.com/user-attachments/assets/a13ea6ac-d088-41ac-b506-282b2a678217" /></div>

Random Forest model consistently maintains higher precision and recall, reflecting its robustness in predicting heart disease cases.

## Feature Selection
Feature importance has been extracted for Random Forest Classifier. 
<div align="center"><img width="750" height="432" alt="11_Feature_Importance_Random_Forest" src="https://github.com/user-attachments/assets/374fbc2a-e6c8-43bf-9fad-2773a3b77334" /></div>

## References









    
