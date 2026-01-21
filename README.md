# Machine Learning – Supervised Learning Project

## 📌 Project Overview
This project demonstrates an **end-to-end Supervised Machine Learning workflow** using multiple
classification algorithms.  
The objective is to train, evaluate, and compare different supervised learning models on labeled data
and select the best-performing model.
---    

## 🧠 Supervised Learning Algorithms Used
The following classification algorithms have been implemented and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier (XGBClassifier)
These models were trained on the same dataset to compare performance.
---    

## ⚙️ Libraries & Tools
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
---

## 🔁 Model Training Approach
All models were trained using a common training loop:
- Data split into training and testing sets
- Models trained using `fit()`
- Predictions generated for both train and test data
- Performance compared across models
---

## python
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier()
}
---

## Model Evaluation Metrics
Each model was evaluated using multiple performance metrics:
1. Accuracy Score
2. Precision Score
3. Recall Score
4. F1 Score
5. ROC-AUC Score
6. Confusion Matrix
7. Classification Report
8. ROC Curve
---

## 🔍 Hyperparameter Tuning
To improve model performance, RandomizedSearchCV was used for hyperparameter tuning.
- Efficient search over hyperparameter space
- Cross-validation applied
- Best parameters selected automatically
- Improved generalization on test data
---

## Workflow Summary
1. Data Loading
2. Data Preprocessing
3. Train-Test Split
4. Model Training (Multiple Algorithms)
5. Hyperparameter Tuning (RandomizedSearchCV)
6. Model Evaluation
7. Performance Comparison
8. Final Prediction
---

## How to Run the Project
1. Clone this repository
2. Open the Jupyter Notebook file (.ipynb)
3. Install required libraries
4. Run all cells sequentially
---

## Key Learnings
1. Understanding of Supervised Learning concepts
2. Practical implementation of multiple ML algorithms
3. Model comparison and evaluation
4. Hyperparameter tuning using RandomizedSearchCV
5. Performance metrics interpretation
---

👤 Author
Md Tajuddin 
