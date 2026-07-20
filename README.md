# retail-credit-risk-assessment-imbalance-optimization
A Python machine learning pipeline optimizing credit risk classification under empirical class imbalance (UTS IT Project).

# Optimizing Retail Credit Risk Assessment Under Empirical Class Imbalance

## 📌 Project Overview
This project simulates a real-world data science consultancy task focused on financial credit risk assessment. The objective is to build an explainable and highly robust machine learning pipeline to predict whether a customer will repay their loan (`loan_paid_back` {0, 1}). 

The core challenge of this project lies in mitigating heavy **empirical class imbalance** to ensure the model makes principled decisions rather than relying on naive accuracy.

## 🛠️ Key Technical Implementations
* **Advanced Preprocessing & Feature Engineering:** Conducted noise reduction, custom attribute grouping, and binary transformations based on domain knowledge.
* **Multi-Classifier Exploration:** Built, compared, and optimized a wide range of algorithms including k-NN, Decision Trees, Support Vector Machines (SVM), Random Forests, and Multilayer Perceptrons (MLP).
* **Robust Validation:** Applied **K-Fold Cross-Validation** and **Stratified Splitting** to optimize hyperparameters (e.g., selecting the optimal 'k' in k-NN) and prevent overfitting.
* **Imbalance Mitigation & Evaluation:** Moved beyond traditional accuracy by introducing rigorous evaluation-level strategies, utilizing **Confusion Matrices, Precision, Recall, F1-Score, and ROC/AUC** curves to benchmark performance.

## 📂 Repository Structure
* `*.ipynb`: The complete Python scikit-learn pipeline covering data exploration, training, and evaluation.
* `README.md`: Project summary and key highlights.
*(Note: In compliance with academic integrity, the raw training dataset `trainData.csv` is restricted and not included in this public repository).*
