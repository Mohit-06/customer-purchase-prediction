# Customer Purchase Prediction – End-to-End Machine Learning Project

## 1. Problem Statement

The goal of this project is to predict whether a customer will purchase a product based on demographic and financial information. This type of prediction helps businesses target the right customers, optimize marketing campaigns, and improve conversion rates.

**Business Objective:**
Given customer details such as age, salary, and gender, predict the probability of purchase and make a final Yes/No decision using a business-defined threshold.

---

## 2. Dataset Description

The dataset contains approximately **400 customer records** with the following features:

* **Age** – Age of the customer
* **EstimatedSalary** – Estimated annual salary
* **Gender** – Male/Female (categorical)
* **Purchased** – Target variable (0 = No, 1 = Yes)

### Key Characteristics

* Small dataset
* Binary classification problem
* Imbalanced target variable:

  * Class 0 (No): ~257 records
  * Class 1 (Yes): ~157 records

---

## 3. Exploratory Data Analysis (EDA)

Key insights obtained from EDA:

* Customers aged **40–55** show a higher purchase rate
* Customers with **salary above 80,000** have a higher probability of purchasing
* Target variable is **imbalanced**, so accuracy alone is not a reliable metric

EDA helped in understanding feature importance and choosing appropriate evaluation metrics.

---

## 4. Feature Engineering & Preprocessing

The following preprocessing steps were applied:

* **Gender Encoding:** Label Encoding (Male = 1, Female = 0)
* **Feature Scaling:** StandardScaler used for Age and Salary
* **Train-Test Split:**

  * Performed before scaling to avoid data leakage
  * Stratified split used to preserve class distribution

This ensured fair model training and reliable evaluation.

---

## 5. Model Training & Comparison

Multiple machine learning models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* XGBoost (after hyperparameter tuning)

### Model Selection Criteria

* ROC-AUC score
* Precision and Recall (due to class imbalance)
* Stability on small datasets

### Final Model Choice

**Random Forest Classifier** was selected because:

* Performs well on small datasets
* Handles non-linear relationships
* Stable and robust
* Supports probability-based predictions

---

## 6. Model Evaluation

Evaluation metrics used:

* Confusion Matrix
* Precision, Recall, F1-score
* ROC-AUC Score

**Why not Accuracy?**
Due to class imbalance, accuracy can be misleading. ROC-AUC and Recall provide better insight into real performance.

---

## 7. Business Threshold Tuning

Instead of using the default 0.5 threshold:

* A **custom threshold of 0.6** was selected
* This reduces false positives and aligns with business needs

Final prediction logic is based on **predicted probability**, not hard labels.

---

## 8. Deployment (FastAPI)

The trained model and scaler were deployed using **FastAPI**.

### API Features

* RESTful `/predict` endpoint
* Input validation using Pydantic
* Probability-based prediction output
* Business threshold applied during inference

### Example API Request

```json
{
  "age": 45,
  "salary": 90000,
  "gender": "Male"
}
```

### Example API Response

```json
{
  "probability": 0.92,
  "prediction": "Yes"
}
```

---

## 9. Key Challenges & Learnings

* Handling class imbalance correctly
* Avoiding data leakage during scaling
* Ensuring consistency between training and deployment pipelines
* Debugging real-world deployment issues (feature mismatch, stale models)

This project closely reflects **real industry ML workflows**.

---

## 10. Tools & Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* FastAPI
* Joblib
* Uvicorn

---

## 11. Conclusion

This project demonstrates an end-to-end Machine Learning solution, from data understanding to production deployment. It highlights not only modeling skills but also real-world engineering practices required for deploying ML systems in production.

---

## 12. Future Improvements

* Model calibration
* ROC curve visualization
* Docker-based deployment
* CI/CD integration
* Monitoring model drift

#Author 
Mohit Kumar
