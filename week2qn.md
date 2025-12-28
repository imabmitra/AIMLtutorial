
---

## 1️⃣ Week-2 Topics You Must Own

You should be comfortable explaining **all of these without notes**:

* Missing value handling
* Categorical encoding strategies
* Feature scaling
* Feature importance
* Data leakage prevention
* EDA for ML (not BI)

📌 Senior signal:

> “Feature engineering is about correctness, trust, and generalization.”

---

## 2️⃣ Rapid Fire Interview Quiz (Say Out Loud)

### Q1. How do you handle missing values?

**Answer:**

> I first understand why data is missing, then choose imputation strategies based on feature type and distribution. All imputation is done inside pipelines to avoid leakage.

---

### Q2. Mean vs median imputation?

**Answer:**

> Mean for symmetric distributions, median for skewed data or when outliers are present.

---

### Q3. Label encoding vs one-hot encoding?

**Answer:**

> Label encoding is suitable for tree-based models. One-hot encoding is preferred for linear models and low-cardinality features.

---

### Q4. How do you handle high-cardinality categorical features?

**Answer:**

> I avoid one-hot encoding and instead use frequency encoding, target encoding with proper validation, or grouping rare categories.

---

### Q5. Which models need feature scaling?

**Answer:**

> Linear models, distance-based models, and neural networks. Tree-based models do not.

---

### Q6. StandardScaler vs MinMaxScaler?

**Answer:**

> StandardScaler standardizes features using mean and variance; MinMaxScaler scales to a fixed range. Choice depends on data distribution and model sensitivity.

---

### Q7. Why is feature importance important?

**Answer:**

> It helps understand model behavior, validate features, detect leakage, and build trust in predictions.

---

### Q8. Which feature importance method do you prefer?

**Answer:**

> Permutation importance, because it is model-agnostic and reflects real performance impact.

---

### Q9. What is data leakage?

**Answer:**

> Using information during training that would not be available at prediction time.

---

### Q10. How do you prevent data leakage?

**Answer:**

> By validating features, using pipelines for preprocessing, and applying correct data splitting strategies.

---

### Q11. How is EDA for ML different from BI EDA?

**Answer:**

> ML EDA focuses on feature quality, distributions, and modeling impact rather than reporting metrics.

---

## 3️⃣ Deep-Dive Senior Questions (Very Important)

### Q12. Can feature engineering cause leakage?

**Answer:**

> Yes. Aggregations, encodings, or imputations done using future or target information are common leakage sources.

---

### Q13. How does data engineering relate to feature engineering?

**Answer:**

> Feature quality depends heavily on data pipelines. Incorrect aggregations, time-unaware joins, or improper ETL logic can introduce leakage.

---

### Q14. What’s more important: model choice or feature engineering?

**Answer:**

> Feature engineering. A simple model with good features often outperforms a complex model with poor features.

---

## 4️⃣ Whiteboard Question (Practice This)

**Question:**
“How would you design a feature pipeline for predicting customer churn?”

**Expected thinking:**

* Time-aware features
* No post-churn data
* Aggregations up to prediction point
* Pipeline-based preprocessing
* Feature importance validation

📌 Senior closing line:

> “I validate features from a prediction-time perspective, not just data availability.”

---

## 5️⃣ Self-Evaluation Checklist

You are **Week-2 complete** if you can:

* ✔ Explain every feature decision
* ✔ Spot leakage in examples
* ✔ Justify encoding & scaling choices
* ✔ Explain importance without buzzwords
* ✔ Speak confidently without notes

---

