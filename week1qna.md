## Week-1 Interview Quiz & Answers (Applied ML Foundations)

This section contains interview-style questions and concise answers based on Week-1 learnings.  
The focus is on **ML fundamentals, evaluation reliability, and production readiness**.

---

### Q1. Walk me through your machine learning workflow.
**Answer:**
I start by defining a baseline model to validate whether ML adds value.  
I split the data into training and test sets to simulate unseen data.  
I analyze bias and variance by comparing training and validation performance, validate results using cross-validation, and finally build a leak-free pipeline using `Pipeline` and `ColumnTransformer`.  
For production readiness, I refactor the workflow into modular Python code.

---

### Q2. Why do you create a baseline model before training ML models?
**Answer:**
A baseline model provides a reference point. If a machine learning model does not outperform the baseline, the issue is usually with feature quality or problem formulation rather than the algorithm itself.

---

### Q3. What is the purpose of train–test split?
**Answer:**
The train–test split helps evaluate how well a model generalizes to unseen data. The test set represents future data that the model has not encountered during training.

---

### Q4. What happens if you train and test on the same dataset?
**Answer:**
The model will memorize the data and produce overly optimistic performance metrics, which leads to poor generalization in production. This is a classic case of overfitting.

---

### Q5. What is bias in machine learning?
**Answer:**
Bias is the error introduced by overly simplistic assumptions in the model, causing it to underfit the data.

---

### Q6. What is variance in machine learning?
**Answer:**
Variance is the error caused by a model being overly sensitive to training data, resulting in overfitting and poor performance on unseen data.

---

### Q7. How do you detect overfitting?
**Answer:**
Overfitting is detected when the model performs very well on training data but significantly worse on validation or test data.

---

### Q8. What is the bias–variance tradeoff?
**Answer:**
Increasing model complexity typically reduces bias but increases variance. The goal is to find a balance where the model generalizes well to unseen data.

---

### Q9. Why is cross-validation better than a single train–test split?
**Answer:**
A single split can give unstable results depending on how the data is divided. Cross-validation evaluates the model across multiple splits, providing a more reliable estimate of generalization performance.

---

### Q10. What value of K do you typically use in K-Fold Cross-Validation?
**Answer:**
I usually start with K = 5 as it provides a good balance between evaluation reliability and computational cost.

---

### Q11. Does cross-validation prevent overfitting?
**Answer:**
No, cross-validation does not prevent overfitting, but it helps detect it more reliably by validating the model on multiple data splits.

---

### Q12. What is data leakage?
**Answer:**
Data leakage occurs when information from outside the training data is used during model training, leading to unrealistically high evaluation performance.

---

### Q13. How do pipelines prevent data leakage?
**Answer:**
Pipelines ensure that preprocessing steps are fitted only on the training data and applied consistently during validation or testing, preventing information leakage.

---

### Q14. Why do you use ColumnTransformer?
**Answer:**
Real-world datasets contain mixed feature types. `ColumnTransformer` allows applying appropriate preprocessing steps to numerical and categorical features safely within a single pipeline.

---

### Q15. Why is modular ML code important?
**Answer:**
Modular ML code improves maintainability, testability, and reproducibility, and integrates more easily with CI/CD and MLOps workflows.

---

### Q16. How does your ML work align with Data Engineering?
**Answer:**
I focus on feature pipelines, data correctness, and reliable model evaluation, ensuring that ML systems are production-ready and integrate smoothly with existing data platforms.

---

### Week-1 Summary
Week-1 focused on building strong applied ML fundamentals with emphasis on:
- Reliable evaluation
- Data leakage prevention
- Production-ready code structure
- Engineering-first ML practices
