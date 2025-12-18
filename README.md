# Phase-1 Applied ML

Goal: Build strong applied ML fundamentals for ML-enabled Data Engineering roles.

Tech:
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost


## Day-2 Learnings
- Implemented train/test split
- Built baseline model
- Compared baseline vs ML model


## Day-3: Bias vs Variance & Overfitting

### Objective
Understand the bias–variance tradeoff and identify underfitting and overfitting in machine learning models using intuitive concepts and visual examples.

---

### Key Concepts
- **Bias**: Error caused by overly simple assumptions in the model, leading to underfitting.
- **Variance**: Error caused by excessive sensitivity to training data, leading to overfitting.
- **Underfitting**: Poor performance on both training and test datasets.
- **Overfitting**: Excellent performance on training data but poor generalization to unseen data.

---

### Intuitive Explanation
The bias–variance tradeoff represents the balance between model simplicity and complexity:
- Simple models tend to have high bias and low variance.
- Complex models tend to have low bias and high variance.
- The goal is to achieve a balance that generalizes well to unseen data.

---

### Practical Detection Strategy
| Training Performance | Test Performance | Diagnosis |
|---------------------|------------------|-----------|
| Low | Low | High Bias (Underfitting) |
| High | Low | High Variance (Overfitting) |
| High | High | Well-Generalized Model |

---

### Mitigation Techniques
**To reduce high bias:**
- Add more informative features
- Increase model complexity
- Reduce regularization 

**To reduce high variance:**
- Collect more data
- Apply regularization
- Use cross-validation
- Perform feature selection

**Note: Regularization discourages the model from learning overly complex patterns that don’t generalize well.**

---

### Implementation
A synthetic dataset was used to visually demonstrate:
- Underfitting using a simple linear regression model
- Overfitting using a high-degree polynomial regression model

Visualizations clearly show how model complexity affects generalization.

---

### Key Takeaways
- Overfitting is identified by a large gap between training and test performance.
- Feature quality is often more important than model complexity.
- Bias–variance analysis is essential before model tuning.

---

### Interview Readiness
After this exercise, I can confidently:
- Explain bias and variance intuitively
- Detect underfitting and overfitting
- Describe strategies to balance model performance

---

### Artifacts
- Notebook: `02_bias_variance.ipynb`
