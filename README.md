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


## Day-4: Cross-Validation (K-Fold)

### Objective
Learn reliable model evaluation using K-Fold Cross-Validation and understand why a single train–test split is often insufficient for estimating real-world performance.

---

### Problem with Single Train–Test Split
A single train–test split can produce unstable evaluation results because:
- Model performance depends heavily on how the data is split
- The test set may not be representative of future data
- Results can vary significantly with different random states

---

### Cross-Validation Overview
K-Fold Cross-Validation works by:
- Splitting the dataset into **K equal folds**
- Training the model on **K−1 folds**
- Validating on the remaining fold
- Repeating the process K times
- Reporting the **average performance across folds**

This ensures that every data point is used for both training and validation.

---

### Why K-Fold Cross-Validation is Reliable
- Reduces evaluation variance
- Provides a more stable estimate of model generalization
- Helps detect overfitting more consistently than a single split

---

### Choosing the Value of K
| Dataset Size | Recommended K |
|-------------|---------------|
| Small | 5 or 10 |
| Medium | 5 |
| Large | 3–5 |

A value of **K = 5** was used as a balance between reliability and computational cost.

---

### Implementation
- Dataset: Titanic survival dataset
- Model: Logistic Regression
- Evaluation Metrics: Accuracy
- Comparison:
  - Single train–test split accuracy
  - Mean accuracy across K-Fold Cross-Validation

Results show that cross-validation provides a more consistent estimate of model performance.

---

### When Not to Use K-Fold Cross-Validation
- Time-series data (data leakage risk)
- Extremely large datasets where computation cost is high

Alternative strategies include time-aware splits or a single validation set.

---

### Key Takeaways
- Cross-validation improves confidence in model evaluation
- It does not prevent overfitting but helps detect it
- Reliable evaluation is more important than high single-split accuracy

---

### Interview Readiness
After this exercise, I can:
- Explain K-Fold Cross-Validation intuitively
- Justify why cross-validation is preferred over a single split
- Choose an appropriate value of K based on dataset size

---

### Artifacts
- Notebook: `03_cross_validation.ipynb`
