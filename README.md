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


## Day-5: ML Pipelines & ColumnTransformer (Data Leakage Prevention)

### Objective
Build a production-ready machine learning pipeline using `Pipeline` and `ColumnTransformer` to ensure correct preprocessing, prevent data leakage, and enable reliable model evaluation.

---

### What Is Data Leakage?
Data leakage occurs when information from outside the training dataset is inadvertently used during model training. This leads to unrealistically high performance during evaluation and poor generalization in production.

---

### Why Pipelines Are Important
Machine learning pipelines:
- Enforce the correct order of preprocessing and training
- Ensure transformations are learned only from training data
- Prevent accidental data leakage
- Enable reproducible and production-ready ML workflows

---

### Why ColumnTransformer Is Needed
Real-world datasets typically contain mixed feature types:
- Numerical features requiring scaling
- Categorical features requiring encoding

`ColumnTransformer` allows different preprocessing strategies to be applied safely to different feature subsets within a single pipeline.

---

### Implementation Details
- Dataset: Titanic survival dataset
- Numerical features: `age`, `fare`
- Categorical features: `sex`
- Preprocessing:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- Model: Logistic Regression
- Evaluation: Accuracy with train–test split and K-Fold Cross-Validation

All preprocessing and model training steps are encapsulated inside a single pipeline to avoid leakage.

---

### Cross-Validation with Pipelines
The pipeline was evaluated using K-Fold Cross-Validation, ensuring:
- Preprocessing steps are refit on each training fold
- Validation data remains unseen during training
- Performance estimates are stable and reliable

---

### Key Takeaways
- Data leakage is a major cause of ML failure in production
- Pipelines enforce safe and consistent preprocessing
- ColumnTransformer is essential for handling mixed data types
- Pipelines integrate seamlessly with cross-validation

---

### Interview Readiness
After this exercise, I can confidently:
- Explain data leakage and its impact
- Justify the use of pipelines in ML workflows
- Build leak-free preprocessing and training pipelines
- Use cross-validation safely with pipelines

---

### Artifacts
- Notebook: `04_pipeline_column_transformer.ipynb`


## Day-6: Code Refactor & Modular Machine Learning

### Objective
Refactor notebook-based machine learning code into a modular, production-style Python project. The goal is to structure ML code in a way that is maintainable, testable, and ready for integration with CI/CD and MLOps pipelines.

---

### Why This Matters
While notebooks are useful for experimentation, production machine learning systems require:
- Clear separation of responsibilities
- Reusable and testable components
- Reproducibility and maintainability

This exercise focuses on converting exploratory ML code into clean, engineering-grade Python modules.

---

### Project Structure
The project was restructured as follows:


```text
phase1-applied-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_bias_variance.ipynb
│   ├── 03_cross_validation.ipynb
│   └── 04_pipeline_column_transformer.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

---

### Module Responsibilities

#### `data_loader.py`
- Handles data loading and basic cleanup
- Isolated from feature and model logic

#### `preprocessing.py`
- Defines feature preprocessing logic
- Uses `ColumnTransformer` for safe handling of mixed data types

#### `model.py`
- Contains model and pipeline construction
- Keeps modeling logic independent from data loading

#### `train.py`
- Orchestrates the end-to-end training workflow
- Handles train–test split, training, and evaluation
- Acts as the entry point for training execution

---

### Key Design Principles
- **Separation of concerns**: Each module has a single responsibility
- **Reproducibility**: Training can be executed consistently from the command line
- **Leakage prevention**: All preprocessing is encapsulated inside pipelines
- **Production readiness**: Code structure aligns with MLOps and CI/CD practices

---

### How to Run Training
From the project root:
```bash
python src/train.py
```

## Week-1 Summary (Applied ML Foundations)

Covered:
- Train/test split and baseline modeling
- Bias vs variance and overfitting detection
- K-Fold cross-validation for reliable evaluation
- Leak-free preprocessing using Pipeline and ColumnTransformer
- Modular, production-style ML code structure

Focus was on correctness, evaluation reliability, and production readiness rather than algorithm complexity.



## Day-8: Missing Values & Imputation (Feature Engineering – Part 1)

### Objective
Understand the causes and types of missing data and apply appropriate imputation strategies using pipeline-based preprocessing to ensure correctness, prevent data leakage, and improve model reliability.

---

### Why Handling Missing Values Is Important
Missing values are common in real-world datasets and can:
- Break machine learning algorithms
- Introduce bias into models
- Lead to incorrect evaluation if handled improperly

Correct handling of missing values is a critical part of feature engineering.

---

### Types of Missing Data
Missing values can occur for different reasons:

| Type | Description |
|-----|------------|
| MCAR | Missing Completely At Random |
| MAR | Missing At Random |
| MNAR | Missing Not At Random |

Understanding *why* data is missing helps determine the correct imputation strategy.

---

### Imputation Strategies

#### Numerical Features
- **Mean**: Suitable for symmetric distributions
- **Median**: Preferred for skewed data or presence of outliers
- **Constant values**: Used when missingness itself has meaning

#### Categorical Features
- **Most frequent value**: Suitable for low-cardinality features
- **New category (e.g., “Unknown”)**: Used when missingness is informative

---

### Best Practices
- Imputation must be learned only from training data
- Avoid imputing before train–test split
- Apply imputation inside a pipeline to prevent leakage
- Treat missing value handling as a feature engineering decision, not a cleanup task

---

### Implementation Details
- Dataset: Titanic survival dataset
- Numerical features: `age`, `fare`
- Categorical features: `sex`, `embarked`
- Techniques used:
  - `SimpleImputer` for numerical and categorical features
  - `StandardScaler` for numerical features
  - `OneHotEncoder` for categorical features
- All preprocessing steps are encapsulated inside a `Pipeline` and `ColumnTransformer`

---

### Evaluation
The preprocessing pipeline was combined with a Logistic Regression model and evaluated using a train–test split to ensure:
- No data leakage
- Consistent preprocessing across training and testing

---

### Key Takeaways
- Missing data handling significantly impacts model performance
- Different feature types require different imputation strategies
- Pipeline-based imputation ensures correctness and reproducibility
- Feature engineering decisions should be guided by data characteristics and business context

---

### Interview Readiness
After this exercise, I can confidently:
- Explain why missing values occur
- Choose appropriate imputation strategies
- Prevent data leakage during imputation
- Implement imputation safely using ML pipelines

---

### Artifacts
- Notebook: `05_missing_values_imputation.ipynb`
