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

## Day-9: Categorical Encoding (Feature Engineering – Part 2)

### Objective
Understand and apply appropriate categorical encoding techniques based on model type and feature cardinality, while preventing data leakage using pipeline-based preprocessing.

---

### Why Categorical Encoding Is Required
Most machine learning algorithms operate on numerical data and cannot directly interpret categorical values.  
Categorical encoding converts categorical information into numerical form without introducing unintended relationships.

---

### Common Encoding Techniques

#### Label Encoding
- Assigns an integer to each category
- Suitable for tree-based models (Random Forest, XGBoost)
- Risk of introducing artificial ordering for linear or distance-based models

#### One-Hot Encoding
- Creates binary features for each category
- Suitable for low-cardinality categorical features
- Commonly used with linear models
- Can increase dimensionality if cardinality is high

#### High-Cardinality Handling
For features with many unique categories (e.g., city, product ID):
- Frequency encoding
- Target encoding (used cautiously with proper validation)
- Grouping rare categories into an “Other” category

---

### Best Practices
- Avoid one-hot encoding high-cardinality features
- Never encode data before train–test split
- Implement encoding inside pipelines to prevent data leakage
- Choose encoding strategy based on model type and feature characteristics

---

### Implementation Details
- Dataset: Titanic survival dataset
- Numerical features: `age`, `fare`
- Categorical features: `sex`, `embarked`
- Techniques used:
  - `SimpleImputer` for missing categorical values
  - `OneHotEncoder` with `handle_unknown='ignore'`
  - `ColumnTransformer` for mixed feature preprocessing
- Model: Logistic Regression

All preprocessing and model training steps are encapsulated within a single pipeline.

---

### Evaluation
The pipeline was evaluated using a train–test split to ensure:
- Correct preprocessing
- No data leakage
- Consistent feature encoding during training and inference

---

### Key Takeaways
- Encoding strategy impacts both model performance and interpretability
- Incorrect encoding can introduce bias or unnecessary complexity
- Pipeline-based encoding ensures correctness and reproducibility
- High-cardinality features require special handling

---

### Interview Readiness
After this exercise, I can confidently:
- Explain different categorical encoding strategies
- Choose the appropriate encoding based on model and data
- Handle high-cardinality categorical features
- Prevent data leakage during encoding

---

### Artifacts
- Notebook: `06_categorical_encoding.ipynb`


## Day-10: Feature Scaling (StandardScaler vs MinMaxScaler)

### Objective
Understand when and why feature scaling is required, choose the appropriate scaling technique based on model type and data distribution, and apply scaling safely using pipeline-based preprocessing.

---

### Why Feature Scaling Is Important
Feature scaling ensures that all numerical features contribute proportionally during model training.  
Without scaling, features with larger numeric ranges can dominate learning and negatively impact model performance.

---

### Models That Require Feature Scaling

| Model Type | Scaling Required | Reason |
|-----------|------------------|--------|
| Linear / Logistic Regression | Yes | Gradient-based optimization |
| Support Vector Machines | Yes | Distance-based |
| K-Nearest Neighbors | Yes | Distance-based |
| Neural Networks | Yes | Optimization stability |
| Tree-based Models | No | Scale-invariant splits |

---

### Common Scaling Techniques

#### StandardScaler
- Standardizes features to mean = 0 and standard deviation = 1
- Works well with linear models and neural networks
- Sensitive to outliers

#### MinMaxScaler
- Scales features to a fixed range, typically [0, 1]
- Preserves relative distances
- Sensitive to outliers

---

### Choosing the Right Scaler
- Use **StandardScaler** for linear and neural network models
- Use **MinMaxScaler** for bounded features or algorithms requiring normalized ranges
- Consider **RobustScaler** when outliers are present

---

### Best Practices
- Always scale numerical features only
- Never scale before train–test split
- Fit scalers only on training data
- Apply scaling inside pipelines to prevent data leakage

---

### Implementation Details
- Dataset: Titanic survival dataset
- Numerical features: `age`, `fare`
- Categorical features: `sex`, `embarked`
- Techniques used:
  - Median imputation for missing values
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- All preprocessing is encapsulated inside a `Pipeline` and `ColumnTransformer`
- Model: Logistic Regression

---

### Evaluation
The model pipeline was evaluated using a train–test split to ensure:
- Correct application of scaling
- Consistent preprocessing across training and testing
- No data leakage

---

### Key Takeaways
- Feature scaling is critical for many ML algorithms
- Incorrect scaling can degrade model performance
- Scaler selection depends on model type and data distribution
- Pipeline-based scaling ensures reproducibility and correctness

---

### Interview Readiness
After this exercise, I can confidently:
- Explain when and why feature scaling is required
- Differentiate between StandardScaler and MinMaxScaler
- Identify models that require scaling
- Apply scaling safely using pipelines

---

### Artifacts
- Notebook: `07_feature_scaling.ipynb`



## Day-11: Feature Importance Methods (Model Interpretability)

### Objective
Understand and apply feature importance techniques to interpret machine learning models, validate feature usefulness, detect potential data leakage, and improve model trust and transparency.

---

### Why Feature Importance Is Important
Feature importance helps answer critical questions such as:
- Which features influence model predictions the most?
- Are there redundant or useless features?
- Is the model relying on unexpected or leaked information?

In production systems, feature importance is essential for debugging, governance, and stakeholder trust.

---

### Types of Feature Importance Methods

#### 1. Coefficient-Based Importance (Linear Models)
- Uses magnitude of model coefficients
- Requires features to be on comparable scales
- Assumes linear relationships

**Limitation:** Not reliable without proper scaling or for non-linear relationships.

---

#### 2. Tree-Based Feature Importance
- Based on reduction in impurity (e.g., Gini importance)
- Available in tree-based models like Random Forest and Gradient Boosting

**Limitation:** Biased toward high-cardinality features.

---

#### 3. Permutation Importance (Preferred)
- Shuffles one feature at a time and measures performance drop
- Model-agnostic
- Reflects true impact on model predictions

This method provides a more reliable estimate of feature importance compared to model-internal metrics.

---

### Best Practices
- Compute feature importance on validation or test data
- Avoid interpreting unscaled coefficients
- Do not rely solely on tree-based importance
- Use feature importance as a diagnostic and validation tool

---

### Implementation Details
- Dataset: Titanic survival dataset
- Model: Pipeline with preprocessing and Logistic Regression
- Feature importance method: Permutation Importance
- Evaluation metric: Accuracy

Permutation importance was computed on the test dataset to reflect real-world generalization behavior.

---

### Key Takeaways
- Feature importance improves model transparency and trust
- Different importance methods have different assumptions and limitations
- Permutation importance is a robust, model-agnostic choice
- Feature importance can guide feature selection and model refinement

---

### Interview Readiness
After this exercise, I can confidently:
- Explain why feature importance matters
- Compare different feature importance techniques
- Justify the use of permutation importance
- Identify pitfalls in feature importance interpretation

---

### Artifacts
- Notebook: `09_feature_importance2.ipynb`



## Day-12: Data Leakage Detection & Prevention

### Objective
Identify, detect, and prevent data leakage in machine learning workflows to ensure reliable model evaluation and real-world performance.

---

### What Is Data Leakage?
Data leakage occurs when information that would not be available at prediction time is used during model training.  
This results in unrealistically high evaluation metrics and poor performance in production.

---

### Types of Data Leakage

#### 1. Train–Test Leakage
Occurs when preprocessing steps (scaling, imputation, encoding) are applied before splitting data into training and test sets.

**Prevention:** Always apply preprocessing inside pipelines after the train–test split.

---

#### 2. Target Leakage
Occurs when a feature contains direct or indirect information about the target variable.

**Examples:**
- Features derived from post-outcome data
- Aggregates that include the target itself

**Prevention:** Validate that every feature would be available at prediction time.

---

#### 3. Time Leakage
Occurs when future information is used to predict past events, commonly in time-series or event data.

**Prevention:** Use time-aware splits and ensure features are computed using only past data.

---

#### 4. Cross-Validation Leakage
Occurs when preprocessing or encoding is performed outside the cross-validation loop.

**Prevention:** Use pipeline-based preprocessing so transformations are refit within each fold.

---

### How to Detect Data Leakage
Common indicators include:
- Extremely high accuracy early in modeling
- Large gap between training and validation performance
- Feature importance highlighting unrealistic or future-based features

Regular feature review and validation are essential.

---

### Prevention Best Practices
- Use `Pipeline` and `ColumnTransformer` for all preprocessing
- Validate feature availability at prediction time
- Apply appropriate validation strategies (time-based, group-based splits)
- Compute feature importance to detect suspicious features

---

### Data Engineering Perspective
Data leakage often originates in data pipelines:
- Aggregations must be time-aware
- Feature stores should enforce point-in-time correctness
- ETL jobs should avoid using future data

Preventing leakage requires collaboration between data engineering and ML workflows.

---

### Key Takeaways
- Data leakage is a system-level issue, not just a modeling issue
- Pipeline-based preprocessing is essential
- Correct validation strategies prevent misleading evaluation
- Leakage prevention improves model trust and reliability

---

### Interview Readiness
After this exercise, I can confidently:
- Explain different types of data leakage
- Detect leakage using evaluation signals and feature analysis
- Prevent leakage using pipelines and validation strategies
- Discuss leakage from a data engineering perspective

## Day-13: EDA for Machine Learning (Feature-Focused Exploration)

### Objective
Perform exploratory data analysis (EDA) with the goal of improving machine learning model quality by understanding feature distributions, feature–target relationships, and potential data issues before modeling.

---

### EDA for ML vs EDA for BI
EDA for machine learning differs from business intelligence analysis in its intent:

| EDA for BI | EDA for ML |
|-----------|-----------|
| Reporting and dashboards | Diagnostic and decision-driven |
| Aggregated metrics | Feature-level distributions |
| KPI-focused | Model performance–focused |

EDA in ML directly informs feature engineering and modeling decisions.

---

### Key Questions Addressed During EDA
- Is the target variable balanced?
- Do numerical features show skewness or outliers?
- Which features have predictive signal?
- Are there redundant or weak features?
- Are there indicators of potential data leakage?

---

### ML-Focused EDA Steps

#### Target Variable Analysis
- Examine class distribution
- Identify imbalance that may require stratified splitting or reweighting

#### Numerical Feature Distributions
- Analyze feature spread and skewness
- Identify outliers
- Decide on transformations or scaling strategies

#### Feature–Target Relationships
- Evaluate how categorical and numerical features relate to the target
- Identify features with strong predictive signal

#### Correlation Analysis
- Detect redundant numerical features
- Avoid multicollinearity in linear models

#### Leakage Signals
- Identify features with unusually strong correlation to the target
- Validate feature availability at prediction time

---

### Best Practices
- Use EDA to guide decisions, not just visualization
- Focus on features, not dashboards
- Document observations and modeling implications
- Avoid over-plotting and unnecessary complexity

---

### Implementation Details
- Dataset: Titanic survival dataset
- Features analyzed:
  - Numerical: `age`, `fare`
  - Categorical: `sex`, `embarked`
- Techniques used:
  - Distribution analysis
  - Feature–target grouping
  - Correlation checks

All findings were documented with clear implications for feature engineering and modeling.

---

### Key Takeaways
- EDA is a critical step in building reliable ML models
- Feature quality matters more than algorithm choice
- Early detection of data issues improves downstream performance
- EDA helps identify transformation, encoding, and scaling needs

---

### Interview Readiness
After this exercise, I can confidently:
- Explain the purpose of EDA in ML
- Differentiate ML-focused EDA from BI analysis
- Identify feature issues that impact model performance
- Use EDA findings to guide feature engineering decisions

---

### Artifacts
- Notebook: `11_eda_for_ml.ipynb`

## Day-15: Linear Regression & Regularization

### Objective
Understand linear regression from a modeling perspective, identify why overfitting can occur even in simple models, and apply regularization techniques (Ridge and Lasso) to improve generalization.

---

### Linear Regression Overview
Linear regression models the relationship between input features and a continuous target by minimizing the squared error between predicted and actual values.

It assumes a linear relationship between features and the target variable.

---

### Why Linear Models Can Overfit
Although linear regression is a simple model, overfitting can occur when:
- The number of features is large
- Features are highly correlated
- Polynomial or interaction features are added
- The dataset size is small

Regularization is used to control model complexity in these scenarios.

---

### Regularization Concept
Regularization adds a penalty to large model coefficients during training.  
This discourages overly complex models and improves performance on unseen data.

The optimization objective becomes:

```Loss = Prediction Error + Regularization Penalty```


---

### Types of Regularization

#### Ridge Regression (L2)
- Penalizes the squared magnitude of coefficients
- Shrinks coefficients toward zero
- Retains all features in the model
- Useful when features are correlated

#### Lasso Regression (L1)
- Penalizes the absolute magnitude of coefficients
- Can shrink some coefficients exactly to zero
- Performs implicit feature selection

#### ElasticNet
- Combines L1 and L2 penalties
- Balances coefficient shrinkage and feature selection

---

### Role of the Regularization Parameter (Alpha)
- Controls the strength of regularization
- Higher alpha increases bias and reduces variance
- Lower alpha allows more complex models

Alpha is typically tuned using cross-validation on training data.

---

### Best Practices
- Always scale features before applying regularization
- Tune regularization strength using cross-validation
- Compare models using the same features and evaluation strategy
- Do not tune hyperparameters on the test set

---

### Implementation Details
- Dataset: Tips dataset
- Model types: Linear Regression, Ridge, Lasso
- Preprocessing: Feature scaling using `StandardScaler`
- Evaluation metric: R² score
- All preprocessing and modeling steps are implemented using pipelines

---

### Key Takeaways
- Regularization improves generalization, not training accuracy
- Ridge and Lasso address overfitting in different ways
- Feature scaling is critical when using regularized models
- Linear models remain powerful when combined with proper regularization

---

### Interview Readiness
After this exercise, I can confidently:
- Explain why regularization is needed
- Differentiate between Ridge and Lasso regression
- Describe how regularization affects bias and variance
- Implement regularized linear models in a production-safe way

---

### Artifacts
- Notebook: `12_linear_regression_regularization.ipynb`

## Day-16: Logistic Regression & Threshold Tuning

### Objective
Understand logistic regression as a probabilistic classification model and learn how decision thresholds affect business outcomes through precision–recall trade-offs.

---

### Logistic Regression Overview
Logistic regression is a linear model that predicts the probability of a binary outcome using a sigmoid function.  
It outputs probabilities in the range [0, 1], not class labels.

---

### Probability vs Decision
Logistic regression produces probabilities, which must be converted into class labels using a decision threshold.

The default threshold of 0.5 is arbitrary and should not be blindly applied.

---

### Threshold Tuning
Decision thresholds directly control the balance between:
- Precision (false positives)
- Recall (false negatives)

Thresholds should be selected based on business costs and risk tolerance.

---

### Evaluation Metrics
Accuracy alone is insufficient, especially for imbalanced datasets.

Key metrics:
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (preferred for class imbalance)

---

### Best Practices
- Always use stratified train–test splits
- Tune thresholds on validation data, not test data
- Evaluate models using multiple metrics
- Treat classification as a probability estimation problem

---

### Implementation Details
- Dataset: Titanic dataset
- Model: Logistic Regression
- Preprocessing: Feature scaling using `StandardScaler`
- Output: Probability-based predictions
- Threshold tuning applied post-training

---

### Key Takeaways
- Logistic regression predicts probabilities, not decisions
- Threshold selection is a business-driven choice
- Precision–recall trade-offs must be explicitly evaluated
- Simple models become powerful with correct thresholding

---

### Interview Readiness
After this exercise, I can confidently:
- Explain how logistic regression works
- Justify why thresholds must be tuned
- Choose appropriate evaluation metrics
- Align model decisions with business objectives

---

### Artifacts
- Notebook: `13_logistic_regression_thresholds.ipynb`


## Day-17: Decision Trees (Classification & Regression)

### Objective
Understand how decision trees work internally, how they select splits using mathematical impurity measures, why they overfit, and how to control their complexity in real-world machine learning systems.

---

## Decision Trees Overview
Decision trees are rule-based models that recursively split data into smaller, more homogeneous groups using feature thresholds.

They are:
- Easy to interpret
- Powerful for non-linear relationships
- Highly prone to overfitting if not controlled

---

## Classification Trees: Impurity Measures

### Gini Impurity
Gini impurity measures how mixed the classes are in a node.

**Formula:**

```Gini = 1 − Σ (pᵢ²)```

Where:
- pᵢ = proportion of class i in the node

**Properties:**
- Gini = 0 → pure node
- Faster to compute
- Default criterion in sklearn

---

### Entropy
Entropy measures uncertainty in a node.

**Formula:**
```Entropy = − Σ (pᵢ log₂ pᵢ)```

**Properties:**
- Entropy = 0 → pure node
- Higher value → more uncertainty
- Used in ID3 algorithm

---

### Information Gain
Information Gain measures the reduction in entropy after a split.

**Formula:**

```Information Gain = Entropy(parent) − Σ (Nⱼ / N) × Entropy(childⱼ)```


Where:
- N = samples in parent node
- Nⱼ = samples in child node

**Goal:** Maximize information gain

---

## Regression Trees: Variance Reduction

### Variance (Node Impurity)
Regression trees measure impurity using variance of target values.

**Formula:**

```Variance = (1 / n) Σ (yᵢ − ȳ)²```


Where:
- yᵢ = target value
- ȳ = mean target value

---

### Variance Reduction
Splits are chosen to maximize reduction in variance.

**Formula:**

```Variance Reduction = Variance(parent) − Σ (Nⱼ / N) × Variance(childⱼ)```



---

## Why Decision Trees Overfit
- Trees can grow very deep
- They can memorize noise
- Small leaf nodes lead to unstable predictions

A fully grown tree usually has low bias but very high variance.

---

## Controlling Tree Complexity

Key hyperparameters:
- `max_depth` – limits tree height
- `min_samples_split` – avoids small splits
- `min_samples_leaf` – smooths predictions
- `max_features` – introduces randomness

Controlling these improves generalization.

---

## Classification vs Regression Trees

| Aspect | Classification Tree | Regression Tree |
|------|---------------------|----------------|
| Target | Categorical | Continuous |
| Impurity Metric | Gini / Entropy | Variance |
| Split Objective | Max impurity reduction | Max variance reduction |
| Prediction | Majority class | Mean value |

---

## Best Practices
- Never allow unrestricted tree growth
- Tune depth and minimum samples using validation data
- Avoid relying only on accuracy
- Trees work best as part of ensembles

---

## Interview Readiness
After this exercise, I can confidently:
- Explain how decision trees choose splits mathematically
- Differentiate Gini, entropy, and information gain
- Explain regression trees using variance reduction
- Justify hyperparameter choices to control overfitting

---

## Key Takeaways
- Trees split data to maximize purity or reduce variance
- Classification trees minimize impurity
- Regression trees minimize variance
- Interpretability comes at the cost of stability

---

## Artifacts
- Notebook: `14_decision_trees.ipynb`

## Day-18: Random Forest (Bagging, Variance Reduction & OOB)

### Objective
Understand how Random Forest improves upon decision trees using bagging and feature randomness, how it reduces variance, and how Out-of-Bag (OOB) error provides an internal validation mechanism.

---

## Why Random Forest?
Single decision trees are:
- Highly sensitive to data variations
- Prone to overfitting
- High variance models

Random Forest addresses this by combining multiple decorrelated trees.

---

## What Is Random Forest?
Random Forest is an ensemble learning algorithm that:
- Trains multiple decision trees
- Uses bootstrap sampling (sampling with replacement)
- Considers a random subset of features at each split
- Aggregates predictions from all trees

Prediction:
- Classification → Majority vote
- Regression → Average of predictions

---

## Bagging (Bootstrap Aggregation)

### Process
1. Sample training data with replacement
2. Train independent trees
3. Aggregate predictions

### Why Bagging Works
Averaging multiple high-variance models reduces overall variance.

**Key idea:**
> Averaging uncorrelated models improves generalization.

---

## Feature Randomness
At each split, Random Forest considers only a subset of features.

This:
- Reduces correlation between trees
- Strengthens ensemble performance
- Improves variance reduction

---

## Out-of-Bag (OOB) Samples

### What Is OOB?
When bootstrapping:
- Not all samples are selected for a tree
- Samples not selected are called **Out-of-Bag (OOB)** samples

Probability a sample is NOT selected:

```(1 − 1/N)ⁿ ≈ e⁻¹ ≈ 36.8% ```


Approximately **36% of data is OOB for each tree**.

---

## OOB Error Estimation

### How OOB Works
For each data point:
1. Collect predictions from trees where the point was OOB
2. Aggregate those predictions
3. Compare with true label

This produces an unbiased estimate of generalization error.

### OOB Error (Conceptual Formula)

``` OOB Error = (1/N) Σ I(yᵢ ≠ ŷᵢᴼᴼᴮ) ```


---

## Why OOB Is Useful
- No separate validation set required
- Acts like built-in cross-validation
- Computationally efficient

**Senior insight:**
> OOB provides a free validation estimate during training.

---

## Random Forest Hyperparameters

| Parameter | Purpose |
|--------|--------|
| n_estimators | Number of trees |
| max_depth | Controls tree complexity |
| max_features | Feature randomness |
| min_samples_leaf | Prevents noisy splits |
| bootstrap | Enables bagging |

---

## Feature Importance in Random Forest
Random Forest provides impurity-based feature importance.

**Caution:**
- Biased toward high-cardinality features
- Should be validated using permutation importance

---

## Best Practices
- Do not rely solely on default parameters
- Monitor OOB score for overfitting signals
- Use Random Forest as a strong baseline
- Prefer permutation importance for interpretation

---

## Random Forest vs Decision Tree

| Aspect | Decision Tree | Random Forest |
|----|----|----|
| Variance | High | Reduced |
| Interpretability | High | Medium |
| Stability | Low | High |
| Overfitting | Likely | Less likely |

---

## Interview Readiness
After this exercise, I can confidently:
- Explain bagging and variance reduction
- Describe OOB samples and OOB error
- Justify Random Forest hyperparameters
- Compare Random Forest with single trees and boosting

---

## Key Takeaways
- Random Forest reduces variance, not bias
- Bagging + feature randomness decorrelates trees
- OOB error provides internal validation
- Random Forest is a strong production baseline

---

## Artifacts
- Notebook: `15_random_forest.ipynb`

## Day-19: XGBoost (Boosting & Residual Learning)

### Objective
Understand gradient boosting from first principles, how XGBoost learns sequentially from errors, and how regularization and shrinkage make it a powerful and robust algorithm in production systems.

---

## Why Boosting?
Boosting addresses the **bias problem** in machine learning models.

Unlike bagging (Random Forest), boosting:
- Trains models sequentially
- Focuses on difficult samples
- Gradually improves predictions

---

## What Is XGBoost?
XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting that:
- Builds decision trees sequentially
- Fits each tree on the residuals of the previous ensemble
- Uses explicit regularization to control overfitting

---

## Residual Learning (Core Idea)
At iteration \(t\):

```ŷ⁽ᵗ⁾ = ŷ⁽ᵗ⁻¹⁾ + η · fₜ(x)```


Where:
- fₜ(x) = newly trained tree
- η = learning rate (shrinkage)

Each new tree corrects the mistakes of previous trees.

---

## Objective Function
XGBoost minimizes a regularized objective:

```Objective = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₜ)```


Where:
- L = loss function
- Ω = regularization term

---

## Regularization in XGBoost
XGBoost explicitly penalizes model complexity:

```Ω(f) = γT + ½λ Σ wⱼ²```


Where:
- T = number of leaves
- wⱼ = leaf weights
- γ, λ = regularization parameters

This helps prevent overfitting.

---

## Why XGBoost Works Well
- Bias reduction through boosting
- Shrinkage via learning rate
- Subsampling of rows and columns
- Explicit regularization
- Efficient implementation

---

## Key Hyperparameters

| Parameter | Purpose |
|--------|--------|
| n_estimators | Number of trees |
| learning_rate | Contribution of each tree |
| max_depth | Tree complexity |
| subsample | Row sampling |
| colsample_bytree | Feature sampling |
| gamma | Split penalty |
| lambda | L2 regularization |

---

## Best Practices
- Use small learning rates with more trees
- Apply early stopping
- Tune regularization parameters
- Monitor validation performance carefully

---

## Random Forest vs XGBoost

| Aspect | Random Forest | XGBoost |
|-----|-----|-----|
| Training | Parallel | Sequential |
| Bias | Moderate | Low |
| Variance | Reduced | Can increase |
| Regularization | Implicit | Explicit |
| Interpretability | Medium | Lower |

---

## When NOT to Use XGBoost
- Very small datasets
- Strict latency requirements
- Scenarios requiring high interpretability

---

## Interview Readiness
After this exercise, I can confidently:
- Explain boosting and residual learning
- Describe XGBoost’s objective function
- Justify key hyperparameters
- Compare XGBoost with Random Forest

---

## Key Takeaways
- XGBoost reduces bias through sequential learning
- Regularization is central to its success
- Learning rate controls stability
- Powerful models require careful tuning

---

## Artifacts
- Notebook: `16_xgboost.ipynb`


## Day-20: Model Comparison & Selection (Bias–Variance Trade-off)

### Objective
Learn how to compare multiple machine learning models fairly, evaluate them using appropriate metrics, and select the best model based on performance, stability, and real-world constraints.

---

## Why Model Comparison Matters
In real-world systems, several models may perform similarly on metrics.  
The goal is not to choose the most complex model, but the one that best fits the data, business objectives, and system constraints.

Model selection is an engineering decision, not a leaderboard exercise.

---

## Fair Model Comparison Principles
To ensure a valid comparison:
- Use the same dataset and features
- Apply identical preprocessing and pipelines
- Use the same train/validation splits
- Evaluate using the same metrics
- Avoid tuning on the test set

Any difference in performance should come from the model, not the setup.

---

## Models Compared
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

These models represent increasing complexity and different bias–variance characteristics.

---

## Bias–Variance Trade-off

| Model | Bias | Variance |
|-----|-----|-----|
| Logistic Regression | High | Low |
| Decision Tree | Low | High |
| Random Forest | Medium | Low |
| XGBoost | Low | Medium–High |

Ensemble methods improve performance by managing this trade-off.

---

## Metric Selection
Metric choice depends on the problem type and business objective.

| Scenario | Preferred Metrics |
|-------|----------------|
| Balanced classification | Accuracy, ROC-AUC |
| Imbalanced classification | Recall, PR-AUC |
| Regression | RMSE, MAE |

Metrics define what “best” means.

---

## Practical Model Comparison Workflow
1. Train all models using pipelines
2. Evaluate with cross-validation or OOB (for Random Forest)
3. Compare both metrics and error patterns
4. Consider non-metric constraints (latency, explainability)

---

## Example Comparison Summary

| Model | Strengths | Limitations |
|----|----|----|
| Logistic Regression | Interpretable, fast | Limited to linear patterns |
| Decision Tree | Explainable rules | Unstable, overfits |
| Random Forest | Stable, strong baseline | Slower inference |
| XGBoost | High accuracy | Requires tuning, less interpretable |

---

## Model Selection Criteria
Final model selection should consider:
- Predictive performance
- Generalization stability
- Interpretability requirements
- Latency and resource constraints
- Maintenance complexity

---

## Common Mistakes
- Always choosing the most complex model
- Optimizing only for accuracy
- Ignoring inference and operational costs
- Comparing models with inconsistent setups

---

## Interview Readiness
After this exercise, I can confidently:
- Explain how to compare models fairly
- Discuss bias–variance trade-offs
- Choose appropriate evaluation metrics
- Justify model selection decisions in interviews

---

## Key Takeaways
- Fair comparison is critical
- Metrics must align with business goals
- Simpler models often outperform in production
- Model selection is a trade-off decision

---

## Day-22: Classification Metrics & Confusion Matrix

### Objective
Understand how classification models are evaluated using the confusion matrix and derived metrics, and learn to select evaluation metrics based on business risk rather than accuracy alone.

---

## Confusion Matrix
The confusion matrix is the foundation of all classification metrics.

|                | Predicted Positive | Predicted Negative |
|----------------|-------------------|-------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

Every evaluation metric is derived from these four values.

---

## Core Classification Metrics

### Accuracy
``` Accuracy = (TP + TN) / (TP + TN + FP + FN) ```


- Measures overall correctness
- Misleading for imbalanced datasets
- Does not differentiate error types

---

### Precision
``` Precision = TP / (TP + FP) ```


- Measures correctness of positive predictions
- High precision means fewer false positives
- Important when false positives are costly

---

### Recall (Sensitivity)
``` Recall = TP / (TP + FN) ```


- Measures how many actual positives were identified
- High recall means fewer false negatives
- Critical when missing positives is costly

---

### F1-Score

``` F1 = 2 × (Precision × Recall) / (Precision + Recall) ```


- Harmonic mean of precision and recall
- Balances false positives and false negatives
- Preferred for imbalanced classification

---

## Why Accuracy Is Often Misleading
In highly imbalanced datasets, a model can achieve high accuracy while failing to detect the minority class.

Example:
- 99% negative class
- Model predicts all negatives
- Accuracy = 99%
- Recall = 0

High accuracy does not imply a useful model.

---

## Metric Selection Based on Business Risk

| Use Case | Priority Metric |
|--------|----------------|
| Fraud detection | Recall |
| Medical diagnosis | Recall |
| Spam filtering | Precision |
| Balanced classification | F1-score / ROC-AUC |

Metrics encode business cost and risk.

---

## Macro, Micro, and Weighted Metrics

| Metric Type | Description |
|-----------|------------|
| Micro | Aggregates contributions of all classes |
| Macro | Treats all classes equally |
| Weighted | Accounts for class imbalance |

Macro metrics are useful for detecting minority-class failure.

---

## Best Practices
- Always inspect the confusion matrix
- Avoid relying on accuracy alone
- Choose metrics aligned with business objectives
- Evaluate multiple metrics simultaneously

---

## Interview Readiness
After this exercise, I can confidently:
- Explain the confusion matrix
- Differentiate precision, recall, and F1-score
- Select evaluation metrics based on problem context
- Explain why accuracy can be misleading

---

## Key Takeaways
- All classification metrics derive from the confusion matrix
- Metric choice defines model success
- Accuracy is often insufficient
- Evaluation should reflect real-world costs

---

## Artifacts
- Notebook: `17_classification_metrics.ipynb`
