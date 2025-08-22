# Student Performance Prediction Project Explanation

## 1. Introduction

This project aims to predict student performance (specifically, math scores) based on various factors such as gender, race/ethnicity, parental level of education, lunch type, and test preparation course completion. The dataset used for this project contains 1000 rows and 8 columns, providing a comprehensive view of student demographics and academic outcomes.

## 2. Exploratory Data Analysis (EDA)

### 2.1 Data Loading and Initial Checks

The project begins by loading the `stud.csv` dataset into a Pandas DataFrame. Initial checks were performed to understand the dataset's structure and content. The `df.head()` command revealed the first five entries, showing the columns: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `math_score`, `reading_score`, and `writing_score`. The dataset's shape (`df.shape`) confirmed it contains 1000 rows and 8 columns.

### 2.2 Data Cleaning and Preprocessing

Crucial data quality checks were performed:
- **Missing Values**: `df.isna().sum()` was used to identify any missing values. The analysis confirmed that there are no missing values across all columns, indicating a clean dataset in this regard.
- **Duplicates**: `df.duplicated().sum()` was used to check for duplicate entries. The result showed no duplicate rows, ensuring the uniqueness of each student record.
- **Data Types**: `df.info()` provided a summary of the DataFrame, including data types. It was observed that `math_score`, `reading_score`, and `writing_score` are numerical (int64), while the other five columns are categorical (object).
- **Unique Values**: `df.nunique()` was used to count the number of unique values in each column, providing insight into the cardinality of categorical features.

### 2.3 Key Insights from EDA

Statistical insights from `df.describe()` for numerical columns (`math_score`, `reading_score`, `writing_score`) revealed:
- The mean scores for all three subjects are very close, ranging from approximately 66 to 69.
- Standard deviations are also similar, between 14.6 and 15.19, suggesting comparable score dispersion across subjects.
- A notable observation is the minimum score: math_score has a minimum of 0, while reading_score and writing_score have higher minimums (17 and 10, respectively). This indicates that some students performed very poorly in math, potentially scoring zero.

Further exploratory analysis involved visualizing the data to understand distributions and relationships. This included:
- Distribution plots for numerical scores to observe their spread and skewness.
- Count plots for categorical features to understand the distribution of students across different categories (gender, race/ethnicity, parental education, lunch, test preparation).
- Bar plots and box plots to analyze the impact of categorical features on student scores, revealing potential correlations and differences in performance based on these factors.

## 3. Model Training

### 3.1 Feature Engineering and Data Preprocessing

Before model training, the data underwent significant preprocessing:
- **Feature Separation**: The dataset was split into features (X) and the target variable (y). The `math_score` was chosen as the target variable (y), while all other columns formed the feature set (X).
- **Column Transformation**: A `ColumnTransformer` was employed to handle different types of features:
    - **One-Hot Encoding**: Categorical features (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`) were transformed using `OneHotEncoder` to convert them into a numerical format suitable for machine learning models.
    - **Standard Scaling**: Numerical features (`reading_score`, `writing_score`) were scaled using `StandardScaler` to normalize their ranges, preventing features with larger values from dominating the model training process.

After transformation, the feature matrix `X` had a shape of (1000, 19), indicating 1000 samples and 19 processed features.

### 3.2 Model Selection and Training Process

The preprocessed data was split into training and testing sets using `train_test_split`, with 80% of the data allocated for training and 20% for testing (`test_size=0.2, random_state=42`).

A custom `evaluate_model` function was defined to calculate key regression metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R2 Score. This function facilitated a standardized evaluation of all models.

The following regression models were trained and evaluated:
- Linear Regression
- Lasso Regression
- Ridge Regression
- K-Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- AdaBoost Regressor
- CatBoosting Regressor
- XGBRegressor

Each model was trained on the `X_train` and `y_train` datasets, and their performance was assessed on both the training and testing sets using the `evaluate_model` function.

### 3.3 Model Evaluation and Best Model Identification

The performance of each model was systematically recorded and compared. The evaluation metrics (RMSE, MAE, R2 Score) for both training and test sets were printed for each model. This allowed for a direct comparison of how well each model generalized to unseen data and identified potential overfitting or underfitting issues.

Based on the R2 Score, which indicates the proportion of variance in the dependent variable that can be predicted from the independent variables, the models were ranked. The model with the highest R2 Score on the test set, while maintaining a low RMSE and MAE, was considered the best performing model for predicting student math scores.

