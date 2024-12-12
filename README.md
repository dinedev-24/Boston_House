**Boston House Price Prediction**
This project uses the Boston House Prices dataset to build a regression model that predicts the price of houses based on various features such as crime rate, average number of rooms, and accessibility to radial highways.

**Key Objectives**
Understand the relationship between house prices and various predictors.
Train and evaluate machine learning models to predict house prices.
Optimize the best model using hyperparameter tuning.
Validate and visualize model performance.
**Dataset**
Source: UCI Machine Learning Repository
**Features:**
CRIM: Per capita crime rate by town.
ZN: Proportion of residential land zoned for large lots.
INDUS: Proportion of non-retail business acres per town.
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
NOX: Nitric oxide concentration (parts per 10 million).
RM: Average number of rooms per dwelling.
AGE: Proportion of owner-occupied units built prior to 1940.
DIS: Weighted distances to five Boston employment centers.
RAD: Index of accessibility to radial highways.
TAX: Full-value property tax rate per $10,000.
PTRATIO: Pupil-teacher ratio by town.
B: Proportion of Black population by town.
LSTAT: % lower status of the population.
MEDV: Median value of owner-occupied homes (target variable).
**Workflow**
**1. Data Preprocessing**
Load and inspect the dataset.
Handle missing values (if any).
Standardize the features using StandardScaler.
**2. Model Selection**
We evaluated multiple regression models:

Linear Regression (LR)
LASSO Regression
ElasticNet (EN)
K-Nearest Neighbors (KNN)
Decision Tree (CART)
Support Vector Regression (SVR)
Gradient Boosting Regressor (GBM)
Random Forest (RF)
Extra Trees (ET)
AdaBoost (AB)
**3. Hyperparameter Tuning**
Performed grid search using GridSearchCV to identify the best parameters for the Gradient Boosting Regressor (GBM).
Optimal parameter: n_estimators=300.
**4. Model Evaluation**
Used cross-validation (KFold) to evaluate model performance using R² Score.
Visualized results with:
Comparison of R² scores for different parameter settings.
Actual vs Predicted scatter plots.
Results
Best Model: Gradient Boosting Regressor
Best Parameters: n_estimators=300
Cross-Validation Mean R²: 0.889515
Validation Set Performance:
Mean Squared Error (MSE): 11.6647
R² Score: Strong alignment between actual and predicted values.
**Visualizations:**
R² Score vs n_estimators:
Shows the performance trend across different values of n_estimators.
**Actual vs Predicted:**
Demonstrates the strong predictive capability of the best model.
**How to Run the Project**
1. Prerequisites
Python 3.x
Required Libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
**2. Installation**
Install the required packages using pip:
pip install numpy pandas matplotlib seaborn scikit-learn
**3. Run the Project**
Clone the repository or download the notebook file.
Open the notebook in Jupyter or any IDE supporting .ipynb files.
Execute the cells to preprocess the data, train models, and visualize results.
**Future Work**
Explore other ensemble methods like XGBoost or CatBoost.
Fine-tune additional hyperparameters (e.g., learning_rate, max_depth, subsample).
Perform feature engineering to extract more insights from the data.
