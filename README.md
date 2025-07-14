
# Big Mart Sales Prediction

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Project Description

This project focuses on predicting sales for a Big Mart outlet using machine learning techniques. The goal is to build a predictive model that can estimate the sales of different products at various outlets based on historical data. This model can assist in inventory management, strategic planning, and overall business decision-making.

The project encompasses several key stages, including data exploration and preprocessing, feature engineering, model training, and performance evaluation. It leverages Python and common data science libraries such as Pandas, NumPy, Scikit-learn, and Matplotlib.

## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Data

The dataset used in this project typically includes the following features:

- **Item_Identifier:** Unique product ID
- **Item_Weight:** Weight of product
- **Item_Fat_Content:** Whether the product is low fat or regular
- **Item_Visibility:** Percentage of total display area allocated to the product
- **Item_Type:** Category of the product
- **Item_MRP:** Maximum Retail Price of the product
- **Outlet_Identifier:** Unique outlet ID
- **Outlet_Establishment_Year:** Year the outlet was established
- **Outlet_Size:** Size of the outlet
- **Outlet_Location_Type:** Type of city the outlet is located in
- **Outlet_Type:** Type of outlet (grocery store or supermarket)
- **Item_Outlet_Sales:** Sales of the product in the specific outlet.  This is the target variable.

> **Note:** Ensure the dataset is placed in the `data/` directory or specify the correct path in the notebook.

## Dependencies

To run this project, you need to install the following dependencies:

- Python (>=3.6)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

You can install these dependencies using pip:

bash
git clone https://github.com/Aniket-desai-24/big-mart-sales-prediction.git
cd big-mart-sales-prediction
2.  **Data Preprocessing:**
    -   Handle missing values (imputation).
    -   Encode categorical variables (one-hot encoding, label encoding).
    -   Scale numerical features (standardization, normalization).

3.  **Feature Engineering:**
    -   Create new features from existing ones to improve model performance (e.g., age of the outlet).

4.  **Model Training:**
    -   Split the data into training and testing sets.
    -   Train a machine learning model (e.g., Linear Regression, Random Forest, Gradient Boosting).
    -   Tune hyperparameters using cross-validation.

5.  **Model Evaluation:**
    -   Evaluate the model's performance on the testing set using appropriate metrics (e.g., RMSE, R-squared).

## Data Preprocessing

The data preprocessing steps include:

-   **Handling Missing Values:** Impute missing values in `Item_Weight` using the mean or median.
-   **Encoding Categorical Variables:** Use one-hot encoding for `Item_Fat_Content`, `Item_Type`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type`.
-   **Scaling Numerical Features:** Apply standardization or normalization to `Item_Weight` and `Item_MRP`.

python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
data = pd.read_csv('data/Train.csv')

# Impute missing values
imputer = SimpleImputer(strategy='mean')
data['Item_Weight'] = imputer.fit_transform(data[['Item_Weight']])

# Encode categorical variables
encoder = LabelEncoder()
data['Outlet_Identifier'] = encoder.fit_transform(data['Outlet_Identifier'])

# Scale numerical features
scaler = StandardScaler()
data[['Item_Weight', 'Item_MRP']] = scaler.fit_transform(data[['Item_Weight', 'Item_MRP']])

print(data.head())
python
import pandas as pd
from datetime import datetime

# Load data
data = pd.read_csv('data/Train.csv')

# Calculate outlet age
current_year = datetime.now().year
data['Outlet_Age'] = current_year - data['Outlet_Establishment_Year']

print(data[['Outlet_Establishment_Year', 'Outlet_Age']].head())
python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the preprocessed data
data = pd.read_csv('data/Train.csv')
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
data = data.fillna(data.mean())

# Prepare features and target
X = data.drop(['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'], axis=1)
y = data['Item_Outlet_Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')
python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
> Briefly summarize the performance of your model. Include key metrics and insights gained from the analysis. For example: "The Random Forest model achieved an RMSE of X on the test set, indicating good predictive performance. Feature importance analysis revealed that Item_MRP and Outlet_Size were the most significant predictors of sales."

## Contributing

We welcome contributions to this project! Here are the steps to contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them.
4.  Submit a pull request.

> **Note:** Please follow the coding style and conventions used in the project.

## License

