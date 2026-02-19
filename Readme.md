Rental Price Prediction Project
Overview

This project predicts rental apartment prices using machine learning.

The dataset was collected from HepsiEmlak listings and contains basic property information such as:

Price

Square meters (m2)

Number of rooms

Region

Building age

The goal of this project is to build a regression model that can estimate rental prices based on property features.

Dataset

Initial dataset size: 117 rows
After cleaning and outlier removal: 112 rows

Main features used:

m2 (square meters)

rooms

building_age

region (encoded)

Target variable:

price (log transformed during training)

Data Preprocessing

The following steps were applied:

Removed duplicates

Cleaned price column (removed dots and converted to integer)

Converted room information (e.g., "3+1", "Stüdyo") to numeric

Extracted building age as numeric

Removed missing values

Removed outliers using IQR method

Applied log transformation to the target variable

Feature Engineering

Additional features created:

m2_squared (nonlinear effect)

region_encoded (Label Encoding)

Log transformation was applied to price to improve model performance and reduce skewness.

Model

Model used:

RandomForestRegressor

Main parameters:

n_estimators = 800

max_depth = 15

min_samples_split = 2

random_state = 42

Train/Test split:

80% training

20% testing

Cross validation:

5-fold cross validation

Results

Mean Absolute Error (MAE): 1996.21

R2 Score (Test Set): 0.7056

Cross Validation R2 Mean: 0.6118

The model explains approximately 70% of the variance in rental prices on the test set.

Interpretation

The strongest feature is square meters (m2).

Rooms and m2 are highly correlated (0.90), meaning they contain similar information.

The relatively small dataset size (112 rows) limits the maximum achievable accuracy. With more data and additional features (floor, furnished status, site information, etc.), performance could be significantly improved.

Project Structure

tam_kod.py → Main training script

model.pkl → Saved trained model

hepsiemlak.csv → Dataset

README.md → Project documentation

How to Run

Create virtual environment

Install dependencies:

pip install pandas numpy scikit-learn matplotlib

Run:

python tam_kod.py

The model will train and save as model.pkl.

Future Improvements

Increase dataset size

Add more property features

Try Gradient Boosting or XGBoost

Use proper categorical encoding (OneHotEncoding)

Deploy as a web application