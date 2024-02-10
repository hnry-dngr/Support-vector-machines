# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:52:08 2024

@author: HP
"""

# Instructions:

# Import you data and perform basic data exploration phase
# Display general information about the dataset
# Create a pandas profiling reports to gain insights into the dataset
# Handle Missing and corrupted values
# Remove duplicates, if they exist
# Handle outliers, if they exist
# Encode categorical features
# Select your target variable and the features.
# Split your dataset to training and test sets
# Build and train an SVM model on the training set
# Assess your model performance on the test set using relevant evaluation metrics
# Discuss with your cohort alternative ways to improve your model performance


import warnings
import pandas as pd
import numpy as np
import sweetviz as sv
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


warnings.filterwarnings("ignore")


data = pd.read_csv("C:/Users/HP/Downloads/Electric_cars_dataset.csv")
data1 = pd.read_csv("C:/Users/HP/Downloads/Electric_cars_dataset.csv")

# Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_more_desc_statistic = data.describe(include = "all")
data_mode = data.mode()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr() 
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_hist = data.hist(figsize = (15, 10), bins = 10)


# my_report = sv.analyze(data)
# my_report.show_html()

# Removing Missing values
# data = data.dropna()

# # Removing Duplicate Values
# data = data.drop_duplicates()

# Handling nan issues
data["Expected Price ($1k)"] = data["Expected Price ($1k)"].replace({"N/": np.nan})




# Relacing missing values
imputer = SimpleImputer(strategy= "most_frequent")
new_data = imputer.fit_transform(data)
new_data = pd.DataFrame(new_data, columns= imputer.feature_names_in_)

# You convert to numeric
new_data["Expected Price ($1k)"] = pd.to_numeric(new_data["Expected Price ($1k)"])
new_data.info()
checknull = new_data.isnull().sum()

# droping irrelivant columns
new_data = new_data.drop(["ID", "VIN (1-10)", "County", "Model", "City", "DOL Vehicle ID", "ZIP Code", "Vehicle Location"], axis = 1)
new_data = pd.get_dummies(new_data, drop_first = True)

# Instantiate StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_data)
scaled_data = pd.DataFrame(scaled_data, columns = scaler.feature_names_in_)


# converting to numerical
new_data.info()

# y = scaled_data["Expected Price ($1k)"]
# x = scaled_data.drop("Expected Price ($1k)", axis = 1)

# x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 10)
# regressor = SVR()
# model = regressor.fit(x_train,y_train)

# y_train_pred = model.predict(x_train)
# y_pred = model.predict(x_test)
    

# mse_train = mean_squared_error(y_train,y_train_pred)
# print("RMSE_train: ",math.sqrt(mse_train))
# print("r_squared_train: ",metrics.r2_score(y_train, y_train_pred))

# mse_test = mean_squared_error(y_test, y_pred)
# print("RMSE_test: ",math.sqrt(mse_test))
# print("r_squared_test: ",metrics.r2_score(y_test, y_pred))




