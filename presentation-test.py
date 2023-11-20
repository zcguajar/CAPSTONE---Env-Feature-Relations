#Zack Guajardo 1219098972
# These are the packages that we are importing that shall be used throughout this Lab

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


lin_reg_df = pd.read_csv("./archive/Crop_recommendation.csv")#YOUR CODE HERE

lin_reg_df.head()

lin_reg_df.isnull().sum()

lin_reg_df = lin_reg_df.drop('label', axis=1)

input_var = input("What variable would you like to predict using our database? Temperature? Humidity? pH? Rainfall?\n")
y = lin_reg_df[["humidity"]]#, "humidity", "ph", "rainfall"]]#YOUR CODE HERE

X = lin_reg_df[["N", "P", "K"]]#YOUR CODE HERE

# After that test the model with random_state - 0, 50 and 101 and report the one that gave the best performance based on MSE, MAE and RMSE

random_state_list = [0, 50, 101]

min_MAE, min_MSE, min_RMSE, best_rdm_st = float('inf'), float('inf'), float('inf'), 0

#for rdm_st in random_state_list:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101, shuffle=True)#YOUR CODE HERE - train-test split: 75 - 25

model_LR = LinearRegression() #YOUR CODE HERE - init the Linear Regression model

model_LR.fit(X_train.values, y_train.values)#YOUR CODE HERE - fit the data into the model

y_pred = model_LR.predict(X_test)#YOUR CODE HERE - Predict using this model
y_pred = pd.DataFrame(y_pred, columns = ["humidity"])#, "humidity", "ph", "rainfall"])
predict_test =model_LR.predict( [[90,42,43]])
#print("PREDICT TEST: " + str(predict_test))
#print(y_pred)
# Use sklearn.metrics to get the values of MAE and MSE
#-------------------------------------------------------------------------------
# print("OVERALL ERROR ========================================================\n")
# MAE =  metrics.mean_absolute_error(y_test, y_pred)#YOUR CODE HERE
# MSE = metrics.mean_squared_error(y_test, y_pred)#YOUR CODE HERE
# RMSE = np.sqrt(MSE)#YOUR CODE HERE -- remember RMSE is sqare root of MSE
# print("Mean Absolute Error: ", MAE)
# print("Mean Squared Error: ", MSE)
# print("Root Mean Squared Error: ", RMSE)
# print("========================================================")
# print("\n")

#----------------------------------------------------------------------
# print("TEMERATURE ERROR =====================================================\n")
# MAE_temp =  metrics.mean_absolute_error(y_test["temperature"], y_pred["temperature"])#YOUR CODE HERE
# MSE_temp = metrics.mean_squared_error(y_test["temperature"], y_pred["temperature"])#YOUR CODE HERE
# RMSE_temp = np.sqrt(MSE_temp)
# print("Mean Absolute Error: ", MAE_temp)
# print("Mean Squared Error: ", MSE_temp)
# print("Root Mean Squared Error: ", RMSE_temp)
# print("========================================================")
# print("\n")
# x = list(range(len(y_test["temperature"])))
# plt.scatter(x, y_test["temperature"], color="blue", label="original")
# plt.plot(x, y_pred["temperature"], color="red", label="predicted")
# plt.legend()
# plt.show()
#----------------------------------------------------------------------
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
print("HUMIDITY ERROR =======================================================\n")
MAE_hum =  metrics.mean_absolute_error(y_test["humidity"], y_pred["humidity"])#YOUR CODE HERE
MSE_hum = metrics.mean_squared_error(y_test["humidity"], y_pred["humidity"])#YOUR CODE HERE
RMSE_hum = np.sqrt(MSE_hum)
print("Mean Absolute Error: ", MAE_hum)
print("Mean Squared Error: ", MSE_hum)
print("Root Mean Squared Error: ", RMSE_hum)
print("========================================================")
print("\n")
x = list(range(len(y_test["humidity"])))
plt.scatter(x, y_test["humidity"], color="blue", label="original")
plt.plot(x, y_pred["humidity"], color="red", label="predicted")
plt.legend()
plt.show()
# #----------------------------------------------------------------------
# print("PH ERROR =======================================================\n")
# MAE_ph =  metrics.mean_absolute_error(y_test["ph"], y_pred["ph"])#YOUR CODE HERE
# MSE_ph = metrics.mean_squared_error(y_test["ph"], y_pred["ph"])#YOUR CODE HERE
# RMSE_ph = np.sqrt(MSE_ph)
# print("Mean Absolute Error: ", MAE_ph)
# print("Mean Squared Error: ", MSE_ph)
# print("Root Mean Squared Error: ", RMSE_ph)
# print("========================================================")
# print("\n")
# x = list(range(len(y_test["ph"])))
# plt.scatter(x, y_test["ph"], color="blue", label="original")
# plt.plot(x, y_pred["ph"], color="red", label="predicted")
# plt.legend()
# plt.show()
# #----------------------------------------------------------------------
# print("RAINFALL ERROR =======================================================\n")
# MAE_rainfall =  metrics.mean_absolute_error(y_test["rainfall"], y_pred["rainfall"])#YOUR CODE HERE
# MSE_rainfall = metrics.mean_squared_error(y_test["rainfall"], y_pred["rainfall"])#YOUR CODE HERE
# RMSE_rainfall = np.sqrt(MSE_rainfall)
# print("Mean Absolute Error: ", MAE_rainfall)
# print("Mean Squared Error: ", MSE_rainfall)
# print("Root Mean Squared Error: ", RMSE_rainfall)
# print("========================================================")
# print("\n")
# x = list(range(len(y_test["rainfall"])))
# plt.scatter(x, y_test["rainfall"], color="blue", label="original")
# plt.plot(x, y_pred["rainfall"], color="red", label="predicted")
# plt.legend()
# plt.show()

