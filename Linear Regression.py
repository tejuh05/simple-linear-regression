#import library and function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#create or load dataset
data = {
    'Rainfall(mm)':[500,600,700,800,900,1000,1100],
    'Crop Yield (Quintals/Ha)':[20,25,30,35,40,45,50]
}
df=pd.DataFrame(data)

#Display dataset
df

#Independent variable (X) and dependent variable (y)
X = df[['Rainfall(mm)']]
y = df['Crop Yield (Quintals/Ha)']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

#Intitialize and fit the Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Coefficients and Intercept
model.intercept_,model.coef_

#Predict values for the test set
y_pred = model.predict(X_test)

#Display presdicts
y_pred

#Calculate mean squared error and R-squared
mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred)

#Plot actual vs predicted values
plt.scatter(X_test,y_test,color='blue',label='Actual')
plt.plot(X_test,y_pred,color='red',linewidth=2,label='Predicted')
plt.xlabel('Rainfall(mm)')
plt.ylabel('Crop Yield (Quintals/Ha)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
