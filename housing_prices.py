import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data=pd.read_csv('housing_data.csv')
pd.read_csv('housing_data.csv')
print(data.head())
x=data[['SquareFootage']]
y=data['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error:{mse}")
plt.scatter(X_test,y_test,color='blue',label='Actual Prices')
plt.plot(X_test,y_pred,color='red',linewidth=2,label='Predicted Prices')
plt.xlabel('Squared Footage')
plt.ylabel('Price')
plt.title('Square Footage vs Price')
plt.legend()
plt.show()