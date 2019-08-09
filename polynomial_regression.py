import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("C:/Users/mahesvar/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression/Position_Salaries.csv")
x=dataset.iloc[:,1:2].values     # x is considered as matrix
y=dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression

lireg = LinearRegression()
lireg.fit(x,y)

# polynomila regression model

from sklearn.preprocessing import PolynomialFeatures

polyreg = PolynomialFeatures(degree = 4)
x_polyreg = polyreg.fit_transform(x)

lireg_2 = LinearRegression()
lireg_2.fit(x_polyreg,y) 

# simple linear regression
plt.scatter(x,y, color = 'red')
plt.scatter(x,lireg.predict(x), color= 'blue')
plt.title('truth or bluff(linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# polynomial regression
x_grid = np.arange(min(x),max(x),0.1)                     # x_grid for smoother curve
x_grid =x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid,lireg_2.predict(polyreg.fit_transform(x_grid)), color= 'green')   # relace y by x_polyreg value
plt.title('truth or bluff(polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.grid()
plt.show()
