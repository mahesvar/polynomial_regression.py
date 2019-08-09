## this polynomial regression model is used to predict the salary of a newly joined employee who has many years of experience from 
## his previous company. he says his salary was 160k in his previous company for the similar job in the new company, so he asks for 
## for similar or higher salary for the same position in the new company
## the HR is not sure about the salary for his level of experience (say 6.5 level), so he biult a model to predict the salary check 
## from the data collected in his company 
## salary data is given in the file position_salaries

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

# predicting a new result with linear regression

lireg.predict(6.5)

# predicting a new result with polynomial regression
lireg_2.predict(polyreg.fit_transform(6.5))
