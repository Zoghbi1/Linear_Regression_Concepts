import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, expon, poisson, uniform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

# Loading DataFrame
diabetes_df = pd.read_csv(
    r"C:\Users\zoghb\Desktop\Py Projekts\AI Engineer\1-Suppervised learning mit scikitlearn\diabetes_clean.csv"
)

print(diabetes_df.head())
# Taking Values as Numpy arrays , X as the Features, y as the Target
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
print(X)
X_bmi = X[:, 4]
print(X_bmi)
print(X_bmi.shape, y.shape)
X_bmi = X_bmi.reshape(-1, 1)
# plotting GLucose "as a function of bmi" Vs. BMI
plt.title("Blood Glucose as a function of BMI")
plt.scatter(X_bmi, y)
plt.ylabel("Blood Glucose levels (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()
# Fitting a regression model to our Data
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.title("Blood Glucose as a function of BMI")
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions, color="red")
plt.ylabel("Blood Glucose levels (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()
# We see , There are a weake to medium positive correlation
# Linear Regression using all Features :
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
# R2
r_squared = reg_all.score(X_test, y_test)
# MSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# The Model has an average error for blood glucose levels of arround 26 mg/dl
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

# Cross-Validation
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cross_validation_scores = cross_val_score(reg, X, y, cv=kf)
# Score of Linear Regression is R°2
print(cross_validation_scores)
# Ridge Regression : We will explain what alpha () are, and how it controls model complexity
scores = []
for alpha in [0.1, 1, 10, 100, 1000]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))
print(scores)
# We saw that ,when alpha increase we will have underfitting
# Lasso Regression is just like Ridge Regression but we use it because it can select important features from our Data
# We we need to know Features Importance we use the entire data set(We don't split it)

X_lasso = diabetes_df.drop("glucose", axis=1).values
y_lasso = diabetes_df["glucose"].values
# Access Features' Names
names = diabetes_df.drop("glucose", axis=1).columns
lasso = Lasso(alpha=0.1)
# We fit the model to the Data and extract coefficients with coef_ attribute
lasso_coefficients = lasso.fit(X_lasso, y_lasso).coef_
plt.bar(names, lasso_coefficients)
plt.xticks(rotation=45)
plt.show()
# and that's  not surprising :p  but that technique allows as to comunicate results to not-tech-audiences ..
