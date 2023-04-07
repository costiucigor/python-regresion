import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Lasso

train_df = pd.read_excel('price_lab3.xlsx')
test_df = pd.read_excel('predict_lab3.xlsx')

missing_values = train_df['price'].isnull()
median_price = train_df['price'].median()
train_df['price'].fillna(median_price, inplace=True)

missing_values = test_df['price'].isnull()
test_df['price'].fillna(median_price, inplace=True)

X_train = train_df[['area', 'rooms', 'floor']] 
y_train = train_df['price']
X_test = test_df[['area', 'rooms', 'floor']] 

linear_reg = LinearRegression()
lasso_reg = Lasso(alpha=0.1)
linear_reg.fit(X_train, y_train)
lasso_reg.fit(X_train, y_train)

test_df['linear_reg_price'] = linear_reg.predict(X_test)
test_df['lasso_reg_price'] = lasso_reg.predict(X_test)
test_df[['area', 'rooms', 'floor', 'linear_reg_price']].to_excel('predicted_values_ap1.xlsx', index=False)
test_df[['area', 'rooms', 'floor', 'lasso_reg_price']].to_excel('predicted_values_ap2.xlsx', index=False)

df = pd.read_excel('predict_lab3.xlsx')

plt.scatter(df.area, df.price, color = 'violet')
plt.title("Regression model for apartment")
plt.xlabel('area (m2)')
plt.ylabel('price (mln.lei)')
plt.plot(df.area, linear_reg.predict(df[['area']]), label="Ordinary Least Squares")
plt.show()