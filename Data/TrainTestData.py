from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib as plt
data = pd.read_csv('Sales.csv', index_col='Date')
print(data.head())
X_train = data[:-30]
X_test = data[-30:]
print(X_test)
X_train.to_csv('X-train.csv', index=False)
X_test.to_csv('X-test.csv', index=False)
X_train.count.plot(figsize=(15, 8), title='Sales', fontsize=14)
X_test.count.plot(figsize=(15, 8), title='Sales', fontsize=14)
plt.show()