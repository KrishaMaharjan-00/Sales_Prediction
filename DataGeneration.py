import pandas as pd
import numpy as np
df = pd.DataFrame()
df['Date'] = pd.date_range(start='01/01/2019', end='12/30/2021')
df['Sales'] = np.random.randint(50000, 100000, size=(len(df['Date']), 1))
print(df['Date'])
print(df['Sales'])
df.to_csv('D:/PycharmProjects/Sales_Prediction/Data/Sales.csv', index=False)