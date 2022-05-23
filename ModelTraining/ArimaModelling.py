import pandas as pd
# from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")
df = pd.read_csv('../FeatureEngineering/min_train.csv')
#                  )
# df.plot()
# arima_model = ARIMA(df, order=(1, 1, 2))
# model = arima_model.fit()
# plt.show()
# print(model.summary())


import statsmodels.api as sm

models = sm.tsa.arima.ARIMA(df['0'].values,order=(1,1,2))
res = models.fit()
print(res.summary())