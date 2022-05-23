import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.api as sm
df_mintrain = pd.read_csv('../FeatureEngineering/min_train.csv')
df_mintest = pd.read_csv('../FeatureEngineering/min_test.csv')
df_mintrain.index = pd.to_datetime(df_mintrain.index)
minmodel = ETSModel(df_mintrain['0'].values)
fit1 = minmodel.fit(df_mintest['0'])
print(fit1.summary())
print('*******************************************************************************')
df_standardtrain = pd.read_csv('../FeatureEngineering/standard_train.csv')
df_standardtest = pd.read_csv('../FeatureEngineering/standard_test.csv')
df_standardtrain.index = pd.to_datetime(df_standardtrain.index)
standardmodel = ETSModel(df_standardtrain['0'].values)
fit2 = standardmodel.fit(df_standardtest['0'])
print(fit2.summary())
# predict = model.predict(df_mintest['0'])
# print(predict.summary())
# final = seasonal_decompose(df['0'], model='additive')
# final.trend.plot()
# final.seasonal.plot()
# fig = final.plot()
