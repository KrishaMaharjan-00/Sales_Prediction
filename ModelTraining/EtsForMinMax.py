import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

df_train = pd.read_csv('../FeatureEngineering/min_train.csv')
df_test = pd.read_csv('../FeatureEngineering/min_test.csv')
df_train.index = pd.to_datetime(df_train.index)
model = ETSModel(df_train['0'].values)
fit = model.fit(df_test['0'])
print(fit.summary())
predict = model.predict(df_test['0'])
print(predict.summary())


