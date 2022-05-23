from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
series = read_csv('../FeatureEngineering/min_train.csv', header=0, index_col=0)
plot_acf(series)
plot_pacf(series, lags=50)
pyplot.show()