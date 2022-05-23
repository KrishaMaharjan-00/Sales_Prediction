from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_data(train_data, scaler):
    scaler = scaler.fit(train_data)
    train_data = scaler.transform(scaler)
    return scaler, train_data


if __name__ == "__main__":
    train_data = pd.read_csv('../Data/X-train.csv')
    test_data = pd.read_csv('../Data/X-test.csv')
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler()
    min_scaler = scaler1.fit(train_data[['Sales']])
    min_train = min_scaler.transform(train_data[['Sales']])
    min_scaler1 = scaler1.fit(test_data[['Sales']])
    min_test = min_scaler1.transform(test_data[['Sales']])
    standard_scaler = scaler2.fit(train_data[['Sales']])
    standard_train = standard_scaler.transform(train_data[['Sales']])
    standard_scaler1 = scaler1.fit(test_data[['Sales']])
    standard_test = standard_scaler1.transform(test_data[['Sales']])
    print(min_train, standard_train)
    pd.DataFrame(min_train).to_csv("min_train.csv")
    pd.DataFrame(standard_train).to_csv("standard_train.csv")
    print('Testing')
    print(min_test, standard_test)
    pd.DataFrame(min_test).to_csv("min_test.csv")
    pd.DataFrame(standard_test).to_csv("standard_test.csv")

