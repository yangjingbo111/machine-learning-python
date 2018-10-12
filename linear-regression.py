import quandl, math,datetime
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# load data
file_name = 'googl.csv'
df = 0  #init df variable
# check file existance, make sure downloading only once
exists = os.path.isfile(file_name)
if exists:
    print('load file from local csv file')
    df = pd.read_csv(file_name)
else:
    print('download file from quandl')
    df = quandl.get('WIKI/GOOGL')
    df.to_csv(file_name)
# end load data


price_date = df[['Date']]

# prepare data
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forcast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)
forcast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forcast_col].shift(-forcast_out)
price_date['Date'] = price_date['Date'].shift(-forcast_out)

# end prepare data

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X = X[:-forcast_out]
X_lately = X[-forcast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# linear regression algorithm
clf = LinearRegression()
clf.fit(X_train, y_train)

# svm
# clf = svm.SVR()
# clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
# print(accuracy)

# predict
forcast_set = clf.predict(X_lately)
# print(forcast_set,accuracy,forcast_out)

df['Forcast'] = np.nan
last_date = datetime.datetime.strptime(price_date.iloc[-forcast_out-1]['Date'], '%Y-%M-%d')
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    print(type(next_date.strftime("%Y-%m-%d")))

    next_unix += one_day
    df.loc[next_date.strftime("%Y-%m-%d")] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.head())
# print(df.tail())

df['Adj. Close'].plot()
df['Forcast'].plot()
# plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
