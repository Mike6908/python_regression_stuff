
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cross_validation import train_test_split
#import pandas_datareader.data as web

#start = dt.datetime(1998, 1, 1)
#end = dt.datetime(2017, 12,31)
#data = web.DataReader('NVDA', 'yahoo', start, end)
#data.to_csv('~/Documents/NVDA.csv')
style.use('ggplot')
data = pd.read_csv('~/Documents/NVDA.csv')
print(data.head())
y = data.iloc[:,4:5]
y.plot()
#plt.show
y_12 = data.iloc[0:3248, 4:5]
y_14 = data.iloc[0:3760, 4:5]
x = pd.DataFrame(list(range(len(y_12))))
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.33, random_state=0)
#print(data.head())
#y.plot()
#print(data.describe())
#on to linear regression stuff
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x, y_12)
y_hat= pd.DataFrame(list(range(len(y_14))))
#define a function
def predF(df):
    for counter,x in enumerate(df):
        df[counter] = reg.intercept_ + reg.coef_ * counter
y_hat.apply(predF)
print(reg.intercept_)
print(reg.coef_)
plt.plot(y_14)
plt.plot(y_hat)
plt.show()
import statsmodels.api as sm

model = sm.OLS(y_14, y_hat).fit()
predict = model.predict(x)
print(model.summary())
