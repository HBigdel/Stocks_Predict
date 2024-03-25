import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import style
import yfinance as yf
import pandas_ta as ta
from ta import momentum
from scipy.stats import norm
from scipy.optimize import minimize
style.use("ggplot")

#Determine the stock and confidence level
stock="Tesla"
c=0.95
ema_period=200
rsi_period=14
ma_period=50



#determine start and end
'''
start=dt.datetime.strptime("2013-01-01","%Y-%m-%d")
end=dt.datetime.strptime("2023-06-10","%Y-%m-%d")
num_days=(end-start).days
num_weeks=num_days//7
print(f"Number of days={num_days} and number of weeks={num_weeks}")

#download the data

df=yf.download(stock,start,end)
df.to_csv("Tesla_data"+".csv")
'''

#read the data
df=pd.read_csv("TSLA_data_new.csv")
df.reset_index(drop=True, inplace=True)
df_SP=pd.read_csv("GSPC_new.csv")
df_SP.reset_index(drop=True, inplace=True)



#slice the data based on given time period
df["Date"]=pd.to_datetime(df["Date"])
df_SP["Date"]=pd.to_datetime(df_SP["Date"])
From=pd.to_datetime("2011-01-01") #this can be start
To=pd.to_datetime("2023-07-06") #this cand be end
num_days=(To-From).days
num_weeks=num_days//7
sliced_df=df[(df["Date"]>=From) & (df["Date"]<=To)]
sliced_df_sp=df_SP[(df_SP["Date"]>=From) & (df_SP["Date"]<=To)]
sliced_df_sp=sliced_df_sp.reset_index(drop=True)
sliced_df=sliced_df.reset_index(drop=True)
df=sliced_df
df_SP=sliced_df_sp

print(f"Tesla data:\n{df}")
print(f"Tesla data:\n{df_SP}")

'''
columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
'''


#checking rows with missing data
columns_with_null=df.columns[df.isnull().any()]
print(f"Columns with null:\n{columns_with_null}")

#plot daily price with volume

fig,ax1=plt.subplots(figsize=(10,6))

ax1.plot(pd.to_datetime(df['Date']),df['Close'],color='blue')
ax1.set_ylabel("Daily price", color="blue")

ax2=ax1.twinx()
ax2.fill_between(pd.to_datetime(df['Date']),0,df["Volume"],alpha=0.7,color="gray")
ax2.set_ylabel("Volume",color="gray")

ax1.set_xlabel("Date")
ax1.set_title("Price chart with volume")

ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
for tick in ax1.get_xticklabels():
    tick.set_fontsize(4)
plt.show()

#plot log of adjusted close
df_log=np.log(1+df["Close"])

fig,ax1=plt.subplots(figsize=(10,6))

ax1.plot(pd.to_datetime(df['Date']),df_log,color='green')
ax1.set_ylabel("Log of daily price", color="green")

ax2=ax1.twinx()
ax2.fill_between(pd.to_datetime(df['Date']),0,df["Volume"],alpha=0.7,color="gray")
ax2.set_ylabel("Volume",color="gray")

ax1.set_xlabel("Date")
ax1.set_title("Log of price chart with volume")
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
for tick in ax1.get_xticklabels():
    tick.set_fontsize(4)
plt.show()

#plot daily returns distribution
return_daily=df["Close"].pct_change().dropna()
fig,ax=plt.subplots(1,1,figsize=(6,4))
ax.hist(return_daily,bins=100,color="orange",edgecolor="black")
plt.xlabel("Percentage of daily return")
plt.ylabel("Frequency")
plt.title(f"Dist. of % of daily returns for {round(num_days/255)} years",fontsize=12 )
plt.show()

#Calculating the descriptive stats, VaR and beta
series=pd.Series(return_daily)
statistics_return_daily=series.describe()
daily_volatility=statistics_return_daily["std"]
cov=np.cov(return_daily,df_SP["Close"].pct_change().dropna())[1,0]
beta=cov/np.var(df_SP["Close"].pct_change().dropna())
z_alpha=norm.ppf(c)
VaR=z_alpha*daily_volatility
data={f"{stock}": [beta,VaR]}
beta_VaR=pd.DataFrame(data,index=["Beta","VaR"])

print(f"The descriptive stats of daily return is \n{statistics_return_daily}")
print(f"The beta and VaR for {stock} is:\n {beta_VaR}")


#Correlation between return and volume
Cor_ret_vol=df["Volume"].corr(return_daily)
print(f"The correlation between the return and the volume is {Cor_ret_vol}")

###################################################################################
#Technicals
#Type two rsi (Simple RSI using SMA) 100-(100/(1+av gain/av loss))
'''
RSI measures the speed and change of price movements.
'''
def calculate_rsi(df,period):
    df["price change"]=df["Close"].diff().fillna(0)
    df["positive change"]=df["price change"].apply(lambda x:x if x>0 else 0)
    df["negative change"]=df["price change"].apply(lambda x:abs(x) if x<0 else 0)
    df["average gain"]=df["positive change"].rolling(window=period).mean()
    df["average loss"]=df["negative change"].rolling(window=period).mean()
    df["RS"]=df["average gain"]/df["average loss"]
    df["RSI"]=100-(100/(1+df["RS"]))
    rsi=df["RSI"]
    return rsi
RSI=calculate_rsi(df,rsi_period)
df["Rsi"]=RSI
df["Rsi"]=df["Rsi"].interpolate(method='linear', limit_direction='both')
print(f"The rsi is:\n{RSI[0:20]}")

#calculating SMA
df["Sma"]=df["Close"].rolling(window=ma_period).mean()

#Calculating EMA=Price(tod)×(2/1+period)+EMA(y)×(1−(2/1+period))
'''
It is a type of moving average that places more weight on recent data points, giving higher significance to the most recent values. 
'''
df['Ema']=df["Close"].ewm(span=ema_period,adjust=False).mean()
print(f"SMA IS \n {df['Sma']} and EMA is \n {df['Ema']}")

#calculating MACD=EMA(12)-EMA(26) and average of nine MACDs is MAND_S
'''
 Moving Average Convergence Divergence is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. 
 '''
df.ta.macd(close='Adj Close', fast=12, slow=26, signal=9, append=True)
MACD=df["MACD_12_26_9"]
MACD_S=df["MACDs_12_26_9"]
MACD_H=df["MACDh_12_26_9"]

#Calculating Stochastic Osilator= 100[(C - L14) / H14 – L14)]
'''
The Stochastic Oscillator compares the closing price of a security to its price range over a specific period of time.
'''
df["stoch_oscilator"]=momentum.stoch(high=df["High"],low=df["Low"],close=df["Close"])
stoch_oscilator=df["stoch_oscilator"]

#bolinger bands



#plot the price and RSI, EMA and MACD


fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(10,8),sharex=True, gridspec_kw={'height_ratios': [2,1,1, 1], 'hspace': 0.05})

ax1.plot(df["Date"],df['Adj Close'],color='blue')
ax1.set_ylabel("Daily price", color="blue")
ax1.set_title('Stock Price')
ax1.plot(df["Date"], df['Ema'], color='purple')  
ax1.legend(['Adj Close', f'{ema_period} days EMA'],loc='upper left', prop={'size': 6})

ax2.plot(df["Date"],RSI,color="red")
ax2.set_ylabel("RSI",color="red")
ax2.legend([f'{rsi_period} days RSI'],loc='upper left', prop={'size': 6})

ax3.plot(df["Date"],MACD,color="black")
ax3.plot(df["Date"],MACD_S,color="yellow")
ax3.fill_between(df["Date"], MACD_H, 0, where=MACD_H >= 0, color="green", alpha=0.3)
ax3.fill_between(df["Date"], MACD_H, 0, where=MACD_H < 0, color="red", alpha=0.3)
ax3.set_ylabel("MACD", color="black")
ax3.legend(["MACD", "MACD Signal"],loc='upper left', prop={'size': 6})

ax4.plot(df["Date"],stoch_oscilator,color="#105500")
ax4.set_ylabel("Stoch. Oscilator",color="#105500")
ax4.set_ylim(0, 100)
ax4.set_xlabel("Date")
ax4.legend("Stoch Oscilator",loc='upper left', prop={'size': 6})

ax1.set_xlim(df['Date'].min(), df['Date'].max())
ax2.set_xlim(df['Date'].min(), df['Date'].max())
ax2.set_ylim(0, 100)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
 
ax4.set_xticks(ax3.get_xticks()[::10]) 

ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
for tick in ax4.get_xticklabels():
    tick.set_fontsize(4)
ax2.axhline(30, linestyle='dotted', color='black')
ax2.axhline(70, linestyle='dotted', color='black') 
ax4.axhline(20, linestyle='dotted', color='black')
ax4.axhline(80, linestyle='dotted', color='black')    

plt.tight_layout()
plt.show()






###########################################################################################
#Training data
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,PowerTransformer
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dropout, Dense, Bidirectional
from keras.callbacks import History 
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import regularizers
import matplotlib.dates as mdates
import datetime

#Using Long Short Term Memory (LSTM)


'''
Tested with different scalers:
Generally, standard works better and is closer to true values. However, it has problem when there is a large move in a day price.

Give the size of training data
Decididng the feature
'''
n_days=60
epochs=80
target = "Close"
pct_of_training=0.85
features=["Close","Volume","Rsi","stoch_oscilator","MACD_12_26_9"]
sample=df[features].interpolate(method='linear', limit_direction='both')
#sample=df[features].dropna().reset_index(drop=True)

#Slicing the data to training, cross vaidation and test. 
sample_size=sample.shape[0]
train_size=int(pct_of_training*sample_size)
valid_size=int((sample_size-train_size)/2)
test_size=sample_size-train_size-valid_size
valid_size+=sample_size-train_size-valid_size-test_size
assert train_size + valid_size + test_size == sample_size
train_data=sample.loc[:train_size]
valid_data=sample.loc[train_size:(train_size+valid_size)]
test_data=sample.loc[(train_size+valid_size):(train_size+valid_size+test_size)]
print(f"training data is\n  {train_data}, cross validation data is \n{valid_data} and test data is \n{test_data}")

#rescaling the whole sample
sc=MinMaxScaler(feature_range=(-1,1))
#sc=StandardScaler()
#sc=PowerTransformer(method='yeo-johnson')

train_scaled=pd.DataFrame(sc.fit_transform(train_data),columns=features)
valid_scaled=pd.DataFrame(sc.transform(valid_data),columns=features)
test_scaled=pd.DataFrame(sc.transform(test_data),columns=features)
print(f"scaled training data is\n  {train_scaled}, scaled cross validation data is \n{valid_scaled} and scaled test data is \n{test_scaled}")
#Preparing a 3D format of the three data

#reshaping the training data
X_train=[]
y_train=[]
for i in range(n_days,train_size):
    X_train.append(train_scaled.iloc[i-n_days:i])
    y_train.append(train_scaled.iloc[i][target])
X_train,y_train=np.array(X_train),np.array(y_train)
print(f"X train  is :\n{X_train}")

#reshaping the validation data
X_val=[]
y_val=[]
#for i in range(train_size+n_days,train_size+valid_size):
for i in range(n_days,valid_size):    
    X_val.append(valid_scaled.iloc[i-n_days:i])
    y_val.append(valid_scaled.iloc[i][target])
X_val,y_val=np.array(X_val),np.array(y_val)
print(f"X validation is: \n{X_val}")
print(f"train size is {train_size}, valid size is {valid_size}, test size {test_size}, sample_size {sample_size}, addition {train_size+valid_size+test_size}")

#reshaping the test data
X_test=[]
y_test=[]
Dates=[]
#for i in range(sample_size-valid_size+n_days,sample_size):
for i in range(n_days,test_size):
    X_test.append(test_scaled.iloc[i-n_days:i])
    y_test.append(test_scaled.iloc[i][target])
    Dates.append(df.iloc[train_size+valid_size+i]["Date"].strftime("%Y-%m-%d"))
X_test,y_test=np.array(X_test),np.array(y_test)
formatted_dates = mdates.date2num(Dates)
print(f"X test data is :{X_test}")

#last 50 days data to predict tomorrow price
tomorrows_price_data=np.array([test_scaled.iloc[test_size-n_days+10:test_size+10]])
print(f"Tomorrow's price data is {tomorrows_price_data} and length of it is {len(tomorrows_price_data)}")


#train the data
model=Sequential()
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
model.add(Bidirectional(LSTM(units=60,return_sequences=True, input_shape=(X_train.shape[1], len(features)),kernel_regularizer=regularizers.l2(0.09))))
model.add(Dropout(0.20))
model.add(Bidirectional(LSTM(units=60,return_sequences=True)))
model.add(Dropout(0.20))
model.add(Bidirectional(LSTM(units=60,return_sequences=True)))
model.add(Dropout(0.20))
model.add(Bidirectional(LSTM(units=60)))
model.add(Dropout(0.20))
model.add(Dense(units=1))
#model.load_weights("LSTM_weights.h5")
model.compile(optimizer="adam", loss="mean_squared_error")
#this is only when we want to save the trained parameters
'''
checkpoint_filepath="LSTM_weights3.h5"
checkpoint=ModelCheckpoint(filepath=checkpoint_filepath,monitor="val_loss",save_best_only=True,mode="min",verbose=1)
'''
history=model.fit(X_train,y_train,epochs=epochs,batch_size=28,validation_data=(X_val,y_val),shuffle=True,callbacks=[early_stopping])#,callbacks=[checkpoint])

#plotting the val loss and train loss
train_loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs=range(1,len(train_loss)+1)
plt.figure(figsize=(8,6))
plt.plot(epochs,train_loss,color="b",label="Training loss")
plt.plot(epochs,val_loss,color="r", label="Validation loss")
plt.title("The training and validation loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Evaluate the model on test data

predictions=model.predict(X_test)
tomorrows_price_scaled=model.predict(tomorrows_price_data)
mse=mean_squared_error(y_test,predictions)
rmse = np.sqrt(mse)
mae=mean_absolute_error(y_test,predictions)
print(f"The square root of mean squared error is {rmse} and the mean absolute error is {mae}")

#plotting the true value vs predictions
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(formatted_dates,y_test,color="b",label="Actual")
ax.plot(formatted_dates,predictions,color="r", label="prediction")
ax.set_title("The prediction vs actual in Test")
ax.set_ylabel("scaled price value")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(axis='x', labelrotation=45)
plt.legend()
plt.show()

#Evaluating the accuracy score of prediction
#accuracy=accuracy_score(y_test,predictions)
performance_accuracy=(1-mse)*100

print(f"The accuracy percentage of this model with given parameters is {performance_accuracy}")


#Converting the true, prediction data and last 50 days data to its original scale
temp_df = pd.DataFrame(columns=features)
temp_df[target] = predictions.ravel()
predictions_original = sc.inverse_transform(temp_df)[:, features.index(target)]

temp1_df = pd.DataFrame(columns=features)
temp1_df[target] = y_test.ravel()
y_test_original = sc.inverse_transform(temp1_df)[:, features.index(target)]

temp_df = pd.DataFrame(columns=features)
temp_df[target] = tomorrows_price_scaled.ravel()
tomorrows_price = sc.inverse_transform(temp_df)[:, features.index(target)]


print(f"The predicted value(original scale) is {predictions_original} and the y_test value (original scale) is {y_test_original}")

#generating tomorrows date
Dates = [datetime.datetime.strptime(date_str, "%Y-%m-%d") for date_str in Dates]
tomorrow=Dates[-1]+datetime.timedelta(days=1)
num_tomorrow=mdates.date2num(tomorrow)

#plotting the true value vs predictions in original scale
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(formatted_dates,y_test_original,color="b",label="Actual")
ax.plot(formatted_dates,predictions_original,color="r", label="prediction")
ax.scatter(num_tomorrow,tomorrows_price,color="purple",label="Tomorrows price")
ax.set_title("The prediction vs actual in Test")
ax.set_ylabel("scaled price value")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(axis='x', labelrotation=45)
plt.legend()
plt.show()


mse_original=mean_squared_error(y_test_original,predictions_original)
rmse_original = np.sqrt(mse_original)
mae_original=mean_absolute_error(y_test_original,predictions_original)
print(f"The square root of mean squared error in original price is {rmse_original} and the mean absolute error in original price is {mae_original}")
print(f"Original y test values are {y_test_original}")
print(f"Based on this model , the price on {tomorrow} will be {tomorrows_price}")
#print(Dates)
model.summary()



















