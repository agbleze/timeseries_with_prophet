#%%
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

#%%
df = pd.read_csv('data/AirPassengers.csv')

#%%
df['Month'] = pd.to_datetime(df['Month'])
df.columns = ['ds', 'y']

#%%
df.head()

#%%
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)

#%%
future = model.make_future_dataframe(periods=365*5)
forecast = model.predict(future)

#%%
fig = model.plot(forecast)
plt.show()


# %% adjusting to forecast on monthly basis
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=12*5, freq='MS')
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()


#%% sub-daily data
data = pd.read_csv('data/divvy_hourly.csv')

df = pd.DataFrame({'ds': pd.to_datetime(data['date']),
                   'y': data['rides']}
                  )

#%%
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=365*24, freq='h')

forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()
fig2 = model.plot_components(forecast)
plt.show()


# %% using data with regular gaps
# suppose that data was collected only b/T 8am and 6pm

df = df[(df['ds'].dt.hour >= 8) & (df['ds'].dt.hour < 18)]

model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=365*24, freq='h')
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()

#%% zoom in on just 3 days
fig = model.plot(forecast)
plt.xlim(pd.to_datetime(['2018-08-01', '2018-08-04']))
plt.ylim(-2000, 4000)
plt.show()


# %%
future2 = future[(future['ds'].dt.hour >= 8) & 
                 (future['ds'].dt.hour < 18)
                 ]
forecast2 = model.predict(future2)
fig = model.plot(forecast2)
plt.show()

#%% ploting 3 days of forecast2
fig = model.plot(forecast2, figsize=(10, 4))
plt.xlim(['2018-08-01', '2018-08-04'])
plt.ylim(-2000, 4000)
plt.show()

#%%
from prophet.plot import plot_seasonality
plot_seasonality(model, 'daily', figsize=(10, 3))
plt.show()

#%%




# %%
