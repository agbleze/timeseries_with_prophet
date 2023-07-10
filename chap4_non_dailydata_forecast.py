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









