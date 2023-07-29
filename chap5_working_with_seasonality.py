#%%
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

df = pd.read_csv('data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.columns = ['ds', 'y']

#%% additive model
model_a = Prophet(seasonality_mode='additive', yearly_seasonality=4)
model_a.fit(df)
forecast_a = model_a.predict()
fig_a = model_a.plot(forecast_a)
plt.show()

#%% multiplicative 
model_m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=4)
model_m.fit(df)
forecast_m = model_m.predict()
fig_m = model_m.plot(forecast_m)

plt.show()

# %% plot components of models
fig_a2 = model_a.plot_components(forecast_a)
plt.show()

#%%
fig_m2 = model_m.plot_components(forecast_m)
plt.show()



#%% ####### working with divvy daily data
df = pd.read_csv('data/divvy_daily.csv')
df.head()

#%%
df = df[['date', 'rides']]
df['date'] = pd.to_datetime(df['date'])
df.columns = ['ds', 'y']

#%%
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()

# %% plot components
fig2 = model.plot_components(forecast)
plt.show()



#%% reducing fourier order by reducing yearly_seasonality to reduce overfitting
model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=4)
model.fit(df)

from prophet.plot import plot_yearly

fig3 = plot_yearly(model, figsize=(10.5, 3.25))
plt.show()




# %%
