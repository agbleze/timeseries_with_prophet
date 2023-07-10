#%%
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

df = pd.read_csv('data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.columns = ['ds', 'y']

#%% additive model
model_a = Prophet(seasonality_mode='additive', yearly_seasonaliyt=4)
model_a.fit(df)
forecast_a = model_a.predict()
fig_a = model_a.plot(forecast_a)
plt.show()





# %%
