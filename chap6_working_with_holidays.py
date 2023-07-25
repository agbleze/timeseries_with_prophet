#%%
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

#%%

df = pd.read_csv("data/divvy_daily.csv")

df = df[['date','rides']]

df['date'] = pd.to_datetime(df['date'])

df.columns = ['ds','y']

#%%
model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=4)

model.add_country_holidays(country_name='US')

#%%
model.fit(df)

#%%

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()

#%%

fig2 = model.plot_components(forecast)
plt.show()

# %%
model.train_holiday_names



# %%
