from matplotlib.pyplot import subplot
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pmdarima import auto_arima
from sklearn.ensemble import ExtraTreesRegressor
import pickle
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_components
import streamlit as st

def convert_month(month):
    if month == 3 or month == 4 or month == 5:
        return 0
    elif month == 6 or month == 7 or month == 8:
        return 1
    elif month == 9 or month == 10 or month == 11:
        return 2
    else:
        return 3


df = pd.read_csv('data/avocado.csv', index_col=0)

df_reg = df.copy()
df_reg['Date'] = pd.to_datetime(df_reg['Date'])
df_reg['day']=df_reg['Date'].dt.day
df_reg['month']=df_reg['Date'].dt.month
df_reg['Season'] = df_reg['month'].apply(lambda x: convert_month(x))

print('hello')
print(df_reg.head())