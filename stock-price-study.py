import pandas as pd
import yfinance as yfin
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date, timedelta
import numpy as np

df = pd.DataFrame()

start_date = '2019-01-01'
end_date = '2024-01-03'

acao = ['VALE3.SA']
acao_vale = yfin.Ticker("VALE3.SA")
df = yfin.download(acao, start=start_date, end=end_date)

# df = yfin.Ticker(acao).history(period='1y')

df.head(5)
