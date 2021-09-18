import pandas as pd
import quandl

df = quandl.get("CHRIS/MGEX_IH1")
print(df.head())
df=df[ ['Open','High','Low','Last','Volume'] ]
df['hl_pct']=(df['High']-df['Last'])/df['Last'] * 100.0    #high-low percentage
df['pct_change']= (df['Last']-df['Open'])/df['Open'] * 100.0 #percentage change

df= df[['Last','hl_pct','pct_change','Volume']]
print(df)
