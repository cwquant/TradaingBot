# Collin Wendel Cryptocurrency Trading AI

"""Import Dependencies"""

import datetime
import math
import time
import warnings
from datetime import datetime, timedelta
import re
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pdpipe as pdp
import plotly.graph_objects as go
import pyEX as ex
import yfinance as yf

from dateutil.relativedelta import relativedelta
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.linalg import eig
from pycoingecko import CoinGeckoAPI
from scipy import stats
from scipy.special import comb
from statsmodels.distributions.empirical_distribution import ECDF
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

#Ignore RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Current Time
currenttime = int(time.time())


T = 360
simulations = 10
ticker = "SPY"

start_date = '2000-01-01'

# Get current date
today = datetime.today()    
today = today.strftime('%Y-%m-%d')

fed_api = 'insert federal reserve api key here'




class Security():
    """ """

    def __init__(self, ticker=None, start_date='2000-01-01',T = 7, sims = 10,predict=True,backtest=True,x_n_ecdfmax = .995,x_n_ecdfmin = .015,y_n_ecdfmax = .99,y_n_ecdfmin = .02, margin = True):


        start1 = time.time()

        tpkey = "Paste or retrieve key here"
        iexcli = ex.Client(api_token= tpkey, version='sandbox')
        self.iexcli = iexcli
        

        
        self.ticker = ticker

        try:
            key_stats_df = self.get_keystats()
            stock_name = key_stats_df["Company"].iloc[-1]
            self.name = stock_name + f" ({self.ticker})"
        except Exception as err:
            self.name = self.ticker
        
        self.start_date = start_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.T = T
        self.sims = sims
        self.predict = predict
        self.backtest = backtest
        self.x_n_ecdfmax = x_n_ecdfmax
        self.x_n_ecdfmin = x_n_ecdfmin
        self.y_n_ecdfmax = y_n_ecdfmax
        self.y_n_ecdfmin = y_n_ecdfmin
        self.margin = margin
        self.print = True

        # Get Stock Data
        stock_df = self.get_stock_data(self.ticker,self.start_date)

        # Reset to numerical indexF
        stock_df = stock_df.reset_index()

        # Get Returns and Acceleration
        stock_df = self.get_returns_and_acceleration(stock_df)

        # Get Teeth
        stock_df = self.get_teeth(stock_df)

        self.stock_df = stock_df
        try:
            rf_df = pd.read_csv("stockdata/" +"Risk_Free_Rate" + '_' + today+'.csv',index_col = 0, parse_dates=['Date'])
        except FileNotFoundError:
            rf_df = self.get_stock_data('^TNX', '2021-01-01F')
            self.export_to_csv(rf_df, "Risk_Free_Rate" + '_' + today)
        
        self.rf_df = rf_df
        self.rf = self.rf_df['Price'].iloc[-1]/100
        
        # Get Prediction
        self.stock_df = self.get_probablistic_predictions_mu_and_sigma(self.stock_df)

        # Ito test
        self.ito_df = self.create_ito_df(self.stock_df)

        self.std = self.ito_df['Returns'].std()
        self.starting_cash = 10000
        end1 = time.time()
        print(f"Done, Stock Runtime = {end1 - start1:.2f} seconds\n")
        

    def __repr__(self):
        
        s = f"\n{self.name} From {self.start_date} to {self.end_date}\nVariables:"
        keys = list(self.__dict__.keys())
        keys = np.sort(keys)

        for i in np.arange(len(keys)):
            s += f"\n {keys[i]}"

        return s


    def get_stock_data(self,ticker,start_date):

        """Takes stock ticker as input and returns a pandas dataframe with stock history using yfinance
        inputs:

        :param ticker: 
        :param start_date: 

        """
        
        stock = yf.Ticker(ticker)
        stock_history = stock.history(interval='1d', start=start_date, end=self.end_date)
        stock_df = pd.DataFrame(stock_history)
        stock_df = stock_df.drop(['Dividends', 'Stock Splits'], axis=1)

        stock_df.columns = ['Open','High','Low','Price', 'Volume']

        return stock_df


    def get_option_data(self):
        """Takes stock ticker as input and returns a pandas dataframe with options history using yfinance"""
        ticker = self.ticker
        stock = yf.Ticker(ticker)
        try:
            exp_dates = stock.options
        except:
            print("Option Data Unavailable for asset")
            exp_dates = 0
        lcalls = []
        lputs = []

        if len(exp_dates) > 1:
            for i in np.arange(len(exp_dates)):
                date = exp_dates[i]
                data = stock.option_chain(date)
                calls = pd.DataFrame(data.calls)
                calls['Exp_Date'] = [date for i in np.arange(len(calls))]
                puts = pd.DataFrame(data.puts)
                puts['Exp_Date'] = [date for i in np.arange(len(puts))]
                calls.set_index(['Exp_Date','strike'],inplace=True)
                puts.set_index(['Exp_Date','strike'],inplace=True)
                #calls = calls.drop(columns=['strike'])
                #puts = puts.drop(columns=['strike'])
                lcalls.append(calls)
                lputs.append(puts)
                #print(calls)
                #print(puts)

                #df = pd.read_table(data)
                #print(data)
                #print(df)
        #print(exp_dates)
        lcalls = pd.concat(lcalls, axis=0)
        lputs = pd.concat(lputs, axis=0)
        self.calls = lcalls
        self.puts = lputs

    
    def get_returns_and_acceleration(self,df):
        """

        :param df: 

        """

        rows = df.shape[0] - 1
        for row in np.arange(rows):
            df['Returns'] = df['Price'].pct_change(1)
            df['Acceleration'] = df['Returns'].pct_change(1)
        return df

    
    def get_teeth(self,df, t=7):
        """

        :param df: 
        :param t:  (Default value = 7)

        """

        stock_df = df.copy()
        rows = stock_df.shape[0] - 1
        # Get Momentum
        stock_df['momentum'] = np.where(stock_df['Returns'] > 0, 1, -1)
        stock_df['momentum'] = stock_df['momentum'].rolling(window=50).mean()

        # Get Price Teeth and Return Teeth at previous price( p(-1)), p(-t), p(2 * -t)

        stock_df['Rt_1'] = stock_df['Returns'].shift(-t)
        stock_df['Rt_2'] = stock_df['Returns'].shift(-2*t)
        stock_df['Rt_0'] = stock_df['Returns'].shift(-1)

        # Drop nan values
        stock_df = stock_df.fillna(0)
        return stock_df

        
    def get_keystats(self):
        """ """
        
        
        stats_df = self.iexcli.advancedStatsDF(self.ticker)
        stats_df = stats_df[['companyName','beta','peRatio','debtToEquity','putCallRatio','sharesOutstanding','float','marketcap','enterpriseValue']] 
        
        stats_pipeline = pdp.ColRename({'companyName':'Company','beta':'Beta','peRatio':'PE_Ratio','debtToEquity':'Debt/Equity','putCallRatio':'Put/Call','sharesOutstanding':'Shares_Outstanding','float':'Float','marketcap':'MarketCap','enterpriseValue':'Enterprise_Value'})
        stats_df = stats_pipeline(stats_df)
        
        
        
        return stats_df

    
    def get_probablistic_predictions_mu_and_sigma(self,df):
        """

        :param df: 

        """

        stock_df = df

        # Reset index after dropping nan values
        stock_df = stock_df.reset_index()
        stock_df = stock_df.drop(['index'], axis=1)

        """ Probability distribution Predictions """


        # Return average using teeth
        def rt_avg(row): return (row['Rt_0'] + row['Rt_1'] + row['Rt_2'])/3

        # Add average functions to pipeline
        pipeline = pdp.ApplyToRows(rt_avg, 'Rt_avg')
        pipeline += pdp.ColDrop(['Rt_0', 'Rt_1', 'Rt_2'])

        # Apply Average Pipeline to Dataframe
        stock_df = pipeline(stock_df)

        # Fill nan values with 0
        stock_df = stock_df.fillna(0)



        for x in [3, 5, 7, 10, 20, 50]:
            """

            :param row): return (row['Rt_0'] + row['Rt_1'] + row['Rt_2'])/3# Add average functions to pipelinepipeline:  (Default value = pdp.ApplyToRows(rt_avg)
            :param 'Rt_avg')pipeline +:  (Default value = pdp.ColDrop(['Rt_0')
            :param 'Rt_1': 
            :param 'Rt_2'])# Apply Average Pipeline to Dataframestock_df:  (Default value = pipeline(stock_df)# Fill nan values with 0stock_df = stock_df.fillna(0)

            """
            if x != 7:
                stock_df[f'mavg_{x}'] = stock_df['Returns'].rolling(
                    window=x).mean()


        return stock_df

    
    def create_ito_df(self,df):
        """

        :param df: 

        """
        start1 = time.time()
        stock_df = df
        # Create new dataframe for easier comparison
        ito_df = pd.DataFrame(data=stock_df[['Date', 'Price', 'Volume','Returns', 'Acceleration', 'Rt_avg']])
        ito_df['Momentum'] = stock_df['mavg_5']
        ito_df['Acceleration'] = ito_df['Acceleration'].rolling(window = 5).mean()
        
        stock_df['RW_Price'] = stock_df['Price'] * stock_df['Volume']
        ito_df['VWAP'] = stock_df['RW_Price'].rolling(window = 180).sum() / stock_df['Volume'].rolling(window = 180).sum()
        ito_df['M_0'] = np.where(stock_df['Returns'] > 0, 1, -1)
        stock_df['VW_Returns'] = stock_df['Returns'] * stock_df['Volume']   
        ito_df.fillna(0)
    
        # Get desired columns
        vwar = lambda row: stock_df['VW_Returns'].iloc[:int(row.name)].sum() / stock_df['Volume'].iloc[:int(row.name)].sum()
 
        def d9r(row): return row['Rt_avg'] / ((1 - row['Returns']) * row['Returns'])  if ((1 - row['Returns']) * row['Returns']) > 0 else 1
        pipeline2 = pdp.ApplyToRows(vwar, 'C_VWAR')
        pipeline2 += pdp.ApplyToRows(d9r, 'L_EST')
        ito_df = pipeline2(ito_df)
 
        end1 = time.time()

        return ito_df

        
    def create_std_df(self):
        """

        :param row): return row['Rt_avg'] / ((1 - row['Returns']) * row['Returns'])  if ((1 - row['Returns']) * row['Returns']) > 0 else 1pipeline2:  (Default value = pdp.ApplyToRows(vwar)
        :param 'C_VWAR')pipeline2 +:  (Default value = pdp.ApplyToRows(d9r)
        :param 'L_EST')ito_df:  (Default value = pipeline2(ito_df)end1 = time.time()#print(f"Ito_df Runtime = {end1 - start1:.2f} seconds\n")return ito_dfcreate_std_df(self)

        """
        start1 = time.time()
        df = self.ito_df
        std_df = df[['Date', 'Price', 'Returns', 'Volume']].copy(deep=True)
        
        std_df['std_360'] = std_df['Returns'].rolling(window=360).std()
        std_df['std_180'] = std_df['Returns'].rolling(window=180).std()
        std_df['std_90'] = std_df['Returns'].rolling(window=90).std()
        std_df['std_60'] = std_df['Returns'].rolling(window=60).std()
        std_df['std_30'] = std_df['Returns'].rolling(window=30).std()

        def spread_360(row): return (abs(row['std_360'] - row['std_180']) + abs(row['std_360'] - row['std_90']) + abs(
            row['std_360'] - row['std_30']) + abs(row['std_360'] - row['std_60'])) / 4

        def spread_180(row): return (abs(row['std_180'] - row['std_360']) + abs(row['std_180'] - row['std_90']) + abs(
            row['std_180'] - row['std_30']) + abs(row['std_180'] - row['std_60'])) / 4

        def spread_90(row): return (abs(row['std_90'] - row['std_360']) + abs(row['std_90'] - row['std_180']) + abs(
            row['std_90'] - row['std_30']) + abs(row['std_90'] - row['std_60'])) / 4

        def spread_60(row): return (abs(row['std_60'] - row['std_360']) + abs(row['std_60'] - row['std_180']) + abs(
            row['std_60'] - row['std_30']) + abs(row['std_60'] - row['std_90'])) / 4

        def spread_30(row): return (abs(row['std_30'] - row['std_360']) + abs(row['std_30'] - row['std_180']) + abs(
            row['std_30'] - row['std_90']) + abs(row['std_30'] - row['std_60'])) / 4
        def avg(row): return (row['spread_360'] + row['spread_180'] +
                                row['spread_90'] + row['spread_60'] + row['spread_30']) / 5

        std_pipeline = pdp.ApplyToRows(spread_360, 'spread_360')
        std_pipeline += pdp.ApplyToRows(spread_180, 'spread_180')
        std_pipeline += pdp.ApplyToRows(spread_90, 'spread_90')
        std_pipeline += pdp.ApplyToRows(spread_60, 'spread_60')
        std_pipeline += pdp.ApplyToRows(spread_30, 'spread_30')
        std_pipeline += pdp.ApplyToRows(avg, 'spread_avg')
        std_df = std_pipeline(std_df)

        df['spread_avg'] = std_df['spread_avg']
        std_df['RC_Returns_90'] = std_df['Returns'].rolling(window=90).sum()
        df['RC_Returns_90'] = std_df['RC_Returns_90']

        # Get Accumulation/Distribution
        std_df['Low_P'] = std_df['Price'].rolling(window=14).min()
        std_df['High_P'] = std_df['Price'].rolling(window=14).max()
        std_df['A_D'] = (((std_df['Price'] - std_df['Low_P']) - (std_df['High_P'] - std_df['Price'])) / (std_df['High_P'] - std_df['Low_P'])) * std_df['Volume']
        
        # Get On-Balance volume
        std_df['OBV'] = [0 for x in np.arange(len(std_df))]
        for row in np.arange(len(std_df)):
            """

            :param row): return (abs(row['std_360'] - row['std_180']) + abs(row['std_360'] - row['std_90']) + abs(row['std_360'] - row['std_30']) + abs(row['std_360'] - row['std_60'])) / 4spread_180(row): return (abs(row['std_180'] - row['std_360']) + abs(row['std_180'] - row['std_90']) + abs(row['std_180'] - row['std_30']) + abs(row['std_180'] - row['std_60'])) / 4spread_90(row): return (abs(row['std_90'] - row['std_360']) + abs(row['std_90'] - row['std_180']) + abs(row['std_90'] - row['std_30']) + abs(row['std_90'] - row['std_60'])) / 4spread_60(row): return (abs(row['std_60'] - row['std_360']) + abs(row['std_60'] - row['std_180']) + abs(row['std_60'] - row['std_30']) + abs(row['std_60'] - row['std_90'])) / 4spread_30(row): return (abs(row['std_30'] - row['std_360']) + abs(row['std_30'] - row['std_180']) + abs(row['std_30'] - row['std_90']) + abs(row['std_30'] - row['std_60'])) / 4avg(row): return (row['spread_360'] + row['spread_180'] +row['spread_90'] + row['spread_60'] + row['spread_30']) / 5std_pipeline:  (Default value = pdp.ApplyToRows(spread_360)
            :param 'spread_360')std_pipeline +:  (Default value = pdp.ApplyToRows(spread_180)
            :param 'spread_180')std_pipeline +:  (Default value = pdp.ApplyToRows(spread_90)
            :param 'spread_90')std_pipeline +:  (Default value = pdp.ApplyToRows(spread_60)
            :param 'spread_60')std_pipeline +:  (Default value = pdp.ApplyToRows(spread_30)
            :param 'spread_30')std_pipeline +:  (Default value = pdp.ApplyToRows(avg)
            :param 'spread_avg')std_df:  (Default value = std_pipeline(std_df)df['spread_avg'] = std_df['spread_avg']std_df['RC_Returns_90'] = std_df['Returns'].rolling(window=90).sum()df['RC_Returns_90'] = std_df['RC_Returns_90']# Get Accumulation/Distributionstd_df['Low_P'] = std_df['Price'].rolling(window=14).min()std_df['High_P'] = std_df['Price'].rolling(window=14).max()std_df['A_D'] = (((std_df['Price'] - std_df['Low_P']) - (std_df['High_P'] - std_df['Price'])) / (std_df['High_P'] - std_df['Low_P'])) * std_df['Volume']# Get On-Balance volumestd_df['OBV'] = [0 for x in np.arange(len(std_df))]for row in np.arange(len(std_df))

            """
            if std_df['Price'].iloc[row] > std_df['Price'].iloc[row-1]:
                std_df['OBV'].iloc[row] = std_df['Volume'].iloc[row]
            if std_df['Price'].iloc[row] < std_df['Price'].iloc[row-1]:
                std_df['OBV'].iloc[row] = std_df['Volume'].iloc[row] * -1
            if std_df['Price'].iloc[row] == std_df['Price'].iloc[row-1]:
                std_df['OBV'].iloc[row] = 0
        
        std_df['OBV'] = std_df['OBV'].cumsum()
        std_df['OBV'] = np.where(std_df['OBV'] > 1000, std_df['OBV'],0)
        std_df['A_D'] = std_df['A_D'].cumsum()
        #print('std_df:\n',std_df)
        df['A_D'] = std_df['A_D']
        df['OBV'] = std_df['OBV']
        df['OBVAP'] = (df['OBV'] * df['Price'])/ df['Volume'].cumsum()
        df['OBVAR'] = (df['OBV'] * df['Returns'])/ df['Volume'].cumsum()
        df['C_OBVAR'] = df['OBVAR'].cumsum()
        end1 = time.time()
        #print(f"Done, Std_df Runtime = {end1 - start1:.2f} seconds\n")
        return std_df, df

 
    def export_to_csv(self,df, name):
        """

        :param df: 
        :param name: 

        """

        df_csv = df.to_csv(f'stockdata/{str(name)}.csv')

    
    def create_comparison_df(self):
        """ """

        stock_df = self.ito_df

        # Create new dataframe for easier comparison
        compare_df = pd.DataFrame(
            data=stock_df[['Date', 'Prediction', 'Price', 'Returns', 'Acceleration', 'Est']])
        

        mae = np.mean(np.absolute(stock_df['Est Difference']))
        # print(stock_df)
        #print('Comparison Dataframe:\n', compare_df)
        return compare_df, mae


    def get_trade_df(self):
        """ """
        stock_df = self.stock_df.copy()
        trade_df = pd.DataFrame(stock_df['Acceleration'])
        trade_df = trade_df.replace([np.inf, -np.inf], 0)
        trade_df['Price'] = stock_df['Price']
        trade_df['Date'] = stock_df['Date']
        trade_df['Returns'] = stock_df['Returns']
        trade_df['Rt_avg'] = stock_df['Rt_avg']
        trade_df['momentum'] = stock_df['mavg_5']
        trade_df = trade_df.replace([np.inf, -np.inf], 0)
        trade_df = trade_df.fillna(0)
        self.trade_df = trade_df
        return trade_df

  

    def update_daterange(self, start_date):
        """

        :param start_date: 

        """
        self.start_date = start_date
        try:
            stock_df = pd.read_csv( "stockdata/" + self.ticker + '_' + today+'.csv',index_col = 0, parse_dates=['Date'])
            stock_df = stock_df.loc[stock_df['Date'] >= self.start_date,:]
            self.csv = True
        except FileNotFoundError:
            
            print(f"Fetching {self.ticker} data...\n")
            # Get Stock Data
            stock_df = self.get_stock_data(self.ticker,self.start_date)
            
            # Reset to numerical index
            stock_df = stock_df.reset_index()
            
            # Get Returns and Acceleration
            stock_df = self.get_returns_and_acceleration(stock_df)

            # Get Teeth
            stock_df = self.get_teeth(stock_df)

            #Export to csv
            self.export_to_csv(stock_df, ticker + '_' + today)
            print('Done\n')
            self.csv = True
        
        self.stock_df = stock_df
        # Get Prediction
        self.stock_df = self.get_probablistic_predictions_mu_and_sigma(self.stock_df)

        # Ito test
        self.ito_df   = self.create_ito_df(self.stock_df)

        self.std_df, self.ito_df = self.create_std_df()
        self.trade_df = self.get_trade_df()


    def predict_future_prices(self,df, T, sims, predict, backtest, rf,test=False):
        """
        This one's up to you guys :)
        :param self:
        :param df:
        :param T:
        :param sims:
        :param predict:
        :param backtest:
        :param rf:
        :param test:
        :return:
        """
        return None


    def test_predictions(self):
        """
        This one's up to you guys :)
        :param self:
        :param df:
        :param T:
        :param sims:
        :param predict:
        :param backtest:
        :param rf:
        :param test:
        :return:
        """
        return None


    def graph_predictions(self, figsize = (15,8),show=True):
        """

        :param figsize:  (Default value = (15)
        :param 8): 
        :param show:  (Default value = True)

        """

        predictions_df = self.predictions_df
        sims = self.sims
        fig, ax = plt.subplots(figsize=figsize)
        x2 = list(predictions_df.index.values)
        predictions_df['M_0'] = np.where(predictions_df['Returns'] > 0, 1, -1)
        if sims != 1:
            for s in np.arange(sims):
                ax.plot(x2, predictions_df[f'Predictions_{s}'], color = 'cornflowerblue', alpha = .25)
            ax.plot(x2,predictions_df['mean'],color = 'red', linewidth = 1.75,label = 'Mean Prediction')
            ax.plot(x2,predictions_df['median'],color = 'cornflowerblue', linewidth = 1.75,label = 'Median')
        else:
            ax.plot(x2, predictions_df[f'Predictions_0'])
        ax.plot(x2,predictions_df['Price'], color = 'gold', linewidth = 1.75, label = f'{ticker} Price')
        ax.plot(x2,predictions_df['rmean'],color = 'firebrick', linewidth = 1.75,label = 'Mean Price')
        ax.plot(x2,predictions_df['rmedian'],color = 'cornflowerblue', linewidth = 1.75,label = 'median')
        
        
        formatter = mtick.FormatStrFormatter('$%1.2f')
        ax.yaxis.set_major_formatter(formatter)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_visible(True)
            tick.label2.set_visible(False)
        
        plt.xlabel('Time')
        plt.legend(framealpha=0.7)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        

        if show == True:
            plt.show()


    def graph_predictions_analysis(self, figsize = (15,8)):
        """

        :param figsize:  (Default value = (15)
        :param 8): 

        """
        
        #%matplotlib widget
        predictions_df,sims = self.predictions_df,self.sims
        ticker = self.ticker
        x = list(predictions_df.index.values)
        
        predictions_df = predictions_df.replace(0,np.nan)
        predictions_df['tmode'] = predictions_df['tmode'].replace(np.nan,0)
        predictions_df['rmedian'] = predictions_df['Price'].rolling(len(predictions_df),min_periods=2).median()
        predictions_df['rmean'] = predictions_df['Price'].rolling(len(predictions_df),min_periods=2).mean()
        
        
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(211)
        for s in np.arange(sims):
            ax1.plot(x, predictions_df[f'Predictions_{s}'], alpha = .2)
        ax1.plot(x,predictions_df['mean'],color = 'red', linewidth = 1.75,label = 'Mean')
        ax1.plot(x,predictions_df['median'],color = 'cornflowerblue', linewidth = 1.75,label = 'Median')
        ax1.plot(x,predictions_df['Price'], color = 'gold', linewidth = 1.75, label = 'Real Price')
        plt.legend()
        
        ax2 = plt.subplot(212)
        ax2.plot(x, predictions_df['tmode'], label = 'Mode',color='cornflowerblue')
        ax2.plot(x, predictions_df['indicator'], label = 'Indicator',color='r')
        ax2.plot(x, predictions_df['Indicator'], label = 'Indicator',color='Gold')
        
        ax2.axhline(y=0, color='white',alpha=.5)
        plt.legend()
        
        
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(511)
        for s in np.arange(sims):
            ax1.plot(x, predictions_df[f'Predictions_{s}'], alpha = .2)
        ax1.plot(x,predictions_df['mean'],color = 'red', linewidth = 1.75,label = 'Mean')
        ax1.plot(x,predictions_df['median'],color = 'cornflowerblue', linewidth = 1.75,label = 'Median')
        ax1.plot(x,predictions_df['Price'], color = 'gold', linewidth = 1.75, label = 'Real Price')
        plt.legend()
        
        ax2 = plt.subplot(514, sharex=ax1)
        ax2.plot(x,predictions_df['drift'], label = 'drift', color = 'cornflowerblue')
        ax2.plot(x,predictions_df['diffusion'], label = 'diffusion',color = 'lime')
        #ax2.plot(x, predictions_df['er'], label = 'er',color='cornflowerblue')
        #ax2.plot(x,df['Returns'], label = 'Actual Return', color = 'red')
        ax2.axhline(y=0, color='white',alpha=.5)
        plt.legend()
        
        
        
        ax4 = plt.subplot(513, sharex = ax1)
        ax4.plot(x, predictions_df['tmode'], label = 'Mode',color='cornflowerblue')
        ax4.plot(x, predictions_df['yn_indicator'], label = 'Y_n Indicator',color='r')
        ax4.plot(x, predictions_df['xn_indicator'], label = 'X_n Indicator',color='limegreen')
        ax4.axhline(y=0, color='white',alpha=.5)
        plt.legend()
        
        ax3 = plt.subplot(512, sharex=ax1)
        
        ax3.plot(x,predictions_df['gt'], label = 'gt')
        ax3.axhline(y=1, color='white',alpha=.5)
        plt.legend()
        
        
        ax5 = plt.subplot(515, sharex = ax1)
        ax5.plot(x,(1 + predictions_df['drift']).cumprod(), label = 'drift',color ='cornflowerblue')
        ax5.plot(x,(1 + predictions_df['diffusion']), label = 'diffusion', color = 'lime')
        ax5.plot(x,(1 + predictions_df['Rt_avg']).cumprod(), label = 'rt_avg')
        ax5.plot(x,(1 + predictions_df['l_dist']).cumprod(), label = 'l_dist')
        ax5.plot(x,(1 + predictions_df['rand']).cumprod(), label = 'rand')
        ax5.axhline(y=1, color='white',alpha=.5)
        plt.tight_layout()    
        plt.legend()





        #Plot Predictions
        plt.figure(figsize=figsize)
        ax6 = plt.subplot(511)
        
        predictions_df['M_0'] = np.where(predictions_df['Returns'] > 0, 1, -1)
        
        for s in np.arange(sims):
            ax6.plot(x, predictions_df[f'Predictions_{s}'], alpha = .2)
        
        ax6.plot(x,predictions_df['Price'], color = 'gold', linewidth = 1.75, label = 'Real Price')
        #ax6.plot(x,[price * (1+ ws[i]) for i in np.arange(len(ws))],label = 'W')
        ax6.plot(x,predictions_df['mean'],color = 'red', linewidth = 1.75,label = 'Mean Prediction')
        ax6.plot(x,predictions_df['median'],color = 'cornflowerblue', linewidth = 1.75,label = 'Median')
        
        

        graph_a = predictions_df['Var']
        
        
        
        ax7 = plt.subplot(512)
        ax7.plot(x,predictions_df['lambda'], label = 'lambda')
        plt.legend()
        
        
        ax8 = plt.subplot(513, sharex = ax6)
        ax8.plot(x,predictions_df['mu_r'], label = 'Mu R')
        plt.legend()

        

        
        ax9 = plt.subplot(515, sharex = ax6)
        ax9.plot(x,predictions_df['Y_N'], label = 'Y_n')
        ax9.plot(x,predictions_df['bayes'], label = 'bayes')
        ax9.axhline(y=.5, color='white',alpha=.5)
        ax9.axhline(y=0, color='white',alpha=.5)
        ax9.axhline(y=1, color='white',alpha=.5)
        plt.legend()

        ax10 = plt.subplot(514, sharex = ax6)
        
        ax10.plot(x,predictions_df['xn'], label = 'xn')
        plt.legend()
        plt.tight_layout()
        
        
        
        fig, ax = plt.subplots(figsize=figsize)
        x2 = list(predictions_df.index.values)
        predictions_df['M_0'] = np.where(predictions_df['Returns'] > 0, 1, -1)
        if sims != 1:
            for s in np.arange(sims):
                ax.plot(x2, predictions_df[f'Predictions_{s}'], color = 'cornflowerblue', alpha = .25)
            ax.plot(x2,predictions_df['mean'],color = 'red', linewidth = 1.75,label = 'Mean Prediction')
            ax.plot(x,predictions_df['median'],color = 'cornflowerblue', linewidth = 1.75,label = 'Median')
        else:
            ax.plot(x2, predictions_df[f'Predictions_0'])
        ax.plot(x2,predictions_df['Price'], color = 'gold', linewidth = 1.75, label = f'{ticker} Price')
        ax.plot(x2,predictions_df['rmean'],color = 'firebrick', linewidth = 1.75,label = 'Mean Price')
        ax.plot(x,predictions_df['rmedian'],color = 'cornflowerblue', linewidth = 1.75,label = 'median')
        
        
        formatter = mtick.FormatStrFormatter('$%1.2f')
        ax.yaxis.set_major_formatter(formatter)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_visible(True)
            tick.label2.set_visible(False)
        
        plt.xlabel('Time')
        plt.legend(framealpha=0.7)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        

        df = predictions_df
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(511)
        for s in np.arange(sims):
            ax1.plot(x, predictions_df[f'Predictions_{s}'], alpha = .2)
        ax1.plot(x,predictions_df['mean'],color = 'red', linewidth = 1.75,label = 'mean')
        ax1.plot(x,predictions_df['Price'], color = 'gold', linewidth = 1.75, label = 'Real Price')
        plt.legend()

        ax2 = plt.subplot(512, sharex=ax1)
        #ax2.plot(x, df['y_nr'], label = 'y_nr',color='paleturquoise')
        ax2.plot(x, df['y_nr_sma'], label = 'y_nr sma',color='darkturquoise')
        plt.legend()

        ax3 = plt.subplot(514, sharex=ax1)
        ax3.plot(x,df['y_necdf'], label='y_necdf',color='cornflowerblue')
        plt.legend()

        ax4 = plt.subplot(513, sharex=ax1)
        ax4.plot(x, df['x_nr_sma'], label = 'x_nr sma',color='seagreen')
        plt.legend()

        ax5 = plt.subplot(515, sharex=ax1)
        ax5.plot(x,df['x_necdf'], label='x_necdf',color='cornflowerblue')
        
        
        plt.legend()

        df = df.fillna(0)
        
        
        fig, (ax, ax1, ax2, ax3)  = plt.subplots(4, 1, sharex=True,figsize=figsize)

        ax.title.set_text(self.name)
        ax.plot(x,df['pxnr'] , label = 'x_necdf', color = 'cornflowerblue')
        
        ax.plot(x, df['xn_max'],label='xn_max', color='springgreen')
        ax.plot(x, df['xn_min'],label = 'xn_min', color='firebrick')
        ax.legend()
        
        ax1.plot(x,df['pynr'], label = 'y_necdf', color = 'cornflowerblue')
        ax1.plot(x, df['yn_max'],label='yn_max', color='springgreen')
        ax1.plot(x, df['yn_min'],label = 'yn_min', color='firebrick')
        ax1.legend()
        
        ax2.plot(x, df['upsum'], label = 'Up Sum', color='lime')
        ax2.legend()
        
        ax3.plot(x, df['downsum'], label = 'Down Sum', color='red')
        ax3.legend()
        

        

        
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(511)
        ax1.plot(x, df['div'],label='div', color='firebrick')
        plt.legend()

        ax2 = plt.subplot(512)
        ax2.plot(x, df['divt'],label='divt', color='springgreen')
        plt.legend()

        ax3 = plt.subplot(513)
        ax3.plot(x, df['divcov'],label='divcov', color='orange')
        plt.legend()

        ax4 = plt.subplot(514)
        ax4.plot(x, df['var'],label='var', color='cornflowerblue')
        plt.legend()

        ax5 = plt.subplot(515)
        ax5.plot(x, df['diff'],label='diff', color='darkturquoise')
        ax5.axhline(y=0, color='white',alpha=.5)
        plt.legend()

        plt.figure(figsize=figsize)
        ax1 = plt.subplot(111)
        ax1.plot(x, df['ltcm'],label='ltcm',color='cornflowerblue')
        ax1.plot(x, [np.sqrt(1/253) for x in np.arange(len(x))],label='1/253',color='white')
        
        plt.show()


    def analyze_probability(self, df=False):
        """

        :param df:  (Default value = False)

        """
        start1 = time.time()
        try:
            if df == False:
                df = self.ito_df
        except ValueError:
            df = df

        df['U'] = np.where(df['Returns'] > 0, 1, 0)
        df['t'] = [x for x in np.arange(len(df))]
        list_dict = {}
        lists = ['div', 'divt', 'divcov', 'var', 'max', 'min', 'diff', 'xn_max', 'xn_min', 'yn_max', 'yn_min', 'p',
                 'y_nr', 'y_nr_sma', 'y_nmin', 'y_necdf', 'y_n_ecdfmax', 'y_n_ecdfmin', 'x_nr', 'x_nr_sma', 'x_nmin',
                 'x_nmax', 'og_xnr', 'x_necdf', 'x_n_ecdfmax', 'x_n_ecdfmin', 'yn_indicator', 'xn_indicator',
                 'indicator', 'Indicator', 'upsum', 'downsum', 'mode', 'p_u', 'p_d', 'p_i','upper_bound','lower_bound']
        for l in lists:
            list_dict[l] = []

        x_nr = 0
        og_xnr = 0
        mode = 1
        tmode = mode
        T = 7
        x_nmax = 9
        x_nmin = .25
        y_nmin = .25
        x_nr_sma = 1
        y_nr_sma = .5
        x_nr_smamax = 4
        summax = 10

        up_sum = 0
        down_sum = 0
        rf = self.rf
        x_n_ecdfmax = self.x_n_ecdfmax
        x_n_ecdfmin = self.x_n_ecdfmin
        y_n_ecdfmax = self.y_n_ecdfmax
        y_n_ecdfmin = self.y_n_ecdfmin

        self.x_nmax = x_nmax
        self.x_nmin = x_nmin
        self.y_nmin = y_nmin
        self.x_nr_smamax = x_nr_smamax
        rf = self.rf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        returns = df['Returns'].to_numpy()
        prices = df['Price'].to_numpy()
        sigmas = df['Returns'].rolling(window=30).std()

        rcovs = []
        pcovs = []
        rpcovs = []
        rcov_eigvs, rcov_eigs = [], []
        pcov_eigvs, pcov_eigs = [], []
        rpcov_eigvs, rpcov_eigs = [], []
        rm = df[['Returns', 't']].to_numpy()
        pm = df[['Price', 't']].to_numpy()
        rpm = df[['Returns', 'Price']].to_numpy()

        for i in np.arange(len(df)):
            rmt = rm[:i]
            pmt = pm[:i]
            rpmt = rpm[:i]

            rmcov = np.cov(rmt, rowvar=False, bias=False)
            pmcov = np.cov(pmt, rowvar=False, bias=False)
            rpmcov = np.cov(rpmt, rowvar=False, bias=False)

            rcovs.append(rmcov)
            pcovs.append(pmcov)
            rpcovs.append(rpmcov)
            try:

                rcov_eigv, rcov_eig = np.linalg.eig(rmcov)
                pcov_eigv, pcov_eig = np.linalg.eig(pmcov)
                rpcov_eigv, rpcov_eig = np.linalg.eig(rpmcov)
            except:
                rcov_eigv, rcov_eig = 0, 0
                pcov_eigv, pcov_eig = 0, 0
                rpcov_eigv, rpcov_eig = 0, 0

            rcov_eigvs.append(rcov_eigv)
            rcov_eigs.append(rcov_eig)
            pcov_eigvs.append(pcov_eigv)
            pcov_eigs.append(pcov_eig)
            rpcov_eigvs.append(rpcov_eigv)
            rpcov_eigs.append(rpcov_eig)

        end2 = time.time()
        # print(f"Prob Check 1 Runtime = {end2 - start1:.2f} seconds\n")

        for i in np.arange(len(df)):
            xn_indicator = 0
            yn_indicator = 0

            mu = np.mean(returns[:i])
            e_r = returns[i]
            sigma = df['Returns'].iloc[:i].std()
            scale = np.sqrt(abs((mu - rf) / sigma) ** 2)
            sem = stats.sem(returns[:i])
            r = np.sum(np.log1p(returns[:i]))
            if i > 30:

                sigma = sigmas.iloc[i]

                rcov_eigv, rcov_eig = rcov_eigvs[i], rcov_eigs[i]
                pcov_eigv, pcov_eig = pcov_eigvs[i], pcov_eigs[i]
                rpcov_eigv, rpcov_eig = rpcov_eigvs[i], rpcov_eigs[i]

                scale = np.sqrt(abs((mu - rf) / sigma) ** 2)

                div = abs((mu - rf) / sigma)
                divt = (((1 + mu) ** 360 - 1) - rf) / sigma
                divcov = rpcov_eigv[1] / pcov_eigv[0]

                var = (np.sqrt(divt / rcov_eigv[1] * divcov / sigma) * pcov_eigv[0] / rpcov_eigv[1])
                var = var / np.sqrt(len(df) - i)
                var_norm = var / np.sqrt(len(df) - i)
                maximum = rpcov_eigv[0] / rcov_eigv[0]
                minimum = (var + ((1 - maximum) - var)) * np.sqrt(divcov) * np.sqrt(div)

                diff = max(((np.sqrt(divcov) - div)) / divt, scale * -1)

                xn_max = maximum - sigma * divcov  # * np.sqrt(div)
                xn_min = minimum + sigma * divcov  # * np.sqrt(div)
                yn_max = maximum - sigma * divcov  # * np.sqrt(div)
                yn_min = minimum + sigma * divcov * np.sqrt(div)
                xn_max = xn_max * max(.000000001, scale + var * div - sigma) + sigma ** 2
                xn_min = xn_min * min(max(.000000001, scale - var * div + sigma), 5)
                yn_max = yn_max * min(max(.000000001, scale + var * div - sigma), 5)
                yn_min = yn_min * min(max(0, scale - var * div + sigma), 5) - sigma

                list_dict['div'].append(div)
                list_dict['divt'].append(divt)
                list_dict['divcov'].append(divcov)
                list_dict['var'].append(var)
                list_dict['max'].append(maximum)
                list_dict['min'].append(minimum)
                list_dict['diff'].append(diff)
                list_dict['xn_max'].append(xn_max)
                list_dict['xn_min'].append(xn_min)
                list_dict['yn_max'].append(yn_max)
                list_dict['yn_min'].append(yn_min)
            else:
                list_dict['div'].append(0)
                list_dict['divt'].append(0)
                list_dict['divcov'].append(0)
                list_dict['var'].append(0)
                list_dict['max'].append(0)
                list_dict['min'].append(0)
                list_dict['diff'].append(0)
                list_dict['xn_max'].append(0)
                list_dict['xn_min'].append(0)
                list_dict['yn_max'].append(0)
                list_dict['yn_min'].append(0)

            # Number of Positive Returns
            k = df['U'].iloc[:i].sum()
            # Positive Return Probability
            p = k / (i + 1)
            # Probability that t+1 is same as t
            y_nr = min(((1 - p) / p) ** x_nr, 1)
            list_dict['y_nr'].append(y_nr)

            

            upper_bound = r + np.sqrt(sem) * max(x_nr, 1)
            lower_bound = r - np.sqrt(sem)

            p_tmp = p
            p_tmp -= sigma * x_nr
            q = 1 - p_tmp

            num = 1 - (q / p_tmp) ** abs(lower_bound)
            denom = 1 - (q / p_tmp) ** (upper_bound + abs(lower_bound))
            p_u = num / denom
            if p_u > 1:
                p_u = (p_u % 1) ** (1 / p_u)
            p_d = 1 - p_u

            list_dict['p_u'].append(p_u)
            list_dict['p_d'].append(p_d)
            list_dict['upper_bound'].append(upper_bound)
            list_dict['lower_bound'].append(lower_bound)
            if i > 30:
                x_n_ecdfmaxt = xn_max
                x_n_ecdfmint = xn_min
                y_n_ecdfmaxt = yn_max
                y_n_ecdfmint = yn_min
            else:
                x_n_ecdfmaxt = x_n_ecdfmax * scale
                x_n_ecdfmint = x_n_ecdfmin * scale
                y_n_ecdfmaxt = y_n_ecdfmax * scale
                y_n_ecdfmint = y_n_ecdfmin * scale

            list_dict['x_n_ecdfmax'].append(x_n_ecdfmaxt)
            list_dict['x_n_ecdfmin'].append(x_n_ecdfmint)
            list_dict['y_n_ecdfmax'].append(y_n_ecdfmaxt)
            list_dict['y_n_ecdfmin'].append(y_n_ecdfmint)

            # print(p)
            if e_r > 0:
                x_nr += 1
                og_xnr += 1
            if e_r < 0:
                x_nr += (-1 * x_nr) + (p ** x_nr * 2 * abs(e_r))
                og_xnr -= 1
            list_dict['x_nr'].append(x_nr)

            if summax < x_nr:
                summax = x_nr

            if i > 8:
                x_nr_sma = np.mean(list_dict['x_nr'][i - T:])
                y_nr_sma = np.mean(list_dict['y_nr'][i - T:])

            list_dict['x_nr_sma'].append(x_nr_sma)
            list_dict['y_nr_sma'].append(y_nr_sma)
            x_necdf = ECDF(list_dict['x_nr_sma'])
            y_necdf = ECDF(list_dict['y_nr_sma'])

            p_xnr = x_necdf(x_nr_sma) * scale
            p_ynr = y_necdf(y_nr_sma) * scale

            if p_xnr > x_n_ecdfmaxt:
                xn_indicator = -1.0
            elif p_xnr < x_n_ecdfmint:
                xn_indicator = 1.0
            else:
                xn_indicator = round(p_xnr - p_ynr, 8)
               

            if p_ynr > y_n_ecdfmaxt:
                yn_indicator = 1.0
            elif p_ynr < y_n_ecdfmint:
                yn_indicator = -1.0
            else:
                yn_indicator = round(p_ynr - p_xnr, 8)
                

            if p <= .49:
                yn_indicator = p_u - p_tmp
                xn_indicator = -p_d

            indicator = yn_indicator + xn_indicator
            list_dict['yn_indicator'].append(yn_indicator)
            list_dict['xn_indicator'].append(xn_indicator)

            if i > T:
                xn_indicator_sma = np.mean(list_dict['xn_indicator'][i - 3:])
                yn_indicator_sma = np.mean(list_dict['yn_indicator'][i - 3:])
                indicator = yn_indicator_sma + xn_indicator_sma

            if type(indicator) != float:
                indicator = float(indicator)
            
            p_i = (p_u - p_d)  # * sigma
            if i > T:
                p_i = np.mean(list_dict['p_u'][i - 5:]) - np.mean(list_dict['p_d'][i - 5:])
                p_i *= np.sqrt(sigma)
            p_i = max(min(p_i, 1), -1)
            if p < .5:
                p_i = p_i / np.sqrt(sigma)
            indicator += p_i

            list_dict['Indicator'].append(indicator)

            if i > T:
                indicator = np.mean(list_dict['Indicator'][i - (T - 1):])

            list_dict['indicator'].append(indicator)
            list_dict['p_i'].append(p_i)

            if indicator < 0:
                # down_sum += max(-1,min(1,abs(indicator)))
                down_sum += abs(indicator)
                # down_sum += 1
            elif indicator > 0:
                # up_sum += max(-1,min(1,indicator))
                up_sum += indicator
                # up_sum += 1

            if tmode == 1:
                # Continue going up
                if indicator > 0:
                    up_sum += 1
                    down_sum -= 1
                    tmode = 1
                # Not sure
                elif indicator == 0:
                    if up_sum > 1:
                        up_sum -= 1
                    elif up_sum <= 1:
                        up_sum = 0
                        tmode = 0
                # Switch
                elif indicator <= -1:
                    tmode = -1
                    up_sum -= 1
                elif indicator < 0 and up_sum == 0:
                    tmode = 0
                    up_sum = 0

            elif tmode == -1:
                # Continue going down
                if indicator < 0:
                    down_sum += 1
                    up_sum -= 1
                    tmode = -1
                # Not sure
                elif indicator == 0:
                    if down_sum > 1:
                        down_sum -= 1
                        tmode = 0
                    elif down_sum <= 1:
                        down_sum = 0
                        tmode = 0
                # Switch
                elif indicator >= 1:
                    tmode = 1
                    down_sum -= 1
                elif indicator > 0 and down_sum == 0:
                    tmode = 0
                    down_sum = 0

            elif tmode == 0:
                down_sum -= 1
                up_sum -= 1
                down_sum = min(max(0, down_sum), summax)
                up_sum = min(max(0, up_sum), summax)

                if down_sum == 0 and up_sum == 0:

                    # Switch Down
                    if indicator < -1:
                        tmode = -1
                    # Switch Up
                    elif indicator > 1:
                        tmode = 1
                    # Stay
                    elif indicator == 0:
                        tmode = 0
                elif down_sum > up_sum and up_sum > 1:
                    down_sum -= 1
                    # tmode = 1
                elif down_sum < up_sum and down_sum > 1:
                    up_sum -= 1
                    # tmode = -1

                if indicator > 1:
                    tmode = 1
                elif indicator <= -2:
                    tmode = -1

            down_sum -= 1
            up_sum -= 1
            down_sum = min(max(0, down_sum), summax)
            up_sum = min(max(0, up_sum), summax)

            mode = tmode

            list_dict['downsum'].append(down_sum)
            list_dict['upsum'].append(up_sum)
            list_dict['og_xnr'].append(og_xnr)
            list_dict['mode'].append(mode)
            list_dict['p'].append(p)

            list_dict['y_nmin'].append(y_nmin)
            list_dict['x_nmin'].append(x_nmin)
            list_dict['x_nmax'].append(x_nmax)
            list_dict['x_necdf'].append(p_xnr)
            list_dict['y_necdf'].append(p_ynr)
        end3 = time.time()
        

        for x in lists:
            col = list_dict[x]
            df[x] = col

        df['mode_0'] = np.where(df['mode'] == -1, df['Price'], np.nan)
        df['mode_1'] = np.where(df['mode'] == 1, df['Price'], np.nan)
        df['mode_neutral'] = np.where(df['mode'] == 0, df['Price'], np.nan)

        self.probability_analysis_df = df
        end1 = time.time()
        
        return df




    def graph_probability_analysis(self, figsize=(15, 8)):
        """

        :param figsize:  (Default value = (15)
        :param 8):

        """

        df = self.probability_analysis_df
        df['indicator'] = np.where(df['indicator'] < -2, -2, df['indicator'])
        x_nmax = self.x_nmax
        x_nmin = self.x_nmin
        y_nmin = self.y_nmin
        x_nr_smamax = self.x_nr_smamax
        pricemin = df['Price'].min() * .9
        df['indicator_0'] = np.where(df['indicator'] < 0, df['Price'], np.nan)
        df['indicator_1'] = np.where(df['indicator'] > 0, df['Price'], np.nan)

        plt.figure(figsize=figsize)
        x = df['Date']

        ax1 = plt.subplot(411)
        ax1.title.set_text(self.name)
        ax1.plot(x, df['Price'], color='gold')
        ax1.plot(x, df['mode_1'], label='Up', color='limegreen')
        ax1.fill_between(x, df['mode_1'], pricemin, color='limegreen', alpha=.2)
        ax1.fill_between(x, df['indicator_1'], pricemin, color='limegreen', alpha=.2)
        ax1.plot(x, df['mode_0'], label='Down', color='firebrick')
        ax1.fill_between(x, df['mode_0'], pricemin, color='firebrick', alpha=.2)
        ax1.fill_between(x, df['indicator_0'], pricemin, color='firebrick', alpha=.2)
        ax1.plot(x, df['mode_neutral'], label='Neutral', color='gold')
        plt.legend()

        ax2 = plt.subplot(412, sharex=ax1)
        color = 'cornflowerblue'

        ax2.plot(x, df['mode'], label='Mode', color='white')
        ax2.plot(x, df['indicator'], label='Indicator', color='gold')
        ax2.set_ylabel('Indicator', color="white", alpha=.75)
        ax2.axhline(y=0, color='white', alpha=.5)
        ax2.tick_params(axis='y', labelcolor="white")
        plt.legend()

        ax3 = plt.subplot(413, sharex=ax1)
        ax3.plot(x, df['xn_indicator'].rolling(window=7).mean(), label='X_n Indicator', color='lime')
        ax3.plot(x, df['yn_indicator'].rolling(window=7).mean(), label='Y_n Indicator', color=color)
        ax3.axhline(y=0, color='white', alpha=.5)
        ax3.axhline(y=1, color='springgreen', alpha=.5)
        ax3.axhline(y=-1, color='firebrick', alpha=.5)
        plt.legend()

        ax4 = plt.subplot(414, sharex=ax1)
        ax4.plot(x, df['p_i'], label='p_i', color='lime')
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=figsize)
        x = [x for x in np.arange(len(df))]

        ax1 = plt.subplot(511)
        ax1.title.set_text(self.name)
        ax1.plot(x, df['Price'], color='gold')
        ax1.plot(x, df['mode_1'], label='Up', color='limegreen')
        ax1.fill_between(x, df['mode_1'], pricemin, color='limegreen', alpha=.2)
        ax1.fill_between(x, df['indicator_1'], pricemin, color='limegreen', alpha=.2)
        ax1.plot(x, df['mode_0'], label='Down', color='firebrick')
        ax1.fill_between(x, df['mode_0'], pricemin, color='firebrick', alpha=.2)
        ax1.fill_between(x, df['indicator_0'], pricemin, color='firebrick', alpha=.2)
        ax1.plot(x, df['mode_neutral'], label='Neutral', color='gold')
        plt.legend()

        ax2 = plt.subplot(512, sharex=ax1)
        ax2.plot(x, df['y_nr_sma'], label='y_nr sma', color='darkturquoise')
        plt.legend()

        ax3 = plt.subplot(514, sharex=ax1)
        ax3.plot(x, df['y_necdf'], label='y_necdf', color='cornflowerblue')
        ax3.plot(x, df['y_n_ecdfmax'], label='y_n_ecdfmax', color='springgreen', ls='--')
        ax3.plot(x, df['y_n_ecdfmin'], label='y_n_ecdfmin', color='lightcoral', ls='--')
        plt.legend()

        ax4 = plt.subplot(513, sharex=ax1)
        ax4.plot(x, df['x_nr_sma'], label='x_nr sma', color='seagreen')
        plt.legend()

        ax5 = plt.subplot(515, sharex=ax1)
        ax5.plot(x, df['x_necdf'], label='x_necdf', color='cornflowerblue')
        ax5.plot(x, df['x_n_ecdfmax'], label='x_n_ecdfmax', color='springgreen', ls='--')
        ax5.plot(x, df['x_n_ecdfmin'], label='x_n_ecdfmin', color='lightcoral', ls='--')

        plt.tight_layout()
        plt.legend()

        df = df.fillna(0)
        x = df['Date']

        fig, (ax, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True, figsize=figsize)

        ax.title.set_text(self.name)
        ax.plot(x, df['x_necdf'], label='x_necdf', color='cornflowerblue')

        ax.plot(x, df['xn_max'], label='xn_max', color='springgreen')
        ax.plot(x, df['xn_min'], label='xn_min', color='firebrick')
        ax.legend()

        ax1.plot(x, df['y_necdf'], label='y_necdf', color='cornflowerblue')
        ax1.plot(x, df['yn_max'], label='xn_max', color='springgreen')
        ax1.plot(x, df['yn_min'], label='xn_min', color='firebrick')
        ax1.legend()

        ax2.plot(x, df['p_i'], label='p_i', color='cornflowerblue')
        ax2.legend()
        ax3.legend()
        plt.tight_layout()

        plt.figure(figsize=figsize)
        x = df['Date']

        ax = plt.subplot(211)
        ax.plot(x, df['Price'], label='Price', color='cornflowerblue')
        ax.legend()
        ax1 = plt.subplot(212, sharex=ax)
        ax1.plot(x, df['indicator'], label='indicator', color='gold')
        plt.legend()
        plt.tight_layout()
        plt.figure(figsize=figsize)
        x = df['Date']

        ax1 = plt.subplot(511)
        ax1.plot(x, df['div'], label='div', color='firebrick')
        plt.legend()

        ax2 = plt.subplot(512, sharex=ax1)
        ax2.plot(x, df['divt'], label='divt', color='springgreen')
        plt.legend()

        ax3 = plt.subplot(513, sharex=ax1)
        ax3.plot(x, df['divcov'], label='divcov', color='orange')
        plt.legend()

        ax4 = plt.subplot(514, sharex=ax1)
        ax4.plot(x, df['var'], label='var', color='cornflowerblue')
        plt.legend()

        ax5 = plt.subplot(515, sharex=ax1)
        ax5.plot(x, df['p_i'].rolling(window=7).mean(), label='diff', color='darkturquoise')
        ax5.axhline(y=0, color='white', alpha=.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.show()


    def trade(self, trading_fees=10):
        """
        This one's up to you guys :)
        :param self:
        :param df:
        :param T:
        :param sims:
        :param predict:
        :param backtest:
        :param rf:
        :param test:
        :return:
        """
        return None


    def calc_margin(self):
        """
        This one's up to you guys :)
        :param self:
        :param df:
        :param T:
        :param sims:
        :param predict:
        :param backtest:
        :param rf:
        :param test:
        :return:
        """
        return None


    

    def calc_price_volume(self):
        """ """
        
        df = self.ito_df[['Date','Price','Volume','Returns']]
        
        max = df['Price'].max()
        div = 10
        while max//div > 10:
            div += 1
        else:
            div = 1
        df['Price'] = round(df['Price'],-1 * div)
        ndf = pd.DataFrame(df['Date'])
        
        prices = df['Price']
        uprices = pd.unique(prices)

        for i in np.arange(len(uprices)):
            col = uprices[i]
            ndf[col] = [0 for x in np.arange(len(df))]

        for i in np.arange(len(df)):
            
            price = prices.iloc[i]
            volume = df['Volume'].iloc[i]
            r = df['Returns'].iloc[i]
            if r >= 0:
                ndf[price].iloc[i] = volume# * (.5 + r)#* (.5  * r/abs(r))
            else:
                ndf[price].iloc[i] = volume #* r/abs(r)
        #print(ndf)
        return ndf


    def graph_volume_profile(self,show=True):
        """

        :param show:  (Default value = True)

        """

        df = self.calc_price_volume()
        df.index = pd.to_datetime(df['Date'],format="%Y-%m-%d")
        df = df.cumsum()
        df = round(df,0)
        df = df.drop(columns=['Date'])
        df.replace(0,np.nan)
        
        x = []
        if len(df) > 365 * 1.5:
            denom = 365
        elif len(df) > 365//4:
            denom = 30
        else:
            denom = 1
        prices = round(self.ito_df['Price'],-1)
        print(len(df.columns)-1)
        for i in np.arange(len(df)):
            x_t = [i/denom]
            for ii in np.arange(len(df.columns)-1):
                if i != len(df):
                    x_t.append(i/denom)
                else:
                    x_t.append(np.nan)
            x.append(x_t)
        x = np.array(x)
        y = np.array([list(df.columns)[:] for i in np.arange(len(df))])
        z = np.array([df.iloc[i,:] for i in np.arange(len(df)) ])
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        fig.update_layout(scene = dict(
                    xaxis_title='Time',
                    yaxis_title='Price',
                    zaxis_title='Volume'),
                    )
        fig.update_layout(title=f'{self.ticker}', autosize=False,width=1500, height=750, margin=dict(l=65, r=50, b=65, t=90))
        camera = dict(eye=dict(x=0., y=2.5, z=0.))
        fig.update_layout(scene_camera=camera)
        if show == True:
            fig.show()
        fig.write_html(f"./3dmodels/{self.ticker}.html")
        return fig


    def analyze_outliers(self):
        """ just something I was messing around with"""

        df = self.ito_df
        df['Returns_Std'] = df['Returns'].expanding().std()
        df['two sigma'] = df['Returns_Std'] * 2
        df['outlier'] = [0 for x in np.arange(len(df))]
        df['outlier returns'] = [0 for x in np.arange(len(df))]

        count = 0
        for i in np.arange(len(df)):
            if abs(df['Returns'].iloc[i]) >= df['two sigma'].iloc[i]:
                df['outlier'].iloc[i] = -1 * count
                count = 0
                df['outlier returns'].iloc[i] = df['Returns'].iloc[i]
            else:
                df['outlier'].iloc[i] = 1
                count += 1

        plt.figure(figsize=(15,8))
        ax = plt.subplot(211)
        plt.title(f"{self.ticker}")
        x = df['Date']
        y = df['outlier'].cumsum()
        ax.plot(x,y,label='Outliers')
        plt.legend()

        ax1 = plt.subplot(212)
        y1 = np.log1p(df['outlier returns']).cumsum()
        ax1.plot(x, y1, label='Outlier Returns')
        ax1.axhline(y=0,ls='--',color='w')
        plt.legend()

        plt.show()

        return df

    def mean_analysis(self):
        """ Function I wrote quickly to test an idea I had"""
        bdf = self.ito_df
        bdf['Date'] = pd.to_datetime(bdf['Date'])

        bdf.index = bdf['Date']

        bdf = bdf.drop(columns=['Date'])

        rbdf = np.log1p(bdf[['Returns']]).groupby(pd.Grouper(freq="M")).sum()

        train_bdf = rbdf.copy().iloc[:len(rbdf) * 2 // 3]

        test_bdf = rbdf.copy().iloc[len(rbdf) * 2 // 3:]


        returns = rbdf['Returns'] + 1
        mean, geomean, hmean, median = np.mean(returns), stats.gmean(returns), stats.hmean(returns), np.median(returns)
        print('mean:\n', mean, '\ngeomean:\n', geomean, '\nhmean:\n', hmean, '\nmedian:\n', median)
        m_preds = []
        gm_preds = []
        med_preds = []
        hm_preds = []
        diff = len(rbdf) - len(train_bdf)
        for i in np.arange(diff):
            m_preds.append(mean)
            gm_preds.append(geomean)
            med_preds.append(median)
            hm_preds.append(hmean)


        value = bdf['Price'].loc[train_bdf.index.values[-1]]
        test_df = rbdf.copy().iloc[len(train_bdf):].dropna().to_numpy()
        test_df = pd.DataFrame(test_df, columns=['Returns'])
        tmp_df = pd.DataFrame(test_df, columns=['Returns'])

        test_df['mean'] = m_preds
        test_df['median'] = med_preds
        test_df['gmean'] = (gm_preds)
        test_df['hmean'] = (hm_preds)

        tmp_df['mean'] = m_preds
        tmp_df['median'] = med_preds
        tmp_df['gmean'] = (gm_preds)
        tmp_df['hmean'] = (hm_preds)

        test_df['median'] = test_df['median'] - 1
        test_df['mean'] = test_df['mean'] - 1
        test_df['gmean'] = test_df['gmean'] - 1
        test_df['hmean'] = abs(test_df['hmean'] - 1)


        test_df = test_df.cumsum()
        #print(test_df)
        test_df['median'] = np.cumsum((test_df['median']))
        test_df['mean'] = np.cumsum((test_df['mean']))
        test_df['gmean'] = np.cumsum((test_df['gmean']))
        test_df['hmean'] = np.cumsum((test_df['hmean']))
        test_df['h*g'] = (test_df['hmean'] * test_df['gmean'])
        test_df['h*med'] = (test_df['hmean'] * test_df['median'])
        test_df['gcombined'] = np.sqrt(test_df['h*med'] * test_df['h*g'])
        test_df['mcombined'] = (test_df['h*med'] + test_df['h*g']) / 2
        test_df['em+hm'] = (test_df['hmean'] + test_df['h*med']) / 2


        test_df = np.exp(test_df)
        test_df['emedian'] = np.log1p(np.cumsum(np.exp(tmp_df['median'] - 1 ))) #* np.exp(1)
        test_df['emean'] = np.log1p(np.cumsum(np.exp(tmp_df['mean'] - 1))) #* np.exp(1)
        test_df['egmean'] = np.log1p(np.cumsum(np.exp(tmp_df['gmean'] - 1))) #* np.exp(1)
        test_df['ehmean'] = np.log1p(np.cumsum(np.exp(tmp_df['hmean'] - 1))) #* np.exp(1)


        comp_df = test_df.drop(columns=['median', 'gmean', 'hmean','mean'])
        comp_df.plot(kind='line')
        (comp_df * value).plot(kind='line')
        plt.show()

class Portfolio():
    """ """

    def __init__(self, tickers, start_date='2000-01-01',T = 7, sims = 1,predict=False,backtest=False):
        start2 = time.time()
        self.tickers = tickers
        self.start_date = start_date
        self.T = T
        self.sims = sims
        self.predict = predict
        self.backtest = backtest
        stocks = []
        for i in np.arange(len(tickers)):
            print(f"Adding {tickers[i]} to Portfolio...")
            stock = Security(tickers[i], self.start_date, self.T, self.sims, self.predict, self.backtest)
            setattr(self, tickers[i], stock)
            stocks.append(stock)
        
        self.stocks = stocks
        end2 = time.time()

        print(f"Portfolio Complete, Runtime = {end2 - start2:.2f} seconds\n")
        

    def __repr__(self):
        return f"{self.start_date}\n{self.tickers}"

    def add_stock(self, ticker):
        """
        Add stock to portfolio
        :param ticker:

        """

        stock = Security(ticker, self.start_date, self.T, self.sims, self.predict, self.backtest)
        self.stocks.append(stock)
        self.tickers.append(ticker)


    def compare_trading(self,margin=True,fees=10):
        """
        Compare trading algo with other assets
        :param margin:  (Default value = True)
        :param fees:  (Default value = 10)

        """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            if margin == False:
                stock.margin = False
            else:
                stock.margin = True
            if fees == False:
                stock.trade(trading_fees=fees)
            else:
                stock.trade()
            


    def remove_stock(self,ticker):
        """

        :param ticker: 

        """
        tickers = self.tickers
        stocks = self.stocks
        for i in np.arange(len(tickers)):
            if tickers[i] == ticker:
                self.stocks.remove(tickers[i])


    def graph_trades(self):
        """ """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            stock.plot_full_trade_analysis()


    def plot_probability_analysis(self):
        """ """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            stock.graph_probability_analysis()


    def update_daterange(self, start_date):
        """

        :param start_date: 

        """
        stocks = self.stocks
        self.start_date = start_date
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            print(f"Updating {stock.ticker} Date Range...\n")
            stock.start_date = start_date
            try:
                stock_df = pd.read_csv("stockdata/" + stock.ticker + '_' + today+'.csv',index_col = 0, parse_dates=['Date'])
                stock_df = stock_df.loc[stock_df['Date'] >= stock.start_date,:]
 
            except FileNotFoundError:
                
                print(f"Fetching {stock.ticker} data...\n")
                # Get Stock Data
                stock_df = stock.get_stock_data(stock.ticker,stock.start_date)
                
                # Reset to numerical index
                stock_df = stock_df.reset_index()
                
                # Get Returns and Acceleration
                stock_df = stock.get_returns_and_acceleration(stock_df)

                # Get Teeth
                stock_df = stock.get_teeth(stock_df)

                #Export to csv
                stock.export_to_csv(stock_df, ticker + '_' + today)


                print('Done\n')

            
            stock.stock_df = stock_df
            # Get Prediction
            stock.stock_df = stock.get_probablistic_predictions_mu_and_sigma(stock.stock_df)

            # Ito test
            stock.ito_df, stock.est_return = stock.create_ito_df(stock.stock_df)

            stock.std_df, stock.ito_df = stock.create_std_df()
            print("Done\n")
        return self
   

    def copy(self):
        """ """
        return self
    

    def predict_future_prices(self,T, sims, predict, backtest,test=False):
        """

        :param T: 
        :param sims: 
        :param predict: 
        :param backtest: 
        :param test:  (Default value = False)

        """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            stock.predict_future_prices(stock.ito_df,T,sims,predict,backtest,stock.rf)

    
    def graph_predictions(self):
        """ """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            stock.graph_predictions(show=False)
        plt.show()
    

    def graph_return_analysis(self):
        """ """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            stock.graph_return_analysis()
        plt.show()


    def graph_volume_profile(self):
        """ """
        stocks = self.stocks
        graphs = []
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            graph = stock.graph_volume_profile(show=False)
            graphs.append(graph)
        
        for i in np.arange(len(graphs)):
            fig = graphs[i]
            
            fig.show()

    def analyze_outliers(self):
        """ """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            stock.analyze_outliers()

    def get_options_data(self):
        """ """
        stocks = self.stocks
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            stock.get_option_data()


    def test_correlations(self):
        """ """

        df = self.analyze_probability()
        df = self.stock_df
        print(df.columns.values)

        test_col = 'Returns'
        cols = ['Price', 'Returns', 'Acceleration', 'Rt_avg','Momentum',   'VWAR','C_VWAR',  'A Return Diff.', 'spread_avg', 'RC_Returns_90','A_D', 'OBV', 'OBVAP', 'OBVAR', 'C_OBVAR',  'div', 'divcov',   'y_nr', 'y_nr_sma', 'y_necdf', 'x_nr', 'x_nr_sma','og_xnr','x_necdf', 'yn_indicator', 'xn_indicator','indicator', 'Indicator', 'upsum', 'downsum', 'mode', 'mode_0','mode_1',test_col]
        cols = df.columns.values[1:]
        covs_list = {}
        t = 90
        for col in cols:
            covs_list[col] = {}
            for i in np.arange(-1*t,t+1):


                tmp_df = df.copy(deep=True)
                tmp_df[col] = df[col].shift(i)

                tmp_df = tmp_df[[test_col,col]]
                """
                tmp = tmp_df.to_numpy()
                cov = np.cov(tmp,rowvar=False,bias=False)
                """
                cov = tmp_df.corr()
                x = cov.iloc[0][0]
                y = cov.iloc[0][1]

                covs_list[col][i] = y

        dfs = []
        for i,col in enumerate(cols):
            df = pd.DataFrame.from_dict(covs_list[col],orient='index',columns=[col])
            dfs.append(df)

        maxs = {}
        mins = {}
        t_df = pd.concat(dfs,axis=1)
        print(t_df)
        l = len(cols)//4
        t_df.iloc[:,:l].plot(kind='line')
        t_df.iloc[:,l:l*2].plot(kind='line')
        t_df.iloc[:,l*2:l*3].plot(kind='line')
        t_df.iloc[:,l*3:].plot(kind='line')
        plt.show()
        return t_df

    # Portfolio Optimization Functions
    def get_portfolio_df(self):
        """ """

        stocks = self.stocks
        data = []
        for i in np.arange(len(stocks)):
            stock = stocks[i]
            ticker = stock.ticker
            tmp_df = stock.ito_df
            tmp_df.index = tmp_df['Date']
            tmp_df = tmp_df[['Price']]
            tmp_df = tmp_df.rename(columns={'Price':ticker + '_Price'})
            data.append(tmp_df)
        rf_df = stocks[0].rf_df
        rf_df['Rf_Price'] = rf_df['Price']/100
        rf_df = rf_df[['Rf_Price']]
        data.append(rf_df)


        df = pd.concat(data, axis=1, sort=True)
        self.portfolio_rf = df[['Rf_Price']]
        df = df.dropna()
        #df = df[:-1]
        print(df)
        self.portfolio_df = df


    def get_securitiess_in_portfolio(self,portfolio):
        """Takes a dataframe of a portfolio and returns a list of the stocks and/ or risk-free rate
        ASSUMPTION: uses the get_stock_data function above

        :param portfolio: 

        """

        stock = re.compile('([\w-]+)_Price')

        stocks = re.findall(stock, str(portfolio.columns.values))


        return stocks

    def rand_weights(self,n):
        """Produces n random weights that sum to 1

        :param n: 

        """
        k = np.random.uniform(0, 1, n)
        return k / np.sum(k)

    def get_returns(self,portfolio):
        """Takes a dataframe of a portfolio with stock and risk-free prices and returns dataframe with the excess returns

        :param portfolio: 

        """

        newportfolio = portfolio.copy(deep=True)
        stocks = self.get_securitiess_in_portfolio(portfolio)

        for i in range(len(stocks)):
            newportfolio[stocks[i] + '_Returns'] = portfolio[stocks[i] + '_Price'].astype(float).pct_change(1)
            newportfolio = newportfolio.drop([stocks[i] + '_Price'], axis=1)

        newportfolio = newportfolio.iloc[1:]
        return newportfolio

    def get_weights(self, portfolio, num_portfolios=100):
        """Takes a Dataframe of returns and creates an amount of portfolios equal to num_portfolios (default = 100) with random weighting

        :param portfolio: 
        :param num_portfolios:  (Default value = 100)

        """

        stocks = self.get_securitiess_in_portfolio(portfolio)
        stocks = [x + '_Weight' for x in stocks]
        num_stocks = len(stocks)
        newportfolio = pd.DataFrame(columns=stocks)

        for row in range(num_portfolios):
            weights = self.rand_weights(num_stocks)
            newportfolio.loc[row] = weights

        return newportfolio

    def get_risky_return_statistics(self, portfolio, portfoliodata, stocklist):
        """takes a dataframe of porfolios with weights and a dataframe with stock returns and calculates the portfolio returns and std for each portfolio

        :param portfolio: 
        :param portfoliodata: 
        :param stocklist: 

        """
        print('portfolio:\n',portfolio)
        print('portfoliodata\n',portfoliodata)
        stocks = stocklist
        weight_list = [x + '_Weight' for x in stocks]
        return_list = [x + '_Returns' for x in stocks]
        portfolio = portfolio[weight_list]
        portfoliodata = portfoliodata[return_list]
        e_r = []
        exreturns = []
        stdevs = []
        for i in range(len(stocks)):
            er = (portfoliodata[stocks[i] + '_Returns']).mean()
            print(er)
            e_r.append(er)
        #downside = pd.DataFrame(np.where(portfoliodata - self.rf_df < 0, portfoliodata,np.sqrt(portfoliodata)))
        cov = portfoliodata.cov()
        print(cov)
        returns = [[x] for x in e_r]
        er_df = pd.DataFrame(e_r)
        for i in range(portfolio.shape[0]):
            weights = portfolio.iloc[i].to_numpy()

            # Price Matrix
            p = np.asmatrix(returns)

            # Weights Matrix
            w = np.asmatrix(weights)

            # Covariance Matrix
            C = np.asmatrix(cov.to_numpy())

            # Filters out portfolios with outlier standard deviations and average returns
            mu = w * p
            sigma = w * (C * w.T)
            sigma = np.sqrt(sigma)

            if sigma >= 35:
                sigma = np.asmatrix([np.nan])

            if mu >= 50:
                mu = np.asmatrix([np.nan])
            if mu <= -10:
                mu = np.asmatrix([np.nan])

            exreturns.append(float(mu.item(0, 0)))
            stdevs.append(float(sigma.item(0, 0)))

        portfolio['Daily_Returns'] = exreturns
        portfolio['StDev'] = stdevs
        er_df = er_df.transpose()
        er_df.columns = stocks
        portfolio.dropna(inplace=True)
        print('rportfolio:\n',portfolio)
        return portfolio

    def get_sharpe_ratio(self, portfolio, prices, stock_list):
        """takes a portfolio and returns the a new column with the sharp ratio for each weighted portfolio

        :param portfolio: 
        :param prices: 
        :param stock_list: 

        """

        rf = prices['Rf_Price'].astype(float).iloc[-1] / 365
        for i in range(len(stock_list)):
            if stock_list[i] != 'Rf' and i < 2:
                E_r = portfolio['Daily_Returns']
                stdev = portfolio['StDev']
                sharp_ratio = (E_r - rf) / stdev
                portfolio['Sharpe_Ratio'] = sharp_ratio
        return portfolio

    def get_beginning_strategy_weights(self, portfolio, cutoff=90):
        """

        :param portfolio: 
        :param cutoff:  (Default value = 90)

        """



        # Get Price Data
        cportfolio_df = portfolio

        # Get Number of securities in portfolio
        c_num_stocks = len(self.get_securitiess_in_portfolio(cportfolio_df))

        weights = []

        # Assume at least 90 days of historical data
        rportfolio_data_df = cportfolio_df[:cutoff].copy(deep=True)

        # Remove risk-free asset
        rportfolio_data_df = rportfolio_data_df.drop(['Rf_Price'], axis=1)

        # Get List of Securities in Risky Portfolio
        stock_list = self.get_securitiess_in_portfolio(rportfolio_data_df)

        # Get Number of securities in Risky Portfolio
        r_num_stocks = len(stock_list)

        # Get Risky Portfolio returns
        rportfolio_returns_df = self.get_returns(rportfolio_data_df)

        # Get Portfolios and their respective weights
        risky_portfolios_df = self.get_weights(rportfolio_data_df, 100)

        # Get Expected Return and StDev for each portfolio
        self.get_risky_return_statistics(risky_portfolios_df, rportfolio_returns_df, stock_list)

        # Sort by Returns
        risky_portfolios_df.sort_values(by=['Daily_Returns'], inplace=True, ascending=False)

        # Get Sharpe Ratios for Risky Portfolios
        risky_portfolios_df = self.get_sharpe_ratio(risky_portfolios_df, cportfolio_df, stock_list)

        # Set inf values to zero
        risky_portfolios_df = risky_portfolios_df.replace([np.inf, -np.inf], 0.0)

        # Get the optimal tangent portfolio at max sharpe ratio
        optimal_portfolio_sharpe = risky_portfolios_df['Sharpe_Ratio'].max()

        # Sort Portfolios by Sharpe Ratio
        risky_portfolios_df.sort_values(by=['Sharpe_Ratio'], inplace=True, ascending=False)

        # Get the optimal portfolio to use as first day weights in backtest
        optimal_portfolio_weights = \
        risky_portfolios_df[[stock_list[x] + '_Weight' for x in range(len(stock_list))]].iloc[0]

        weights.append(optimal_portfolio_weights)
        weights_df = pd.concat(weights, axis=1, sort=True)
        print('Starting Weights:\n', weights_df.T)
        return weights_df.T

    def strategy_backtest(self, portfolio,cutoff=90, fees=0):
        """Takes a datafram with historical price data data and bactests trading strategy

        :param portfolio: 
        :param cutoff:  (Default value = 90)
        :param fees:  (Default value = 0)

        """
        df = portfolio
        # Get Price Data
        cportfolio_df = df
        print('cportfolio_df:\n',cportfolio_df)
        # Get Stock Returns
        cportfolio_returns_df = self.get_returns(cportfolio_df)

        # Get Number of securities in portfolio
        c_num_stocks = len(self.get_securitiess_in_portfolio(cportfolio_df))

        # Get List of Securities in Risky Portfolio
        stock_list = self.get_securitiess_in_portfolio(cportfolio_df)
        self.stock_list = stock_list

        # Create list for accumulation of dataframes with portfolio weights
        weights = []

        # Get beginning portfolio weights based on historical data and training cutoff
        beginning_weights = self.get_beginning_strategy_weights(cportfolio_df, cutoff)

        # Copy dataframe after cutoff day for testing
        test_df = cportfolio_df[cutoff:].copy(deep=True)

        # Add beginning weights
        weights.append(beginning_weights.T)



        # Loop over testing day
        for day in range(1, len(test_df)):
            # Set dataframe to only provide the previous day for testing in addition to the historical training data
            rportfolio_data_df = cportfolio_df[:cutoff + day].copy(deep=True)

            # Remove risk-free asset
            rportfolio_data_df = rportfolio_data_df.drop(['Rf_Price'], axis=1)

            # Print training day (used for debugging)
            #print('Test Day = ', day)

            # Get List of Securities in Risky Portfolio
            stock_list = self.get_securitiess_in_portfolio(rportfolio_data_df)

            # Get Number of securities in Risky Portfolio
            r_num_stocks = len(stock_list)

            # Get Risky Portfolio returns
            rportfolio_returns_df = self.get_returns(rportfolio_data_df)

            # Get Portfolios and their respective weights
            risky_portfolios_df = self.get_weights(rportfolio_data_df, 100)

            # Get Expected Return and StDev for each random portfolio
            self.get_risky_return_statistics(risky_portfolios_df, rportfolio_returns_df, stock_list)

            # Sort by Returns
            risky_portfolios_df.sort_values(by=['Daily_Returns'], inplace=True, ascending=False)

            # Get Sharpe Ratios for Random Risky Portfolios
            risky_portfolios_df = self.get_sharpe_ratio(risky_portfolios_df, cportfolio_df, stock_list)

            # Set inf values to zero
            risky_portfolios_df = risky_portfolios_df.replace([np.inf, -np.inf], 0.0)

            # Sort Portfolios by Sharpe Ratio
            risky_portfolios_df.sort_values(by=['Sharpe_Ratio'], inplace=True, ascending=False)

            # Get the optimal portfolio
            optimal_portfolio_weights = \
            risky_portfolios_df[[stock_list[x] + '_Weight' for x in range(len(stock_list))]].iloc[0]
            weights.append(optimal_portfolio_weights)
            opt_w = np.asmatrix(optimal_portfolio_weights.to_numpy())

        # Create dataframe of portfolio weights
        weights_df = pd.concat(weights, axis=1, sort=False)

        # Format dataframe
        weights_df = weights_df.T

        # Get Stock Returns for Portfolio Dataframe          #First Day of testing                       #Last Day of testing
        backtest_returns_df = self.get_returns(
            cportfolio_df[list(cportfolio_df[:cutoff].index.values)[-1]:list(test_df.index.values)[-1]].iloc[:, :-1])

        # Get Portfolio Returns
        portfolio_returns = self.get_risky_return_statistics(weights_df, backtest_returns_df, stock_list)

        # Set Dates to index
        portfolio_returns.index = list(test_df.index.values)

        # Remove Standard Deviation
        portfolio_returns = portfolio_returns.drop(['StDev'], axis=1)

        # Rename returns to Portfolio Returns for ID
        portfolio_returns = portfolio_returns.rename(columns={'Daily_Returns': 'Portfolio_Return'})

        # Get Portfolio Momentum for Signal Use
        portfolio_returns['Portfolio_Momentum'] = np.where(portfolio_returns['Portfolio_Return'] > 0, 1, -1)
        portfolio_returns['Portfolio_Momentum'] = portfolio_returns['Portfolio_Momentum'].rolling(window=30).mean()
        portfolio_returns['Portfolio_Momentum'] = portfolio_returns['Portfolio_Momentum'].pct_change(1)

        # Get Market Returns
        market_returns = Security('SPY',start_date=self.start_date).stock_df[['Date','Price']]
        market_returns = market_returns.rename(columns={'Price':'Market_Price'})
        market_returns.index = market_returns['Date']
        #market_returns = market_returns.rename(columns={'Adj Close': 'Market_Price'})
        market_returns = self.get_returns(market_returns)
        print('market_returns:\n',market_returns)
        # Get Market Momentum for Signal Use
        portfolio_returns['Market_Return'] = market_returns[
                                             list(cportfolio_df[:cutoff].index.values)[-1]:list(test_df.index.values)[
                                                 -1]].iloc[:, 1:].loc[:, 'Market_Returns']
        portfolio_returns['Market_Momentum'] = np.where(portfolio_returns['Market_Return'] > 0, 1, -1)
        portfolio_returns['Market_Momentum'] = portfolio_returns['Market_Momentum'].rolling(window=30).mean()
        portfolio_returns['Market_Momentum'] = portfolio_returns['Market_Momentum'].pct_change(1)

        # Create empty series for signal and strategy returns
        portfolio_returns['Signal'] = [0 for x in range(len(portfolio_returns))]
        portfolio_returns['Strategy_Return'] = [0 for x in range(len(portfolio_returns))]
        portfolio_value = [0 for x in range(len(portfolio_returns))]
        portfolio_value[0] = 10000
        portfolio_returns['Portfolio_Value'] = portfolio_value

        # Reset Index
        portfolio_returns = portfolio_returns.reset_index()

        # Create sell portfolio for market signal
        for row in range(1, len(portfolio_returns)):
            if portfolio_returns.loc[:, 'Market_Momentum'].iloc[row - 1] > \
                    portfolio_returns.loc[:, 'Portfolio_Momentum'].iloc[row - 1]:
                portfolio_returns.loc[row, 'Signal'] = 1
            else:
                portfolio_returns.loc[row, 'Signal'] = -1

        # Set Strategy Return to Market Return or Portfolio Return based on signal
        for row in range(len(portfolio_returns)):

            if portfolio_returns['Signal'].iloc[row] == 1:
                portfolio_returns.loc[row, 'Strategy_Return'] = portfolio_returns.loc[:, 'Portfolio_Return'].iloc[row]
            if portfolio_returns['Signal'].iloc[row] == 1:
                portfolio_returns.loc[row, 'Strategy_Return'] = portfolio_returns.loc[:, 'Market_Return'].iloc[row]

        for row in range(1, len(portfolio_returns)):
            if fees == 0:
                portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[row, 'Strategy_Return']) * \
                                                                portfolio_returns.loc[row - 1, 'Portfolio_Value']
            if fees == 1:
                if portfolio_returns.loc[row, 'Signal'] == -1:
                    portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[
                        row, 'Strategy_Return']) * portfolio_returns.loc[
                                                                        row - 1, 'Portfolio_Value'] - 20  # $10 for selling portfolio + $10 for buying market
                else:
                    portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[
                        row, 'Strategy_Return']) * portfolio_returns.loc[row - 1, 'Portfolio_Value']

        # Create Abnormal Returns Series
        portfolio_returns['Abnormal Returns'] = portfolio_returns['Strategy_Return'] - portfolio_returns[
            'Market_Return']

        if fees == 1:
            portfolio_returns['Strategy_Return'] = portfolio_returns['Portfolio_Value'].pct_change(1)
        self.portfolio_returns = portfolio_returns
        return portfolio_returns

    def portfolio_returns(self, portfolio,fees=0):
        """Takes a datafram with historical price data data and bactests trading strategy

        :param portfolio: 
        :param fees:  (Default value = 0)

        """
        try:
            self.get_portfolio_df()
        except:
            print(';)')
        df = portfolio
        cutoff = 90
        # Get Price Data
        cportfolio_df = df
        #print('cportfolio_df:\n', cportfolio_df)
        # Get Stock Returns
        cportfolio_returns_df = self.get_returns(cportfolio_df)

        # Get Number of securities in portfolio
        c_num_stocks = len(self.get_securitiess_in_portfolio(cportfolio_df))

        # Get List of Securities in Risky Portfolio
        stock_list = self.get_securitiess_in_portfolio(cportfolio_df)
        self.stock_list = stock_list

        # Create list for accumulation of dataframes with portfolio weights
        weights = []

        # Get beginning portfolio weights based on historical data and training cutoff
        beginning_weights = self.get_beginning_strategy_weights(cportfolio_df, cutoff)

        # Copy dataframe after cutoff day for testing
        test_df = cportfolio_df[cutoff:].copy(deep=True)

        # Add beginning weights
        weights.append(beginning_weights.T)

        # Loop over testing day
        for day in range(1, len(test_df)):
            weights.append(beginning_weights.T)


        # Create dataframe of portfolio weights
        weights_df = pd.concat(weights, axis=1, sort=False)

        # Format dataframe
        weights_df = weights_df.T
        print(len(cportfolio_df[list(cportfolio_df[:cutoff].index.values)[-1]:list(test_df.index.values)[-1]].iloc[:, :-1]))
        print(len(cportfolio_df.iloc[cutoff:cutoff+day]))
        # Get Stock Returns for Portfolio Dataframe          #First Day of testing                       #Last Day of testing
        backtest_returns_df = self.get_returns(
            cportfolio_df[list(cportfolio_df[:cutoff].index.values)[-1]:list(test_df.index.values)[-1]].iloc[:, :-1])

        # Get Portfolio Returns
        portfolio_returns = self.get_risky_return_statistics(weights_df, backtest_returns_df, stock_list[:-1])

        # Set Dates to index
        portfolio_returns.index = list(test_df.index.values)

        # Remove Standard Deviation
        portfolio_returns = portfolio_returns.drop(['StDev'], axis=1)

        # Rename returns to Portfolio Returns for ID
        portfolio_returns = portfolio_returns.rename(columns={'Daily_Returns': 'Portfolio_Return'})

        # Get Portfolio Momentum for Signal Use
        portfolio_returns['Portfolio_Momentum'] = np.where(portfolio_returns['Portfolio_Return'] > 0, 1, -1)
        portfolio_returns['Portfolio_Momentum'] = portfolio_returns['Portfolio_Momentum'].rolling(window=30).mean()
        portfolio_returns['Portfolio_Momentum'] = portfolio_returns['Portfolio_Momentum'].pct_change(1)

        # Get Market Returns
        market_returns = Security('SPY', start_date=self.start_date).stock_df[['Date', 'Price']]
        market_returns = market_returns.rename(columns={'Price': 'Market_Price'})
        market_returns.index = market_returns['Date']
        # market_returns = market_returns.rename(columns={'Adj Close': 'Market_Price'})
        market_returns = self.get_returns(market_returns)
        #print('market_returns:\n', market_returns)
        # Get Market Momentum for Signal Use
        portfolio_returns['Market_Return'] = market_returns[
                                             list(cportfolio_df[:cutoff].index.values)[-1]:list(test_df.index.values)[
                                                 -1]].iloc[:, 1:].loc[:, 'Market_Returns']
        portfolio_returns['Market_Momentum'] = np.where(portfolio_returns['Market_Return'] > 0, 1, -1)
        portfolio_returns['Market_Momentum'] = portfolio_returns['Market_Momentum'].rolling(window=30).mean()
        portfolio_returns['Market_Momentum'] = portfolio_returns['Market_Momentum'].pct_change(1)

        # Create empty series for signal and strategy returns
        portfolio_returns['Signal'] = [0 for x in range(len(portfolio_returns))]
        portfolio_returns['Strategy_Return'] = [0 for x in range(len(portfolio_returns))]
        portfolio_value = [0 for x in range(len(portfolio_returns))]
        portfolio_value[0] = 10000
        portfolio_returns['Portfolio_Value'] = portfolio_value

        # Reset Index
        portfolio_returns = portfolio_returns.reset_index()

        # Create sell portfolio for market signal
        for row in range(1, len(portfolio_returns)):
            if portfolio_returns.loc[:, 'Market_Momentum'].iloc[row - 1] > \
                    portfolio_returns.loc[:, 'Portfolio_Momentum'].iloc[row - 1]:
                portfolio_returns.loc[row, 'Signal'] = 1
            else:
                portfolio_returns.loc[row, 'Signal'] = -1

        # Set Strategy Return to Market Return or Portfolio Return based on signal
        for row in range(len(portfolio_returns)):

            if portfolio_returns['Signal'].iloc[row] == 1:
                portfolio_returns.loc[row, 'Strategy_Return'] = portfolio_returns.loc[:, 'Portfolio_Return'].iloc[row]
            if portfolio_returns['Signal'].iloc[row] == 1:
                portfolio_returns.loc[row, 'Strategy_Return'] = portfolio_returns.loc[:, 'Market_Return'].iloc[row]

        for row in range(1, len(portfolio_returns)):
            if fees == 0:
                portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[row, 'Strategy_Return']) * \
                                                                portfolio_returns.loc[row - 1, 'Portfolio_Value']
            if fees == 1:
                if portfolio_returns.loc[row, 'Signal'] == -1:
                    portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[
                        row, 'Strategy_Return']) * portfolio_returns.loc[
                                                                        row - 1, 'Portfolio_Value'] - 20  # $10 for selling portfolio + $10 for buying market
                else:
                    portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[
                        row, 'Strategy_Return']) * portfolio_returns.loc[row - 1, 'Portfolio_Value']

        # Create Abnormal Returns Series
        portfolio_returns['Abnormal Returns'] = portfolio_returns['Strategy_Return'] - portfolio_returns[
            'Market_Return']

        if fees == 1:
            portfolio_returns['Strategy_Return'] = portfolio_returns['Portfolio_Value'].pct_change(1)
        self.portfolio_returns = portfolio_returns
        return portfolio_returns

    def plot_prices(prices_df):
        """Takes a dataframe containing stock prices and plots them

        :param prices_df: 

        """
        copy_df = prices_df.copy(deep=True)
        copy_df = copy_df.fillna(0)
        copy_df.plot(title='Stock Prices in Portfolio and Market Price (S&P500)')
        plt.show()

    def plot_returns(self):
        """Takes a dataframe containing backtest Market Returns, Strategy Returns, and Abnormal Returns and plots the cummulative returns"""
        backtest_df = self.portfolio_returns
        copy_df = backtest_df[['Market_Return', 'Strategy_Return', 'Abnormal Returns']].copy(deep=True)
        copy_df = copy_df.fillna(0)
        copy_df.plot(title='Market, Strategy, & Abnormal Returns')
        plt.show()

    def plot_cumulatve_returns(self):
        """Takes a dataframe containing backtest Market Returns, Strategy Returns, and Abnormal Returns and plots the cummulative returns"""
        backtest_df = self.portfolio_returns
        copy_df = backtest_df[['Market_Return', 'Strategy_Return', 'Abnormal Returns']].copy(deep=True)
        copy_df = copy_df.fillna(0)
        copy_df = copy_df.cumsum(axis=0)
        copy_df.plot(title='Cumulative Returns')
        plt.show()

    def compute_drawdown(self):
        """which processes a backtested dataframe adn returns the drawdown percentage of strategy and market returns"""
        backtest_df = self.portfolio_returns
        copy_df = backtest_df[['Market_Return', 'Strategy_Return', 'Abnormal Returns']].copy(deep=True)
        copy_df['Market Prev Max'] = copy_df.loc[:, 'Market_Return'].cummax()
        copy_df['Strategy Prev Max'] = copy_df.loc[:, 'Strategy_Return'].cummax()
        copy_df['Strategy dd_pct'] = (copy_df['Strategy Prev Max'] - copy_df.iloc[:, 0]) / copy_df['Strategy Prev Max']
        copy_df['Market dd_pct'] = (copy_df['Market Prev Max'] - copy_df.iloc[:, 0]) / copy_df['Market Prev Max']
        print(copy_df)
        self.drawdown_df = copy_df
        return copy_df

    def plot_drawdown(self):
        """create and show two charts: 1 - The historical price and previous maximum price. 2 - The drawdown since previous maximum price as a percentage lost."""
        dd = self.drawdown_df
        dd[['Market dd_pct', 'Strategy dd_pct']].plot(title='Drawdown Percentage')
        dd[['Market_Return', 'Market Prev Max', 'Strategy_Return', 'Strategy Prev Max']].plot(title='Maximun Drawdown')
        plt.xlabel('Date')
        plt.show()

    def construct_portfolio(self):
        """ """
        self.strategy_backtest(self.portfolio_df)

    def graph_portfolio_analysis(self):
        """ """
        backtest_df = self.portfolio_returns
        #self.plot_returns()
        self.plot_cumulatve_returns()
        #self.compute_drawdown()
        #self.plot_drawdown()
        print(backtest_df.describe())

if __name__ == '__main__':
    
    start = time.time()
    start_date = '2000-01-01'

    spy = Security("SPY", start_date=start_date, T=T, sims =simulations, predict=predict,backtest=backtest)
    spy.analyze_probability()
    spy.graph_probability_analysis()
    spy.graph_volume_profile()

    
    end = time.time()
    print(f"Runtime = {end - start:.2f} seconds")
    
