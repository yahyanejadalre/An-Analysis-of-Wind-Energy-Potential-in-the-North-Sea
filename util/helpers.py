'''
This utility file contains some functions that are usefule
for visualizing the data, computing all monthly or yearly parameters or doing an inhomogeneity test.

'''


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt
import tueplots as tp
import datetime as dt
from sklearn.gaussian_process import GaussianProcessRegressor as GP
import sklearn.gaussian_process.kernels as GPK
from weibull import Weibull


def yearly_params(first: int, last: int, dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns a dataframe that has the parameters (estimated mit the MLE) for all the years in the intervall [start,end], 
    based on the dataframe that contains all our data
    '''

    # initialize a dataframe that has the years as indices
    yearly_df=pd.DataFrame()
    yearly_df['Years']=np.arange(first, last+1)
    yearly_df['param_lambda']=0.0
    yearly_df['param_beta']=0.0
    yearly_df.set_index('Years', inplace=True)

    # compute the parameters for each year
    for y in yearly_df.index:
        mask=dataframe['MESS_DATUM'].dt.year == y
        weibull=Weibull.estimate(dataframe[mask]['FF_10_wind'])
        yearly_df.loc[y, 'param_lambda' ]=weibull.lambd
        yearly_df.loc[y, 'param_beta' ]=weibull.beta

    return yearly_df


def monthly_params(first: int, last: int, dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns a dataframe that has the parameters (estimated with the MLE) for all the months of the years in the intervall [start,end], 
    based on the dataframe that contains all our data
    '''

    # make a dataframe that has year-month combinations as indices
    months_range = pd.date_range(start=f'{first}-01', end=f'{last+1}-01', freq='M').to_period('M')
    monthly_df = pd.DataFrame(index=months_range, columns=['param_lambda', 'param_beta'])
    monthly_df['param_lambda']=0.0
    monthly_df['param_beta']=0.0
    # compute the parameters for all year-month combinations
    for m in monthly_df.index:
        mask=(dataframe['MESS_DATUM'].dt.month == m.month)& (dataframe['MESS_DATUM'].dt.year == m.year)
        weibull=Weibull.estimate(dataframe[mask]['FF_10_wind'])
        monthly_df.loc[m, 'param_lambda' ]=weibull.lambd
        monthly_df.loc[m, 'param_beta' ]=weibull.beta

    return monthly_df


def plot_timeframe(  df: pd.DataFrame, year: int, month: int =-1, day:int =-1):
    '''
    Only has a side-effect, no return
    Plots the windspeed the wind speed for given (year, month, day) or  ( (year, month) or year
    Is dependent on the exact dataframe df as we use it 
    '''

    #write a mask according to the input
    if day!=-1 and month!=-1:
        mask = (df['MESS_DATUM'].dt.day == day) & (df['MESS_DATUM'].dt.month == month) & (df['MESS_DATUM'].dt.year == year)
    elif month!=-1:
         mask = (df['MESS_DATUM'].dt.month == month) & (df['MESS_DATUM'].dt.year == year)
    else:
        mask = (df['MESS_DATUM'].dt.year == year)

    # raise an error, if the input was not sensible
    if len(df[mask])==0:
        raise Exception('The input is not valid or there are no data points for the desired timeframe')
    

    else: 
        plt.plot(df[mask]["MESS_DATUM"], df[mask]["FF_10_wind"])
        plt.xlabel('Time')
        plt.ylabel('Wind in m/s')
        plt.title('Windspeed over Time in the Chosen Timeframe')
        plt.show()


def plot_timeframe_pdf(df: pd.DataFrame, year: int, month: int =-1, day:int =-1):
    '''
    function that plots for the chosen year (and optionaly month, day)
    the empiric prodbability function as well as the fitted weibull ditribution
    '''

    #write a mask according to the input
    if day!=-1 and month!=-1:
        mask = (df['MESS_DATUM'].dt.day == day) & (df['MESS_DATUM'].dt.month == month) & (df['MESS_DATUM'].dt.year == year)
    elif month!=-1:
         mask = (df['MESS_DATUM'].dt.month == month) & (df['MESS_DATUM'].dt.year == year)
    else:
        mask = (df['MESS_DATUM'].dt.year == year)


    
    timeframe_df = df[mask]
    Y = timeframe_df["FF_10_wind"].dropna().to_numpy()
    
    # raise an error, if the input was not sensible
    if len(Y)==0:
        raise Exception('The input is not valid or there are no data points for the desired timeframe')
    
    fig, ax = plt.subplots()
    # estimate the weibull parameters and plot the corresponding probability density function
    weibull = Weibull.estimate(Y)
    X = np.arange(0, np.max(Y), 0.1)

    ax1 = ax.twinx()
    ax1.plot(X, weibull.pdf(X), color=tp.constants.color.rgb.tue_blue, label=r"$p(v \mid \hat{\lambda}, \hat{\beta})$")
    ax1.set_ylim(0)
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.legend(loc="lower right")

    # plot the histogram for the empiric pdf
    
    # a sensible number of bins is k=\delta \cdot \sqrt(n), 
    #where \delta is the expected perecentage of error for the probability of each bin, and n the number of data points
    k= int(0.1* np.sqrt(len(Y)))+1
    timeframe_df.hist(column="FF_10_wind", bins=k, ax=ax, color=tp.constants.color.rgb.tue_red, label="Frequency")
    ax.set_title("")
    ax.set_xlim(0)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_title(f"Yearly Frequencies \& Weibull Estimation of Wind Speeds for the Chosen Timeframe")



def bf_classifier(data):
    '''
    classifies the input according to the Beaufort-scale
    '''

    if data >24.5:
        return 10
    elif data >20.8:
        return 9
    elif data >17.2:
        return 8
    elif data >13.9:
        return 7
    elif data >10.8:
        return 6
    elif data >8.0:
        return 5
    elif data >5.5:
        return 4
    elif data >3.4:
        return 3
    elif data >1.6:
        return 2
    elif data >0.3:
        return 1
    else:
        return 0
    

def snh_test( X: np.array) -> list:
   '''
   Computes the test statistic of the standard normal homogeneity test for the given data X
   '''
   Y=np.array(X)
   s=Y.std()
   m=Y.mean()
   T=[]
   for k in range (0, len(Y)):
       z_1= 0 
       for i in range (0, k+1):
           z_1=z_1+(Y[i]- m)/s
       z_1=z_1/(k+1)
       
       z_2= 0 
       for i in range (k+1, len(Y)):
           z_2=z_2+(Y[i]- m)/s
       z_2=z_2/(len(Y) - (k))


       T.append(k*z_1**2 + (len(Y) -k)*z_2**2)
   return T


def pettitt_test( X: np.array) -> list:
   '''
   Computes the test statistic of the pettitt test for the given data X
   '''
   Y=np.array(X)
   R=np.argsort(Y)
   T=[]
   x_k=0
   for k in range(0, len(R)):
      x_k=0
      for i in range(0,k):
         x_k=x_k + 2*(R[i]+1)
      
      x_k=x_k-(k+1)*(len(R)+1)
      T.append(np.abs(x_k))
   return T

def days_to_date(start_dt: dt.datetime, days):
    return list(map(lambda delta_day: dt.timedelta(int(delta_day)) + start_dt, days.flatten()))