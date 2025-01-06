import plotly.graph_objects as go
import pandas as pd 
import numpy as np 
import plotly.express as px


def get_violin(df, interval):
    fig = go.Figure()

    periods = sorted(df[interval].unique())
    for period in periods:
        fig.add_trace(go.Violin(y=df[df[interval] == period]['SPEED_MA'], name=f'{period}', box_visible=False, meanline_visible=True))
        
    fig.update_layout(
        title='Violin Plots of Weekly Moving Average of Wind Speed for Each {} Over the Years 1959-2023'.format(interval),
        xaxis_title='{}'.format(interval),
        yaxis_title='MA Wind Speed'
    )
    
    fig.update_layout(yaxis_range=[0, 220])

    return fig

def load_dataset(path, ma_window=7, max_window=30, is_random=False):
    df = pd.DataFrame()
    df = pd.DataFrame(columns=['SPEED', 'YEAR', 'MONTH', 'DAY', 'DECADE'])
    # df = pd.DataFrame(columns=['STAID', 'SOUID','DATE','FG','Q_FG'])
    star_parsing = False 
    with open(path, "r") as file:
        ith_row = 0
        while True:
            line = file.readline() 
            
            if len(line) == 0:
                break

            elif line.startswith(' STAID, SOUID,    DATE,   FG, Q_FG'):
                star_parsing = True
            elif star_parsing:
                fields = line.rstrip().strip().split(',')

                STAID = fields[0]
                SOUID = fields[1]
                DATE = fields[2]
                FG = int(fields[3])
                Q_FG = fields[4]
                YEAR = int(DATE[0:4])
                MONTH = int(DATE[4:6])
                DAY = int(DATE[6:8])
                DECADE = int(str(YEAR)[0:-1] + '0')

                if int(Q_FG) != 9:
                    # df.loc[ith_row] = [STAID, SOUID, DATE, FG, Q_FG]
                    df.loc[ith_row] = [FG, YEAR, MONTH, DAY, DECADE]
                    ith_row += 1

    if is_random:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['SPEED_MA'] = df['SPEED'].rolling(ma_window).mean()
    df['SPEED_MAX'] = df['SPEED'].rolling(max_window).max() # not the prettiest way but is it correct?!
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear


    return df
    