from IPython.display import display
import pandas as pd 
import numpy as np
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import sys
sys.path.append('../')
import commons as cm
from data_curation import Normalization
from pandasql import sqldf
import json
nm=Normalization()



def plot_pupil_with_events(pupil_df:pd.DataFrame,annotattion_df:pd.DataFrame,time_col:str='timestamp_s',signal_col:str='diameter_3d'):
    """Function to plot annotations and overlaying the signal of interest

    Args:
        pupil_df (pd.DataFrame): signal pandas data frame of interest
        annotattion_df (pd.DataFrame): _description_
        time_col (str): time column name of the dataframe
        signal_str (_type_): _description_
    """

    sns.set_theme()
    fig, ax=plt.subplots(1,)
    ax.plot(pupil_df[time_col],pupil_df[signal_col],'.')
    for  index, row in annotattion_df.iterrows():
        plt.axvline(row[time_col], color='r', label='axvline - full height')
        ax.text((row[time_col]),3,row['label'])
    ax.set_title('pupil diameter and events')    
    ax.set_xlabel(f'{time_col}')
    ax.set_ylabel(f'{signal_col}')
    fig.set_size_inches((18, 5.5))


