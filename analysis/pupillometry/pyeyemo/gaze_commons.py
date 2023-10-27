"""Vetical index calculation of preprocessed data
In this code we will load the data and calculate the vertical index as
VI=V-H/V+H
Names of files used in session are the ones used to load the
correspoding preprocessed data

Returns:
    _pd.dataframe_: A pandas dataframe with the vertical index data
      for each
    subject and per asset.
"""

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
from IPython.display import display
nm=Normalization()


def calculate_contrast(x,y):
    """Function to calculate michealson contrast
    Args:
        x (_np.array_): _description_
        y (_np.arry_): _description_
    """
    contrast=(x-y)/(x+y)
    # print(contrast)
    return contrast

def compute_vertical_index(vector_index):
    """Calculate vertical index from a vector of 0 and 1
    Args:
        vector_index (_type_): 0 and ones int numpy array vector
    """
    vertical=vector_index[vector_index==1].size
    horizontal=vector_index[vector_index==0].size
    vi=calculate_contrast(vertical,horizontal)
    # print(f'vertical: {vertical}, horizontal: {horizontal},verticality: {vector_index}')
    return vi

def vertical_index(gaze_pd_frame,annotations_pd):
    """Global function with all the steps to calcultate de vertical index
    """
    ### Cut all data by annotations of interest
    event_initial=annotations_pd['label'].values[0]
    event_final=annotations_pd['label'].values[-1]
    initial_anotation,_,_=cm.extract_annotations_timestamps(event_initial,'label',annotations_pd)
    end_anotation=annotations_pd.iloc[-1]

    gaze_pd_frame=cm.filter_rows_by_temporal_values(
            dataframe=gaze_pd_frame,
            time_column='start_timestamp',
            ini_value=initial_anotation['timestamp'].values[0],
            end_value=end_anotation['timestamp']
            )

    ## Extract the data of interest
    # Removing out of surface events
    gaze_pd_frame=gaze_pd_frame.query('on_surf==True')
    gaze_pd_frame['on_surf'].unique()

    ## Order events
    filter_events=annotations_pd['label'].str.contains('Asset') | annotations_pd['label'].str.contains('Control') | annotations_pd['label'].str.contains('Surprise')
    event=annotations_pd.loc[filter_events,['label']].values.flatten().tolist()
    event.sort()
    event_strip=[image.split('.')[0] for image in event] 

    ## Calculate Vertical Index
    data_dict=dict([(key,[None]) for key in event_strip])# dict with empty keys 
    vertical_index_df=pd.DataFrame()#pd.DataFrame(data_dict,index=np.arange(0,800))
    data_list=[]
    for im,im_strip in zip(event,event_strip):
        initial_anotation,end_anotation,index_annotation=cm.extract_annotations_timestamps(im,'label',annotations_pd)
        segmented_df=cm.filter_rows_by_temporal_values(
            dataframe=gaze_pd_frame,
            time_column='start_timestamp',
            ini_value=initial_anotation['timestamp'].values[0],
            end_value=end_anotation['timestamp']
            )   
        # display(segmented_df)
        verticality=segmented_df['verticality'].values
        vi=compute_vertical_index(verticality)
        data_dict[im_strip]=[vi]
    vertical_index_df=pd.DataFrame(data_dict)
    return vertical_index_df

