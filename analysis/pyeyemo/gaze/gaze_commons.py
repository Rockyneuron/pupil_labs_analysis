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
from IPython.display import display
from itertools import compress
from mungling import DataMungling

nm=Normalization()

def extract_session_path_pupil_labs(recording_location:str,subject:str):
    """Function to retrieve data paths for each subject of annation dataframes,
    fixations, fixations on surface and so on from pupil labs.

    Args:
        recording_location (str): 
        subject (str): _description_

    Returns:
        _dict_: dictionary with the data paths of the csvs of interest
                {'annotations':annotations_csv,
                 'fixations':fixations_csv,
                 'fixations_surf':fixations_surf_dir}
    """
    #Load data in folders
    recording_folder=[record for record in os.listdir(recording_location)  if '00' in record]
    print(recording_folder)
    index_aux = list(map(lambda x: not('_' in x), recording_folder))
    recording_folder=list(compress(recording_folder,index_aux))
    #Load data in folders
    if len(recording_folder)>1:
        ValueError('Ambiguty in folder of experiment')
    recording_location=recording_location.joinpath(recording_folder[0],'exports')
    recording_location_raw=recording_location.joinpath(recording_folder[0],'exports')
    export_folder=[record for record in os.listdir(recording_location)  if '00' in record]
    if len(export_folder)>1:
        ValueError('Ambiguty in folder of exports')
    recording_location=recording_location.joinpath(export_folder[0])
    fixations_surf_csv=[record for record in os.listdir(recording_location.joinpath('surfaces'))  if 'fixations_on_surface' in record][0]
    fixations_surf_dir = os.path.join(recording_location, 'surfaces',fixations_surf_csv)

    annotations_csv = os.path.join(recording_location,'annotations.csv')
    fixations_csv = os.path.join(recording_location,'fixations.csv')

    data_paths= {'annotations':annotations_csv,
                 'fixations':fixations_csv,
                 'fixations_surf':fixations_surf_dir}
    
    return data_paths

def distance_x_y(x:pd.Series,y:pd.Series):
    """function to calculate the distance between two points
    Args:
        x (pd.Series): array of x values 
        y (pd.Series): array of y values
    Returns:
        _type_: _description_
    """
    return np.sqrt((np.diff(x)**2)+((np.diff(y))**2))


def calculate_contrast(x,y):
    """Function to calculate michealson contrast
    Args:
        x (_np.array_): _description_
        y (_np.arry_): _description_
    """
    contrast=(x-y)/(x+y)
    # print(contrast)
    return contrast

def eliminate_duplicates(self,df:pd.DataFrame,column='fixation_id') ->None: 
    """Eliminate duplicate values of columns

    Args:
        df (pd.DataFrame): _description_
        column (str, optional): column to remove duplicates . Defaults to 'fixation_id'.
    """
    self.fixations=(df.drop_duplicates(subset=[column])
            )

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