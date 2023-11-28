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

def extract_session_path_lsl(recording_location:str,subject:str):
    """Function to retrieve data paths for each subject of emotibit data.

    Args:
        recording_location (str): 
        subject (str): _description_

    Returns:
        _dict_: dictionary with the data paths of the csvs of interest
                {'lsl_data':--to complete}
    """

    recording_folder=[record for record in os.listdir(recording_location)]
    recording_folder
    #load data in folder
    if len(recording_folder)>1:
        Warning('Ambiguty in folder of experiment')
    recording_location=recording_location.joinpath(recording_folder[0])
    #add lsl data file to path
    data_file_path=[record for record in os.listdir(recording_location)]
    if len(data_file_path)>1:
        Warning('Ambiguity in number of data files')
    recording_location=recording_location.joinpath(data_file_path[0])  

    data_paths={'lsl_data':recording_location}
    return data_paths

def extract_session_path_emotibit(recording_location:str,subject:str):

    #add emotibit data file to path
    data_file_path=[record for record in os.listdir(recording_location)]
    if len(data_file_path)>1:
         Warning('Ambiguity in number of data files')
    
    heart_rate=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_HR.csv')  
    annotations=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_LM.csv')  
    heart_beat_inter_beat=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_BI.csv')

    electrodermal_activity=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_EA.csv')
    electrodermal_level=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_EL.csv')
    # electrodermal_response=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_ER.csv')

    ppg_infrared=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_PI.csv')
    ppg_red=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_PR.csv')
    ppg_green=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_PG.csv')

    accelerometer_x=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_AX.csv')
    accelerometer_y=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_AY.csv')
    accelerometer_z=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_AZ.csv')


    gyroscope_x=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_GX.csv')
    gyroscope_y=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_GY.csv')
    gyroscope_z=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_GZ.csv')

    magnetometer_x=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_MX.csv')
    magnetometer_y=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_MY.csv')
    magnetometer_z=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_MZ.csv')


    skin_cond_resp_amplitude=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_SA.csv')
    skin_cond_resp_rise_time=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_SR.csv')
    skin_cond_resp_rise_freq=recording_location.joinpath(data_file_path[0].replace('.csv','')+'_SF.csv')


    data_paths={'annotations_LM':annotations,
               'heart_rate_HR':heart_rate,
                'heart_beat_inter_beat_BI':heart_beat_inter_beat,
                'electrodermal_activity_EA':electrodermal_activity,
                'electrodermal_level_EL':electrodermal_level,
                # 'electrodermal_response':electrodermal_response,
                'ppg_infrared_PI':ppg_infrared,
                'ppg_red_PR':ppg_red,
                'ppg_green_PG':ppg_green,
                'accelerometer_x_AX':accelerometer_x,
                'accelerometer_y_AY':accelerometer_y,
                'accelerometer_z_AZ':accelerometer_z,
                'gyroscope_x_GX':gyroscope_x,
                'gyroscope_y_GY':gyroscope_y,
                'gyroscope_z_GZ':gyroscope_z,
                'magnetometer_x_MX':magnetometer_x,
                'magnetometer_y_MY':magnetometer_y,
                'magnetometer_z_MZ':magnetometer_z,
                'skin_cond_resp_amplitude_SA':skin_cond_resp_amplitude,
                'skin_cond_resp_rise_time_SR':skin_cond_resp_rise_time,
                'skin_cond_resp_rise_freq_SF':skin_cond_resp_rise_freq}
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