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
# from pandasql import sqldf
import json
nm=Normalization()


# pupil_pd_frame=final_df
def calculate_pupillometry(pupil_pd_frame,annotations_pd,recording_location,window_s=2,seconds_norm=0.05,signal_str='diameter_3d_z_score'):
   

    ## Extract data of interest
    event_initial=annotations_pd['label'].values[0]
    event_final=annotations_pd['label'].values[-1]

    initial_anotation,_,_=cm.extract_annotations_timestamps(event_initial,'label',annotations_pd)
    end_anotation=annotations_pd.iloc[-1]
    display(initial_anotation)
    display(end_anotation)

    pupil_pd_frame=cm.filter_rows_by_temporal_values(
            dataframe=pupil_pd_frame,
            time_column='pupil_timestamp',
            ini_value=initial_anotation['timestamp'].values[0],
            end_value=end_anotation['timestamp']
            )
    
    ## Extract the data
    # Extracting pupil 3d data for analysis
    pupil_pd_frame['on_surf']=True
    confidence_thr=1
    
    # filter for 3d data
    detector_3d_data = pupil_pd_frame[pupil_pd_frame.method == 'pye3d 0.3.0 real-time']

    pupil_left_eye_interpolated=detector_3d_data.loc[(pupil_pd_frame['eye_id']==1)& (pupil_pd_frame['on_surf']==True)]
    pupil_right_eye_interpolated=detector_3d_data.loc[(pupil_pd_frame['eye_id']==0)& (pupil_pd_frame['on_surf']==True)]

    left_conf_index=(pupil_left_eye_interpolated['confidence']>=confidence_thr).values
    right_conf_index=(pupil_right_eye_interpolated['confidence']>=confidence_thr).values

    pupil_left_eye=detector_3d_data.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']>=confidence_thr)]
    pupil_right_eye=detector_3d_data.loc[(pupil_pd_frame['eye_id']==0) & (pupil_pd_frame['confidence']>=confidence_thr)]

    blinks_left_eye=detector_3d_data.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']< confidence_thr)]
    blinks_right_eye=detector_3d_data.loc[(pupil_pd_frame['eye_id']==0) & (pupil_pd_frame['confidence']< confidence_thr)]

    pupil_left_eye_surface=detector_3d_data.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']>=confidence_thr)& (pupil_pd_frame['on_surf']==True)]
    pupil_right_eye_surface=detector_3d_data.loc[(pupil_pd_frame['eye_id']==0) & (pupil_pd_frame['confidence']>=confidence_thr)& (pupil_pd_frame['on_surf']==True)]

    pupil_left_eye_surface=detector_3d_data.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']>=confidence_thr)& (pupil_pd_frame['on_surf']==True)]
    pupil_right_eye_surface=detector_3d_data.loc[(pupil_pd_frame['eye_id']==0) & (pupil_pd_frame['confidence']>=confidence_thr)& (pupil_pd_frame['on_surf']==True)]

    blinks_left_eye_surface=detector_3d_data.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']< confidence_thr)& (pupil_pd_frame['on_surf']==True)]
    blinks_right_eye_surface=detector_3d_data.loc[(pupil_pd_frame['eye_id']==0) & (pupil_pd_frame['confidence']< confidence_thr)& (pupil_pd_frame['on_surf']==True)]

    out_surface=detector_3d_data.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']>=confidence_thr)& (pupil_pd_frame['on_surf']==True)]
    pupil_left_eye_interpolated['diameter_3d_cubic']=cm.cubic_siplne_interpolation(
                                                    x=pupil_left_eye_interpolated.loc[left_conf_index,['pupil_timestamp']].values.flatten(),
                                                    y= pupil_left_eye_interpolated.loc[left_conf_index,['diameter_3d']].values.flatten(),
                                                    x_interpolate=pupil_left_eye_interpolated['pupil_timestamp'].values.flatten()
                                                    )

    pupil_right_eye_interpolated['diameter_3d_cubic']=cm.cubic_siplne_interpolation(
                                                    x=pupil_right_eye_interpolated.loc[right_conf_index,['pupil_timestamp']].values.flatten(),
                                                    y= pupil_right_eye_interpolated.loc[right_conf_index,['diameter_3d']].values.flatten(),
                                                    x_interpolate=pupil_right_eye_interpolated['pupil_timestamp'].values.flatten()
                                                    )

    #Filters for datasets
    filter_events=annotations_pd['label'].str.contains('Asset') | annotations_pd['label'].str.contains('Control') | annotations_pd['label'].str.contains('Surprise')
    filter_assets=annotations_pd['label'].str.contains('Asset') 
    filter_events_all=~annotations_pd['label'].str.contains('EndOfExperiment') 
    filter_events_surprise= annotations_pd['label'].str.contains('Surprise')
    filter_events_blank= annotations_pd['label'].str.contains('blank')

    # print("eye0 (right eye) data:")
    # display(pupil_right_eye[['pupil_timestamp', 'eye_id', 'confidence', 'norm_pos_x', 'norm_pos_y', 'diameter_3d']].head(10))

    # print("eye1 data (left eye) data:")
    # display(pupil_left_eye[['pupil_timestamp', 'eye_id', 'confidence', 'norm_pos_x', 'norm_pos_y', 'diameter_3d']].head(10))

    # df to ankalyse

    pupil_left_eye=pupil_left_eye
    pupil_right_eye=pupil_right_eye

    blinks_left_eye=blinks_left_eye
    blinks_right_eye=blinks_right_eye


    ##  Explore blinks and Asses data quality, are there many regions with data gaps? 
    parts = list(recording_location.parts)
    parts=parts[0:-2]
    parts=Path(*parts)
    f = open(parts.joinpath('info.player.json'))
    data = json.load(f)
    time_whole_recording=data['duration_s']
    time_roi_left=abs(pupil_left_eye['pupil_timestamp'].values[-1]-pupil_left_eye['pupil_timestamp'].values[0])
    time_roi_right=abs(pupil_right_eye['pupil_timestamp'].values[-1]-pupil_right_eye['pupil_timestamp'].values[0])

    pupil_sampling_freq_left=(pupil_left_eye.shape[0]+blinks_left_eye.shape[0])/time_roi_left
    pupil_sampling_freq_right=(pupil_right_eye.shape[0]+blinks_right_eye.shape[0])/time_roi_right 
    print('sampling frequency for right eye is {}'.format(pupil_sampling_freq_right)) 
    print('sampling frequency for left eye is {}'.format(pupil_sampling_freq_left)) 
    # Closing file
    f.close()

    ### Asses data quality
    total_blinks_left=blinks_left_eye.shape[0]/pupil_sampling_freq_left
    total_blinks_right=blinks_right_eye.shape[0]/pupil_sampling_freq_right
    print(f'Total blinks time left {total_blinks_left:.2f}s and right {total_blinks_right:.2f}s of a total of {time_roi_left:.2f}s left and {time_roi_right:.2f}s right')
    print(f'Total blinks time left {total_blinks_left/time_roi_left:.2%} ')
    print(f'Total blinks time right {total_blinks_right/time_roi_right:.2%} ')

    ## Put timestamp data in seconds

    pupil_left_eye['timestamp_s']=pupil_left_eye['pupil_timestamp']-pupil_left_eye['pupil_timestamp'].values[0]
    pupil_left_eye['timestamp_s']
    pupil_right_eye['timestamp_s']=pupil_right_eye['pupil_timestamp']-pupil_right_eye['pupil_timestamp'].values[0]
    pupil_right_eye['timestamp_s']
    annotations_pd['timestamp_s']=annotations_pd['timestamp']-annotations_pd['timestamp'].values[0]
    

    #Apply normalization
    pupil_left_eye_interpolated['diameter_3d_cubic_z_score']=nm.normalize(values=pupil_left_eye_interpolated['diameter_3d_cubic'],
                                                type='z_score')

    pupil_right_eye_interpolated['diameter_3d_cubic_z_score']=nm.normalize(values=pupil_right_eye_interpolated['diameter_3d_cubic'],
                                                type='z_score')
    pupil_left_eye['diameter_3d_z_score']=nm.normalize(values=pupil_left_eye['diameter_3d'],
                                                type='z_score')

    pupil_right_eye['diameter_3d_z_score']=nm.normalize(values=pupil_right_eye['diameter_3d'],
                                                 type='z_score')

    # pupil_df['speed']=abs(pupil_df['diameter_3d'].diff(periods=100).rolling(3).median())
    #common variables for analysis
    pupil_df=pupil_left_eye_interpolated

    frames_norm=np.round(seconds_norm*pupil_sampling_freq_left).astype(int)
    win_norm=range(frames_norm)
    print('Initial frames used for normalization = {} correspond to {}s'.format(frames_norm,seconds_norm))

    window_frames=np.round(window_s*pupil_sampling_freq_left).astype(int)
    print('Windows frames of interest   = {} correspond to {}s'.format(window_frames,window_s))
    window=range(0,window_frames)
    time_x=np.linspace(0,window_s,window_frames)

    # For Assets only window size data
    event=annotations_pd.loc[filter_events,['label']].values.flatten()
    event_strip=[image.split('.')[0] for image in event] #remove .tiiff

    # event=annotations_pd['label'].values.flatten()
    data_dict=dict([(key,[None]) for key in event_strip])# dict with empty keys 

    pupil_diameter_df=pd.DataFrame()#pd.DataFrame(data_dict,index=np.arange(0,800))
    data_list=[]
    for im,im_strip in zip(event,event_strip):
        initial_anotation,end_anotation,index_annotation=cm.extract_annotations_timestamps(im,'label',annotations_pd)
        segmented_df=cm.filter_rows_by_temporal_values(
            dataframe=pupil_df,
            time_column='pupil_timestamp',
            ini_value=initial_anotation['timestamp'].values[0],
            end_value=end_anotation['timestamp'].values[0]
            )
        
        segmented_df=segmented_df.iloc[window]
        win_blank=segmented_df.iloc[win_norm]
        asset_norm=segmented_df[signal_str]-win_blank[signal_str].mean()
        asset_raw=segmented_df[signal_str]
        data_dict[im_strip]=asset_norm.values
    pupil_diameter_df=pd.DataFrame(data_dict)
    return pupil_diameter_df