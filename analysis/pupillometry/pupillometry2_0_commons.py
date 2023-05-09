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



def plot_signal_with_events(signal_df:pd.DataFrame,
                            annotattion_df:pd.DataFrame,
                            time_col:str='timestamp_s',
                            signal_col:str='diameter_3d'):
    """Function to plot annotations and overlaying the signal of interest

    Args:
        pupil_df (pd.DataFrame): signal pandas data frame of interest
        annotattion_df (pd.DataFrame): _description_
        time_col (str): time column name of the dataframe
        signal_str (_type_): _description_
    """
    sns.set_theme()
    fig, ax=plt.subplots(1,)
    ax.plot(signal_df[time_col],signal_df[signal_col],'.')
    for  index, row in annotattion_df.iterrows():
        plt.axvline(row[time_col], color='r', label='axvline - full height')
        ax.text((row[time_col]),3,row['label'])
    ax.set_title('pupil diameter and events')    
    ax.set_xlabel(f'{time_col}')
    ax.set_ylabel(f'{signal_col}')
    fig.set_size_inches((18, 5.5))
    return fig,ax


def plot_events_and_surprise(signal_df:pd.DataFrame,
                             filter_assets:list,
                             filter_surprise:list,
                             time_x:np.array,
                             subject:str
                            ):

    mat=signal_df.reindex(columns=filter_assets).values
    mat_std=signal_df.reindex(columns=filter_assets).values

    mat2=signal_df.reindex(columns= filter_surprise).values
    mat2_std=signal_df.reindex(columns= filter_surprise).values

    fig, ax=plt.subplots(3,1,)
    ax[0].plot(time_x,mat)
    ax[0].plot(time_x,np.mean(mat,axis=1),linewidth=5,color='black')
    # ax.legend([images,'assets mean'])
    ax[0].set_title(f'Raw assets and mean pupilometry speed for subject: {subject} ')
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('diameter(mm)')


    ax[1].plot(time_x,mat2)
    ax[1].plot(time_x,np.mean(mat2,axis=1),linewidth=5,color='black')
    ax[1].set_title('Raw surprise and mean pupilometry speed')
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('diameter (mm)')

    ax[2].errorbar(time_x,np.mean(mat,axis=1),np.std(mat_std,axis=1),linewidth=2,color='blue')
    ax[2].errorbar(time_x,np.mean(mat2,axis=1),np.std(mat2_std,axis=1),linewidth=0.5,color='red')
    ax[2].legend(['assets','surprise'])
    ax[2].plot(time_x,np.mean(mat2,axis=1),linewidth=2,color='black')
    ax[2].plot(time_x,np.mean(mat,axis=1),linewidth=2,color='black')

    fig.set_size_inches((18, 18))
    fig.tight_layout()
    fig.savefig(f'figures/{subject}_data_assets.png', dpi=300)
        
    def extract_eye_data(pupil_df:pd.DataFrame,eye_id: str = 'best',
               method: str = 'pye3d 0.3.0 real-time',confidence_thr: float=0.95)-> pd.DataFrame:
        # TODO: Refactor function in smaller chunks-- clean code
        """Function to extract pupil data for eye of interest. 
        Left eye, right eye or best eye can be extracted for further analysis

        Args:
            pupil_df (pd.DataFrame): pupil pandas dataframe
            eye_id (str, optional): _description_. Defaults to 'best'.
            method (str, optional): _description_. Defaults to 'pye3d 0.3.0 real-time'.
            confidence_thr: Confidence thershold for data qualiltu filtering. Deafults to 0.95

        Raises:
            ValueError: a string must be put to indicate if we are using left, right eye or best

        Returns:
            pd.DataFrame: pandas dataframe with pupil data of interest
        """
        global SAMPLE_RATE
        pupil_df = pupil_df.loc[pupil_df.method.str.contains(method)]

        pupil_left_eye=pupil_df.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']>=confidence_thr)]
        pupil_right_eye=pupil_df.loc[(pupil_pd_frame['eye_id']==0) & (pupil_pd_frame['confidence']>=confidence_thr)]

        blinks_left_eye=pupil_df.loc[(pupil_pd_frame['eye_id']==1) & (pupil_pd_frame['confidence']< confidence_thr)]
        blinks_right_eye=pupil_df.loc[(pupil_pd_frame['eye_id']==0) & (pupil_pd_frame['confidence']< confidence_thr)]


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
        print(f'time of the whole recording is {time_whole_recording}')
        print(f'time after anottaion cutting {time_roi_left:.2f}s for left eye and {time_roi_right:.2f}s for right eye')
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

        if eye_id == 'left':
            pupil_df = pupil_df[pupil_df.eye_id == 1]
            SAMPLE_RATE=pupil_sampling_freq_left
        elif eye_id == 'right':
            pupil_df = pupil_df[pupil_df.eye_id == 0]
            SAMPLE_RATE=pupil_sampling_freq_right
        elif eye_id == 'best':
            best = pupil_df.groupby(['eye_id'])['confidence'].mean()
            display(best)
            best=best.idxmax()
            pupil_df = pupil_df[pupil_df.eye_id == best]
            eye='left'if best==1 else 'right'
            print(f'data from {eye} eye was selected')
            SAMPLE_RATE=pupil_sampling_freq_left if best==1 else pupil_sampling_freq_right
        else:
            raise ValueError('Eye must be "left", "right" or "best".')
        print('Loaded {} samples'.format(len(pupil_df)))
        return pupil_df


    def calculate_pupillometry(pupil_pd_frame,annotations_pd,recording_location,window_s=2,seconds_norm=0.05,signal_str='diameter_3d_z_score'):

        # Cut data by annotations of interest
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
        
        # ## Data cleaning
        # Extracting pupil 3d data for analysis:
        # 1) Select eye that has a better signal
        # 2) Clean blinks from data
        # 3) Asses that quality
        pupil_df_raw=extract_eye_data(pupil_df=pupil_pd_frame,
                          eye_id='best',
                          method='pye3d 0.3.0 real-time')


        ## Put timestamp data in seconds
        pupil_df_raw['timestamp_s']=pupil_df_raw['pupil_timestamp']-pupil_df_raw['pupil_timestamp'].values[0]
        annotations_pd['timestamp_s']=annotations_pd['timestamp']-annotations_pd['timestamp'].values[0]

        filter_events=annotations_pd['label'].str.contains('Asset') | annotations_pd['label'].str.contains('Control') | annotations_pd['label'].str.contains('Surprise')

        fig,ax=plot_signal_with_events(signal_df=pupil_df_raw,
                                annotattion_df=annotations_pd[filter_events],
                                time_col='timestamp_s',
                                signal_col='diameter_3d')
        ax.set_title(f'raw data for subject: {subject}')

        fig.savefig(f'figures/{subject}_Raw.png', dpi=300)