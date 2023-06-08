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
import pupillometry_commons as cp
from itertools import compress
import json
import importlib
from pyeyemo.pupil import PLR
from pyplr import graphing
from pyplr import preproc
from pyplr import utils
from IPython.display import display

nm=Normalization()


def plot_signal_with_events(signal_df:pd.DataFrame,
                            annotattion_df:pd.DataFrame,
                            time_col:str='timestamp_s',
                            signal_col:str='diameter_3d'
                            ):
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

    pupil_left_eye=pupil_df.loc[(pupil_df['eye_id']==1) & (pupil_df['confidence']>=confidence_thr)]
    pupil_right_eye=pupil_df.loc[(pupil_df['eye_id']==0) & (pupil_df['confidence']>=confidence_thr)]

    blinks_left_eye=pupil_df.loc[(pupil_df['eye_id']==1) & (pupil_df['confidence']< confidence_thr)]
    blinks_right_eye=pupil_df.loc[(pupil_df['eye_id']==0) & (pupil_df['confidence']< confidence_thr)]


    ##  Explore blinks and Asses data quality, are there many regions with data gaps? 
    parts = list(RECORDING_LOCATION.parts)
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

def blink_cleaning(pupil_df:pd.DataFrame,
                   pupil_cols:list,
                   confidence_thr:float):
    # Make figure for processing
    f, axs = graphing.pupil_preprocessing_figure(nrows=5, subject=SUBJECT)

    # Plot the raw data
    pupil_df[pupil_cols].plot(title='Raw', ax=axs[0], legend=True)
    axs[0].legend(loc='center right', labels=['mm', 'pixels'])

    # Mask first derivative
    pupil_df = preproc.mask_pupil_first_derivative(
        pupil_df, threshold=3.0, mask_cols=pupil_cols)
    pupil_df[pupil_cols].plot(
        title='Masked 1st deriv (3*SD)', ax=axs[1], legend=False)

    # Mask confidence
    pupil_df = preproc.mask_pupil_confidence(
        pupil_df, threshold=confidence_thr, mask_cols=pupil_cols)
    pupil_df[pupil_cols].plot(
        title=f'Masked confidence (<{confidence_thr})', ax=axs[2], legend=False)

    # Interpolate
    pupil_df = preproc.interpolate_pupil(
        pupil_df, interp_cols=pupil_cols)
    pupil_df[pupil_cols].plot(
        title='Linear interpolation', ax=axs[3], legend=False)

    # Smooth
    pupil_df = preproc.butterworth_series(
        pupil_df, fields=pupil_cols, filt_order=3,
        cutoff_freq=4/(SAMPLE_RATE/2))
    pupil_df[pupil_cols].plot(
        title='3rd order Butterworth filter with 4 Hz cut-off',
        ax=axs[4], legend=False);


    f.savefig(f'figures/{SUBJECT}_Blinks.png', dpi=300)
    return pupil_df


def calculate_pupillometry(pupil_pd_frame:pd.DataFrame,
                           annotations_pd:pd.DataFrame,
                           recording_location:Path,
                           confidence_thr:float,
                           window_s:int=2,
                           seconds_norm:float=0.05,
                           signal_str:str='diameter_3d_z_score',
                           subject:str='subject',
                           baseline_correction:str='no'):
        
        global RECORDING_LOCATION
        global SUBJECT
        RECORDING_LOCATION=recording_location
        SUBJECT=subject

        #####------------ Cut data by annotations of interest-----------------######

        event_initial=annotations_pd['label'].values[0]

        initial_anotation,_,_=cm.extract_annotations_timestamps(event_initial,'label',annotations_pd)
        end_anotation=annotations_pd.iloc[-1]

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


        #Plot raw signal
        fig,ax=plot_signal_with_events(signal_df=pupil_df_raw,
                                annotattion_df=annotations_pd[filter_events],
                                time_col='timestamp_s',
                                signal_col='diameter_3d')
        ax.set_title(f'raw data for subject: {subject}')

        fig.savefig(f'figures/{subject}_Raw.png', dpi=300)


        #Blink Cleaning
        pupil_df=blink_cleaning(pupil_df=pupil_df_raw,
                                pupil_cols=['diameter_3d'],
                                confidence_thr=confidence_thr)
        
        pupil_df['diameter_3d_z_score']=nm.normalize(values=pupil_df['diameter_3d'],
                                                type='z_score')
        

        #Analysis for all assets
        #First filter events
        filter_events=annotations_pd['label'].str.contains('Asset') | annotations_pd['label'].str.contains('Control') | annotations_pd['label'].str.contains('Surprise')

        #Extract number of freames for baseline norm
        frames_norm=np.round(seconds_norm*SAMPLE_RATE).astype(int)
        win_norm=range(frames_norm)
        print('Initial frames used for normalization = {} correspond to {}s'.format(frames_norm,seconds_norm))

        #Extract number of frames for window of interest
        window_frames=np.round(window_s*SAMPLE_RATE).astype(int)
        print('Windows frames of interest   = {} correspond to {}s'.format(window_frames,window_s))
        window=range(0,window_frames)
        time_x=np.linspace(0,window_s,window_frames)

        # For Assets only window size data
        event=annotations_pd.loc[filter_events,['label']].values.flatten()
        event_strip=[image.split('.')[0] for image in event] #remove .tiiff

        #Dictionaries to create dataframes
        data_dict=dict([(key,[None]) for key in event_strip])# dict with empty keys 
        data_dict_raw=dict([(key,[None]) for key in event_strip])# dict with empty keys 
        data_dict_params=dict([(key,[None]) for key in event_strip])# dict with empty keys 


        #Proceed withg the analysis
        #4 dataframes returned:
        # Raw data
        # Baseline norme
        # surpisr params
        # asset params
        print(f'analysing:... {signal_str}')
        pupil_diameter_df=pd.DataFrame()#pd.DataFrame(data_dict,index=np.arange(0,800))
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

            #Add raw and basilenimo norm values to dictionaries
            data_dict[im_strip]=asset_norm.values
            data_dict_raw[im_strip]=asset_raw.values

        #Create dataframes of interest
        pupil_diameter_df=pd.DataFrame(data_dict)
        pupil_diameter_df_raw=pd.DataFrame(data_dict_raw)

        #To filter assets
        filter_assets=list(pupil_diameter_df.columns)
        filter_surprise=list(pupil_diameter_df.columns)
        filter_assets=[asset for asset in filter_assets if 'blank' in asset ]
        filter_surprise=[asset for asset in filter_surprise if 'Surprise' in asset ]

        if baseline_correction=='no':
            pupil_diameter_assets_df=np.mean(pupil_diameter_df_raw.reindex(columns=filter_assets).values,axis=1)
            pupil_diameter_surprise_df=np.mean(pupil_diameter_df_raw.reindex(columns=filter_surprise).values,axis=1)
            pupil_diameter_all_df=pupil_diameter_df_raw
        elif baseline_correction=='yes':
            pupil_diameter_assets_df=np.mean(pupil_diameter_df.reindex(columns=filter_assets).values,axis=1)
            pupil_diameter_surprise_df=np.mean(pupil_diameter_df.reindex(columns=filter_surprise).values,axis=1)
            pupil_diameter_all_df=pupil_diameter_df
        else:
            raise ValueError('Inocrrect baseline correction parameter')
        #Plot the events and suprise and save
        plot_events_and_surprise(signal_df=pupil_diameter_df,
                                    filter_assets=filter_assets,
                                    filter_surprise=filter_surprise,
                                    time_x=time_x,
                                    subject=subject
                                    )

        #Extract asset parameters
        plr_assets = PLR(pupil_diameter_assets_df,
          sample_rate=int(SAMPLE_RATE),
          onset_idx=frames_norm,
          stim_duration=window_s,
          baseline_duration=frames_norm)
        fig = plr_assets.plot(vel=True, acc=True, print_params=True)
        ax=plt.gca()
        ax.set_title(f'assets: {SUBJECT}')
        fig.savefig(f'figures/{SUBJECT}_assets.png', dpi=300)

        plr_surprise = PLR(pupil_diameter_surprise_df,
                sample_rate=int(SAMPLE_RATE),
                onset_idx=frames_norm,
                stim_duration=window_s,
                baseline_duration=frames_norm)
        fig = plr_surprise.plot(vel=True, acc=True, print_params=True)
        ax=plt.gca()
        ax.set_title(f'surprise: {SUBJECT}')
        fig.savefig(f'figures/{SUBJECT}_surprise.png', dpi=300)

        return (plr_assets.parameters().T,plr_surprise.parameters().T,pupil_diameter_df,pupil_diameter_df_raw)