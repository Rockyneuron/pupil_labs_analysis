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

class Eye(DataMungling):

    @property
    def fixation(self):
        return self.fixation
    
    @property
    def vertical_index_df(self):
        return self._vertical_index_df
    
    @vertical_index_df.setter
    def vertical_index_df(self,new_df):
           self._vertical_index_df=new_df
    # @set
    # def labels(self):
    #     self.labels

    def __init__(self,name='some_subject') -> None:
        self.name=name
  
    def load_annotations(self,annotation_dir:str):
        """function to load annotations
        as pandas dataframe
        """
        self.annotations_csv = annotation_dir
        self.annotations=pd.read_csv(self.annotations_csv)
        self.data_matrix=self.annotations

    def load_fixations(self,fixations_dir:str):
        """method to load fixations csv as 
        pandas dataframe
        """
        self.fixations_dir = fixations_dir
        self.fixations=pd.read_csv(self.fixations_dir)     

    def eliminate_duplicates(self,df:pd.DataFrame,column:str='fixation_id') ->None: 
        """Eliminate duplicate values of columns

        Args:
            df (pd.DataFrame): dataframe to remove duplicates
            column (str, optional): column to remove duplicates . Defaults to 'fixation_id'.
        """
        self.fixations=(df.drop_duplicates(subset=[column])
            )
        
    def distance_x_y(x:pd.Series,y:pd.Series):
        """function to calculate the distance between two points
        Args:
            x (pd.Series): array of x values 
            y (pd.Series): array of y values
        Returns:
            _type_: _description_
        """
        return np.sqrt((np.diff(x)**2)+((np.diff(y))**2))
    
    def calculate_distance(self,x_col:str='norm_pos_x',y_col:str='norm_pos_y'):
        """Create distance column 

        Args:
            x_col (str, optional): x colmun values. Defaults to 'norm_pos_x'.
            y_col (str, optional): y column values. Defaults to 'norm_pos_y'.
        """
        self.fixations['distance']=np.insert(distance_x_y(x=self.fixations[x_col],
                                                          y=self.fixations[y_col]),0,0)
        self.x_col=x_col
        self.y_col=y_col
        
    def calculate_speed_saccades(self,time_col:str='world_timestamp'):

        diff_word_time=np.diff(self.fixations[time_col],prepend=0)

        self.fixations.insert(loc=self.fixations.shape[1],
                            column='speed',
                            value=self.fixations['distance']/diff_word_time )
    
    def saccade_angle(self,angle:str='degrees_90_90'):
        """Calculate sacade angle

        Returns:
            _type_: _description_
        """
        x=np.diff(self.fixations[self.x_col],prepend=0)
        y=np.diff(self.fixations[self.y_col],prepend=0)
        self.fixations['angle_rad']=np.arctan2(y,x)

        if angle=='degrees_90_90':
            self.fixations['angle']=np.arctan2(y,x)*180/np.pi
        if angle=='radians':
            self.fixations['angle_rad']=np.arctan2(y,x)
        if angle=='degrees_0_360':
           self.fixations['angle']=np.arctan2(y,x)*180/np.pi
        #    self.fixations.loc[self.fixations['angle']<0,['angle']]=self.fixations.loc[self.fixations['angle']<0,['angle']]+360
           self.fixations['angle']=self.fixations['angle'].map(lambda x: x+360 if x<0 else x)
    
    def vertical_horizontal_sacades(self,vert:list[int]=[45,135,225,315]):
        """Function to binary classify vertical==1 and horizontal==2 sacades

        Args:
            vert (list[int], optional): _description_. Defaults to [45,135,225,315].
        """
        self.fixations['verticality']=0
        ori=self.fixations['angle']

        self.fixations.loc[(ori>vert[0]) & (ori<vert[1]),['verticality']]=1
        self.fixations.loc[(ori>vert[2]) & (ori<vert[3]),['verticality']]=1   

    def label_data_annotation(self,
                              annotation_col:str='label',
                              annotation_time_col:str='timestamp',
                              label_name:str='asset',
                              fixations_time_col_name:str='start_timestamp'):
        """Method that labels a a new colum with the label of a reference annotation dataframe. 
        Basically i have 2 dataframes, with a similar temporal value range, and i want to assing those
        anotattions to a new column in the time interval in which they are happening.
        For this we turover the annotations_df to sort it by descending values, and then asing those annotation
        to timestamps>= annotation_timestamp

        Args:
            annotation_col (str, optional): name of the annotation label column. Defaults to 'label'.
            label_name (str, optional): name ofd the new column with the asiggned annotaiton. Defaults to 'asset'.
            fixations_time_col_name (str, optional): name of the timestamp colum of the main df. Defaults to 'start_timestamp'.
        """
        self.fixations[label_name]=self.fixations[fixations_time_col_name]

        for anotation in self.annotations.sort_values(by=[annotation_time_col],ascending=False).iterrows():
             self.fixations[label_name]=self.fixations[label_name].map(lambda x: anotation[1][annotation_col]\
                                                    if isinstance(x,float)\
                                                    and (x>=anotation[1][annotation_time_col])
                                                        else x)
    def cut_data_of_interest(self):

        self.fixations=self.cut_dataframe_by_column_values(df=self.fixations,
                                                            inital_label=self.annotations.iloc[0].label,
                                                            final_label='end_of_experiment',
                                                            filter_column='asset')
    
    def labels_to_analyse(self,labels:list[str]):
        self.labels=labels
    
    def filter_labels(self,df:pd.Series):
        self.annotation_list,annotation_index=self.filter_series_list_string(df=df,
                                       label=self.labels)
        
        # Main table filtered by asset where all data will be appended.

        self.data_matrix=pd.DataFrame(self.annotation_list)
        self.data_matrix['session']=self.name
        self.data_matrix.rename(columns={self.data_matrix.columns[0]:'asset'},inplace=True)
        self.data_matrix=self.data_matrix[['session','asset']]

    def vertical_index(self,window_analysis:float,screen_normalization=False,screen:list[str]=[1920,1080]):
        """Method to calculate vertical index.
        using the veriticaility columnn to count the total
        number of horizontal and vertical sacaddes. 
        vi=(H-V)/(H+V)

        Args:
            window_analysis (float): _description_
        """
        
        data_dict=dict([(key,[None]) for key in self.annotation_list])# dict with empty keys 
        vertical_index_df=pd.DataFrame()#pd.DataFrame(data_dict,index=np.arange(0,800))
        data_list=[]
        for asset in self.annotation_list:
            aux_df=self.fixations.query(f"asset == '{asset}'") #break table by asset name
            time_0=aux_df['start_timestamp'].values[0]
            segmented_df=cm.filter_rows_by_temporal_values(
                    dataframe=aux_df,
                    time_column='start_timestamp',
                    ini_value=time_0,
                    end_value=time_0+window_analysis
                    )   
            
            verticality=segmented_df['verticality'].values
            vi=self.compute_vertical_index(verticality,screen_normalization)
            data_dict[asset]=[vi]

        self.vertical_index_df=pd.DataFrame(data_dict)


    def compute_vertical_index(self,vector_index,screen_normalization=False,screen:list[str]=[1920,1080]):
        """Calculate vertical index from a vector of 0 and 1
        Args:
            vector_index (_type_): 0 and ones int numpy array vector
        """
        if screen_normalization:
            vertical=vector_index[vector_index==1].size * screen[1] 
            horizontal=vector_index[vector_index==0].size * screen[0]
            vi=self.calculate_contrast(vertical,horizontal)
        else:
            vertical=vector_index[vector_index==1].size 
            horizontal=vector_index[vector_index==0].size 
            vi=self.calculate_contrast(vertical,horizontal)
            # print(f'vertical: {vertical}, horizontal: {horizontal},verticality: {vector_index}')
        return vi
    
    
    def  number_fixations_on_off_surface(self):
        
        data_quality_fixations=self.group_dataset['world_timestamp'].aggregate('count')
        _,index=self.filter_series_list_string(data_quality_fixations['asset'],self.annotation_list) #obtain only those values that correspond to annoatations
        data_quality_fixations=data_quality_fixations[index]

        data_quality_fixations_on_surf=data_quality_fixations.query('on_surf == True').loc[:,'asset':'world_timestamp':2]
        data_quality_fixations_on_surf.columns=['asset','fixation_on_surface']

        data_quality_fixations_off_surf=data_quality_fixations.query('on_surf == False').loc[:,'asset':'world_timestamp':2]
        data_quality_fixations_off_surf.columns=['asset','fixation_off_surface']
        self.number_fixations=data_quality_fixations_on_surf.merge(data_quality_fixations_off_surf,\
                                                                   on='asset',how='outer').fillna(0)
        
        self.data_matrix=self.data_matrix.merge(self.number_fixations,
                                                on='asset',how='inner')

    def average_saccade_distance(self,col_name:str='distance',new_col:str='mean_distance',how:str='left'):
        self.mean_saccade_distance=self.group_dataset_on_surf[col_name].mean()
        self.data_matrix=self.data_matrix.merge(self.mean_saccade_distance,on='asset',how=how)
        self.data_matrix.rename(columns={self.data_matrix.columns[-1]:new_col},inplace=True)

    def average_sacade_speed(self,col_name:str='speed',new_col:str='mean_speed',how:str='left'):
        self.mean_saccade_speed=self.group_dataset_on_surf[col_name].mean()
        self.data_matrix=self.data_matrix.merge(self.mean_saccade_speed,on='asset',how=how)
        self.data_matrix.rename(columns={self.data_matrix.columns[-1]:new_col},inplace=True)

    def average_fixation_time(self,col_name:str='duration',new_col:str='average_fixation',how:str='left'):
        self.mean_fixation_time=self.group_dataset_on_surf[col_name].mean()
        self.data_matrix=self.data_matrix.merge(self.mean_fixation_time,on='asset',how=how)
        self.data_matrix.rename(columns={self.data_matrix.columns[-1]:new_col},inplace=True)


    def total_fixation_time(self,col_name:str='duration',new_col:str='total_fixation',how:str='left'):
        self.sum_fixation_time=self.group_dataset_on_surf[col_name].sum()
        self.data_matrix=self.data_matrix.merge(self.sum_fixation_time,on='asset',how=how)
        self.data_matrix.rename(columns={self.data_matrix.columns[-1]:new_col},inplace=True)

    def group_data(self,keys:list[str]=['asset','on_surf']):
        self.group_dataset=self.fixations.groupby(keys,as_index=False)

    def group_data_on_surface(self,keys:list[str]=['asset']):
        self.group_dataset_on_surf=self.fixations.query('on_surf == True').groupby(keys,as_index=False)

    def test_data_table(self):
        if self.annotation_list.shape[0] == self.data_matrix.shape[0]:
            print('annotations are correct')
        else:
            raise Exception('Annotation and Data do noSt match')

    