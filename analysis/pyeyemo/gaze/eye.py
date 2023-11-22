import pyeyemo.gaze.gaze_commons as gm
import pandas as pd
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
        self.fixations['distance']=np.insert(gm.distance_x_y(x=self.fixations[x_col],
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

    def vertical_index(self,window_analysis:float,
                       window_onset:float,
                       screen_normalization=False,
                       screen:list[str]=[1920,1080]):
        """Method to calculate vertical index.
        using the veriticaility columnn to count the total
        number of horizontal and vertical sacaddes. 
        vi=(H-V)/(H+V)

        Args:
            window_analysis (float): time range of the window of analysis
            window_onset (float): Onset of the window, to discard analysis
            screen_normalization (bool, optional): _description_. Defaults to False.
            screen (list[str], optional): _description_. Defaults to [1920,1080].
        """
        
        data_dict=dict([(key,[None]) for key in self.annotation_list])# dict with empty keys 
        vertical_index_df=pd.DataFrame()#pd.DataFrame(data_dict,index=np.arange(0,800))
        data_list=[]
        for asset in self.annotation_list:
            self.segment_df(asset,window_onset,window_analysis)
            
            verticality=self.segmented_df['verticality'].values
            vi=self.compute_vertical_index(verticality,screen_normalization)
            data_dict[asset]=[vi]

        self.vertical_index_df=pd.DataFrame(data_dict)
        self.vertical_index_df.index=[self.name] # put name of subject as index 

    def vertical_index_std(self,
                           window_analysis:float,
                           window_onset:float,
                           screen_normalization=False,
                           screen:list[str]=[1920,1080],
                           x_col:str='norm_pos_x',
                           y_col:str='norm_pos_y'):
        
        """Vertical indexm calculation using standard devaition
        (std(y)-std(x)=/(std(x)+std(y))

        Args:
            window_analysis (float): Window size of anaysis
            screen_normalization (bool, optional): _description_. Defaults to False.
            screen (list[str], optional): _description_. Defaults to [1920,1080].
        """
        data_dict=dict([(key,[None]) for key in self.annotation_list])# dict with empty keys 
        vertical_index_df=pd.DataFrame()#pd.DataFrame(data_dict,index=np.arange(0,800))
        data_list=[]
        for asset in self.annotation_list:
            self.segment_df(asset,window_onset,window_analysis)
            x_std=np.std(self.segmented_df[x_col])
            y_std=np.std(self.segmented_df[y_col])

            if screen_normalization:
               vi=self.calculate_contrast(y_std* screen[1],x_std* screen[0])
            else:
                vi=self.calculate_contrast(y_std,x_std)

            data_dict[asset]=[vi]
        self.vertical_index_std_df=pd.DataFrame(data_dict)
        self.vertical_index_std_df.index=[self.name] # put name of subject as index 


    def segment_df(self,asset,window_onset,window_analysis,time_col:str):
            aux_df=self.fixations.query(f"asset == '{asset}'") #break table by asset name
            time_0=aux_df['start_timestamp'].values[0]
            self.segmented_df=cm.filter_rows_by_temporal_values(
                    dataframe=aux_df,
                    time_column='start_timestamp',
                    ini_value=time_0+window_onset,
                    end_value=time_0+window_analysis
                    ) 
              

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


class EyeLink(Eye):

    def __init__(self,name='some_subject') -> None:
        self.name=name
        import mne

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
        self.fixations=mne.io.read_raw_eyelink(fixations_dir, preload=True, create_annotations=['blinks','messages'])
