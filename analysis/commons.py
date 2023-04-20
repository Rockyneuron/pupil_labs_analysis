

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gaze_eyes(x_0,y_0,x_1,y_1):

    """Function to plaot gaze of both eyes
    """
    # sns.set_theme(style='darkgrid')
    fig,ax=plt.subplots(1,)
    ax.plot(x_0,y_0,'.')
    ax.plot(x_1,y_1,'.')
    ax.legend(['Right eye','Left eye'])
    ax.set_title('Vertical Test')
    ax.grid()
    ax.set_xlabel('x coordinates')
    ax.set_ylabel('y coordinates')
    fig.set_figwidth(10)

def plot_pupillometry_both_eyes(df_left_eye,df_right_eye):
    fig2, ax2=plt.subplots(1,1)
    ax2.plot(df_left_eye['pupil_timestamp'],df_left_eye['diameter_3d'])
    ax2.plot(df_right_eye['pupil_timestamp'],df_right_eye['diameter_3d'])
    ax2.legend(['left eye', 'right eye'])
    ax2.set_xlabel('timestamps (s)')
    ax2.set_ylabel('diameter (mm)')

def extract_annotations_timestamps(annotation,annotation_col,dataframe):
    """Function to extract timesatmps from annotations table of pupil core
       It basically serches for the annotation field in a datraframe and returns
       the rows of the annotation and of the next event

    Args:
        annotation (_str_): string of the annotation of interest
        annotation_col(_str_): column name of the annotation
        dataframe: pandas dataframe with annotations
    """
    index_stim=dataframe[annotation_col]==annotation
    index_final=dataframe.index[index_stim]+1

    start_row=dataframe[index_stim]
    end_row=dataframe.iloc[index_final]

    return start_row, end_row, index_final-2

def filter_rows_by_temporal_values(dataframe,time_column,ini_value,end_value):
    """Common funtion to return a segment of a dataframe filtered by the timestamps
      greater than or equal to a column value and less than another timestamp values.

    Args:
        dataframe (_pandas dataframe_): dataframe of interest 
        time_column (_type_): column to use for filtering
        ini_value (_type_): initial temporal value
        end_value (_type_): final temporal value
    """
        
    segmented_df=dataframe[
        (dataframe[time_column]>=ini_value)&
        (dataframe[time_column]< end_value)
    ]
    return segmented_df

def do_heatmap(image,gaze_on_surf_x,gaze_on_surf_y):
    from scipy.ndimage.filters import gaussian_filter

    grid = image.shape[0:2] # height, width of the loaded image
    heatmap_detail = 0.02 # this will determine the gaussian blur kerner of the image (higher number = more blur)
    # flip the fixation points
    # from the original coordinate system,
    # where the origin is at botton left,
    # to the image coordinate system,
    # where the origin is at top left
    gaze_on_surf_y = 1 - gaze_on_surf_y

    # make the histogram
    hist, x_edges, y_edges = np.histogram2d(
        gaze_on_surf_y,
        gaze_on_surf_x,
        range=[[0, 1.0], [0, 1.0]],
        bins=grid
    )

    # gaussian blur kernel as a function of grid/surface size
    filter_h = int(heatmap_detail * grid[0]) // 2 * 2 + 1
    filter_w = int(heatmap_detail * grid[1]) // 2 * 2 + 1
    heatmap = gaussian_filter(hist, sigma=(filter_w, filter_h), order=0)

    # display the histogram and reference image
    print("Cover image with heatmap overlay")
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off');