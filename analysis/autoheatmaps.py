from scipy.ndimage.filters import gaussian_filter
import pandas as pd 
import numpy as np
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import pyxdf
import matplotlib
# matplotlib.use('QtAgg')
import commons as cm



def main():

    recording_location = r"C:\Users\Bolo\Desktop\Laboratorio\incipit\data\pupil_emotibit\ES0001_S002_PC_EM\data\pupil_labs\002"

    def print_file_structure(startpath):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in sorted(files):
                print(f'{subindent}{f}')
    print_file_structure(recording_location)

    exported_gaze_csv = os.path.join(recording_location, 'exports', '000', 'surfaces','gaze_positions_on_surface_Surface_1.csv')
    gaze_pd_frame = pd.read_csv(exported_gaze_csv)
    annotations_csv = os.path.join(recording_location, 'exports', '000', 'annotations.csv')
    annotations_pd = pd.read_csv(annotations_csv)

    image_location="../../../../data/pupil_emotibit/ES0001_S002_PC_EM/images_order"
    print_file_structure(image_location)

    images=[]
    with open(image_location+'/assets.txt','r') as f:
        for image in f:
            images.append(image.replace('\n',''))
        f.close()

    image_order=os.listdir(image_location)
    image_order.remove('assets.txt')
    image_order.sort(key=lambda x: int(x[x.index('_')+1:x.index('.tif')]))

    conf_thr=0.95
    gaze_on_surf=gaze_pd_frame[(gaze_pd_frame.on_surf==True)&(gaze_pd_frame.confidence >conf_thr)]

    for im,im_order in zip(images, image_order):

        initial_anotation,end_anotation=cm.extract_annotations_timestamps(im,'label',annotations_pd)

        gaze_on_surf_im=cm.filter_rows_by_temporal_values(
            dataframe=gaze_on_surf,
            time_column='gaze_timestamp',
            ini_value=initial_anotation['timestamp'].values[0],
            end_value=end_anotation['timestamp'].values[0]
        )

        image=plt.imread(image_location+'/'+im_order)
        cm.do_heatmap(image,gaze_on_surf_im['x_norm'],gaze_on_surf_im['y_norm'])
        plt.savefig('heatmaps/'+im_order)
        plt.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Killed by user')
