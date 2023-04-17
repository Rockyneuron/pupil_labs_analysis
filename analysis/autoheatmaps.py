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
matplotlib.use('QtAgg')


def do_heatmap(image,gaze_on_surf_x,gaze_on_surf_y):
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
        print(im,im_order)
        index_stim=annotations_pd['label']==im
        index_final=annotations_pd.index[index_stim]+1

        value=annotations_pd[index_stim]['timestamp'].values[0]
        value_final=annotations_pd.iloc[index_final]['timestamp'].values[0]
        gaze_on_surf_im=gaze_on_surf[
            (gaze_on_surf['gaze_timestamp']>value)&
            (gaze_on_surf['gaze_timestamp']<value_final)
        ]
        image=plt.imread(image_location+'/'+im_order)
        do_heatmap(image,gaze_on_surf_im['x_norm'],gaze_on_surf_im['y_norm'])
        plt.savefig('heatmaps/'+im_order,dpi=80)
        plt.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Killed by user')
