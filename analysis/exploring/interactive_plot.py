#%%
import time
import pylab as pl
from IPython import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the data

vertical_data=r'../vision_ejes/data/gaze_positions_verticales.csv'
horizontal_data=r'../vision_ejes/data/gaze_positions - copia.csv'

vertical_test=pd.read_csv(vertical_data,delimiter=';',decimal=',')
horizontal_test=pd.read_csv(horizontal_data,delimiter=';',decimal=',')

print(vertical_test.head())


# for timestamp,_ in enumerate(vertical_test['gaze_normal0_x']):
#     if timestamp%10!=0:
#         continue
#     plt.plot(vertical_test['gaze_normal0_x'][timestamp],vertical_test['gaze_normal0_y'][timestamp],'o',color='black')
#     plt.title('time= {}'.format(timestamp))
#     time.sleep(0.005)
#     plt.show()


#%%

import analysis.data_curation as cur
normalization=cur.Normalization()
norm_data=normalization.normalize(vertical_test['gaze_normal0_y'])
print(norm_data)
print(f'Normalized Values min value={min(norm_data)}, max value= {max(norm_data)}')
print(type(norm_data))

data_normalized=vertical_test.copy()
for column in vertical_test.columns:
    data_normalized[column]=normalization.normalize(vertical_test[column],5,10)

# %%
data_normalized.describe()
# %%
