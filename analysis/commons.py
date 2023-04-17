

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
