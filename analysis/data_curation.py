"""Package with useful methods and classes for data 
processing and normalization
"""
import numpy as np
class Normalization():
    """
    Class for normalizing data
    """
    def __init__(self):
        """
        Constructor
        """
        pass

    def normalize(self,values,type='minmax',min_value=0,max_value=1):
        """
        Normalizes data

        Parameters
        ----------
        data : numpy array
            Data to be normalized
        min_value : float
            Minimum value of the data. Default value is 0
        max_value : float
            Maximum value of the data. Default value is 1   
        """
        if type=='minmax':
            norm_data=(max_value-min_value)*(values-np.min(values)) /(np.max(values)-np.min(values))+min_value
        if type=='z_score':
            norm_data=(values - np.mean(values))/(np.std(values))
       
        return norm_data
