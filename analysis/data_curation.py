"""Package with useful methods and classes for data 
processing and normalization
"""

class Normalization():
    """
    Class for normalizing data
    """
    def __init__(self):
        """
        Constructor
        """
        pass

    def normalize(self, values,min_value=0,max_value=1):
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
        norm_data=(max_value-min_value)*(values-min(values)) /(max(values)-min(values))+min_value
        return norm_data