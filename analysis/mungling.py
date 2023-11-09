import pandas as pd

class DataMungling:

    def cut_dataframe_by_column_values(self,
                                    df:pd.DataFrame=pd.DataFrame,
                                    inital_label:str='asset',
                                    final_label:str='end_of_experiment',
                                    filter_column:str='label'):
        
        """Filter any dataframe by two column values. Cut a dataframe by two 
        reference values:
        1) An intial value by first order of appearance in the df
        2) A final value by first order of appearance in the df

        Args:
            df (pd.DataFrame): _description_
            inital_label (str, optional): name of the first label. Defaults to 'asset'.
            final_label (str, optional): name of the final label. Defaults to 'end_of_experiment'.
            filter_column (str, optional): name of the column used for filtering the df. Defaults to 'label'.

        Returns:
            pd.Dataframe: _description_
        """

        index_initial=df.loc[df[filter_column]==inital_label].index[0]
        index_final=df.loc[df[filter_column]==final_label]
        if index_final.empty:
            df_final=df.loc[index_initial::,:]
        else:
            df_final=df.loc[index_initial:index_final.index,:]
        return df_final

    def filter_series_string(self,df:pd.Series,label:str='label'):
        return df.str.contains(label) 
    
    def filter_series_list_string(self,df:pd.Series,label:list[str]):
        """Function to filfer a pd.Series by a common list of strings of 
        coincidences

        Args:
            df (pd.Series): _description_
            label (list[str]): list of strings

        Returns:
            _type_: _description_
        """
        for n,name in enumerate(label):
            if n==0:
                index=self.filter_series_string(df,name)
            else:
                index=self.filter_series_string(df,name) | index

        return df[index]