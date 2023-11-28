import pandas as pd

class DataMungling:

    def __init__(self) -> None:
        pass

    def cut_dataframe_by_column_values(self,
                                    df:pd.DataFrame=pd.DataFrame,
                                    initial_label:str='asset',
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

        index_initial=df.loc[df[filter_column]==initial_label].index[0]
        index_final=df.loc[df[filter_column]==final_label]
        if index_final.empty:
            df_final=df.loc[index_initial::,:]
        else:
            df_final=df.loc[index_initial:index_final.index.values[0],:]
        return df_final

    def filter_series_string(self,df:pd.Series,label:str='label'):
        return df.str.contains(label,na=False) 
    
    def filter_series_list_string(self,df:pd.Series,label:list[str]):
        """Function to filfer a pd.Series by a common list of strings of 
        coincidences

        Args:
            df (pd.Series): _description_
            label (list[str]): list of strings

        Returns:
            _df_: filtered pandas series datrame
            _bool_: boolean index vector  
        """
        for n,name in enumerate(label):
            if n==0:
                index=self.filter_series_string(df,name)
            else:
                index=self.filter_series_string(df,name) | index

        return (df[index], index)
    

    def calculate_contrast(self, x,y):
        """Function to calculate michealson contrast
        Args:
            x (_np.array_): _description_
            y (_np.arry_): _description_
        """
        contrast=(x-y)/(x+y)
        return contrast

    def refactor_df_to_categorical(self,df:pd.DataFrame,col_names:list[str]): 
        """This function refactors any 2d matrix dataframe to a categorical dataframe
        It takes the values of the index and the values of the colums as catetorical 
        variables and the final value as a continous variable.

        Args:
            df (pd.DataFrame): _description_
            col_names (list[str]): [index_column_name,column_var_name,variable name]

        Returns:
            _type_: a 3d categorical matrix with the columns ordered as in col_names.
            col_names (list[str]): [index_column_name,column_var_name,variable name]
        """

        series_list=[]
        df.reset_index(inplace=True)
        df=df.rename(columns={'index':col_names[0]})

        for row in df.iterrows():
            df_aux=pd.DataFrame(row[1][1:]) #remove the index name from the series
            print(col_names[0])
            print(row[1][0])
            df_aux[col_names[1]]=row[1][0]      #add the value of the index as another column
            df_aux.reset_index(inplace=True) #remove index
            df_aux.columns=[col_names[1],col_names[2],col_names[0]] #rename columns
            series_list.append(df_aux)
            
        df_final=pd.concat(series_list)
        df_final=df_final[col_names] #rearragne column names
        return df_final