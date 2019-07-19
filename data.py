import pandas as pd
import numpy as np 

pd.options.display.max_columns = None
pd.options.display.float_format = '{:.2f}'.format


class DataFactory:

    '''
    description : Data handling and manipulation class
    syntax : None
    '''
    pass

    def __init__(self, dataframe , cat_cols=[] , num_cols=[], target_cols=None):
        self._dataframe = dataframe
        self._cat_cols  = cat_cols
        self._num_cols  = num_cols
        self._target_cols = target_cols
        self._cleaned_dataframe = self._dataframe



    # basic operations
    def show_data(self):
        print(self._cleaned_dataframe[:2])

    def format_colnames(self):
        self._cleaned_dataframe.columns = [col.lower() for col in self._dataframe.columns]

    def change_dates(self , cols = [], fmt="%Y%m%d"):
        '''
        description : change cols to pandas datetime from a given format,
        syntax : None
        '''
        for col in cols:
            self._cleaned_dataframe[col] = pd.to_datetime(self._dataframe[col], format = fmt)

    def change_category(self, cols= []):
        '''
        description : Change to 
        syntax : change_category(['x','y','z'])
        '''
        for col in cols:
            self._cleaned_dataframe[col] = self._dataframe[col].astype('category')

    def field_cleanser(self, cols=[], symbols=[]):
        '''
        description : Cleanse field from the unwanted symbol
        syntax : None

        '''
        pass

    def replace_inf(self, cols=[]):
        '''
        description : replace the -inf and inf with max value/ 0 of the column
        syntax : None
        '''
        pass

    # Process the Categorical variables
    def check_lt5cat(self, n=5):
        '''
        Configurable to chang the no. of levels of categorical variables to consider
        '''
        pass

    def check_distribution(self, is_cat=True):
        '''
        check for the event distribution
        if 90% skewed - leave else keep
        tackles:
            - numeric
            - categorical
        '''
        pass

    def categorical_to_continuous(self):
        '''
        For columns with less than 5 in categorical items
        '''
        pass

    # def create_index(self):
    #     '''
    #     Create index for cat variable for more than N levels.
    #     Mention level of mean
    #     '''
    #     pass
        
    # Processing the numeric variables
    def treat_outliers(self):
        '''
        description : capping the lower and upper bracket for 5% and 95%
        syntax : None
        '''
        pass

    def plot_density(self):
        '''
        description : Plot density to check if there is skewness and binning is req
        syntax : None
        '''
        pass

    
    def cont_custom_bins(self):
        '''
        description : Create custom bins for the continuos variables
        syntax : None
        '''
        pass    

    def scaling_continuous(self):

        '''
        description : Scaling the continuous bins
        syntax : None
        '''
        pass










