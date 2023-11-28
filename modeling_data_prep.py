# -*- coding: utf-8 -*-
"""
# Merge SFDC, Claims, Product & Engagement label data to prepare modeling data   
"""

# Import necessary python packages
import sys
import numpy as np
import pandas as pd
import logging.config
from functools import reduce

# Load configurations (parameters & constants)
import constants as CONSTANTS
from config import Config
CONF = Config()

# Create an object for FileHandler logger
# Use logger to log DEBUG, INFO, ERROR comments in file
import logging.config
logging.config.fileConfig(disable_existing_loggers=False, fname='../misc/logging.conf')
logger = logging.getLogger(__name__)



def main(pipeline):
    
    try:          
        # Read sfdc modeling data
        sfdc_data = pd.read_csv(CONSTANTS.sfdc_modeling_data)     
        # Read claims modeling data
        claims_data = pd.read_csv(CONSTANTS.claims_modeling_data)    
        # Read product name modeling data
        product_data = pd.read_csv(CONSTANTS.product_modeling_data)
        
        # Merge all data sources (Claims, SFDC, Product Name modeling data)
        # Get data for members having either or both of SFDC & CLaims data
        df_merged = claims_data.merge(sfdc_data, on='CONSISTENT_MEMBER_ID', how='outer').reset_index(drop=True)
        df_merged = df_merged.merge(product_data, on='CONSISTENT_MEMBER_ID', how='left').reset_index(drop=True) 
        df_merged['CONSISTENT_MEMBER_ID'] = df_merged['CONSISTENT_MEMBER_ID'].astype(str)
        
        # Impute placeholders for missing data
        # Impute 0 for features in CONSTANTS.na_imputation_col_list, else impute -999
        # based on logical imputation stratergy for particular feature
        df_merged.fillna(-999, inplace=True)
        list_col = CONSTANTS.na_imputation_col_list
        for col in list_col:
            df_merged[col].replace(-999, 0, inplace=True)
      
      
        if pipeline=='training':
            # Filter data on members for which we've an Engagement Score
            # Merge target label with modeling data
            target_data = pd.read_csv(CONF.engagement_score_filename)
            target_data = target_data[['CONSISTENT_MEMBER_ID', 'segment']]
            df_merged = df_merged.merge(target_data, on='CONSISTENT_MEMBER_ID', how='inner').reset_index(drop=True)
        else:
            # Filter data on members list passed in Inference batch file
            inference_input = pd.read_csv(CONF.inference_input_file)
            inference_cmid_list = list(inference_input['CONSISTENT_MEMBER_ID'].unique())
            df_merged = df_merged[df_merged['CONSISTENT_MEMBER_ID'].isin(inference_cmid_list)].reset_index(drop=True)
        
        
        # Version modeling data
        df_merged.to_csv(CONF.modeling_data_filename,index=False)
        logger.info('Aggregated claims, SFDC, member plan modeling data saved at location: '+str(CONF.modeling_data_filename)) 
        
    except Exception as my_error:
        _, _, exc_tb = sys.exc_info()
        logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')
        logger.info('Aborting modeling data preparation & pipeline run due to above error')
        sys.exit(1)
        