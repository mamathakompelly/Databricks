# -*- coding: utf-8 -*-
"""

@author: Suraj Gade
"""

'''
Import necessary packages
'''
from tracemalloc import start
import pandas as pd
import numpy as np
from datetime import timedelta
import multiprocessing
from functools import partial
import time
import re
import glob
import os
import sys
import logging.config
from statistics import mean
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Load configurations (parameters & constants)
from config import Config
CONF = Config()
import constants as CONSTANTS

# Create an object for FileHandler logger
# Use logger to log DEBUG, INFO, ERROR comments in file
logging.config.fileConfig(disable_existing_loggers=False, fname='../misc/logging.conf')
logger = logging.getLogger(__name__)

BUCKET_SIZE = 1000

def no_of_claims_features(claims,rx):
    '''
    Returns features related to no of claims

    Parameters
    ----------
    claims : Claims data of a given member 
    rx : Pharmacy data of given member
    
    Returns
    -------
    no_of_claims 
    pct_of_denied_claims : percentage od denied claims
    '''
    if not claims.empty:
        no_of_iop_claims = len(set(claims['LEGACY_CLAIM_ID']))
        no_of_denied_claims = len(claims[claims['CLAIM_PAYMENT_STATUS_CODE']=='D']['LEGACY_CLAIM_ID'].unique())  
    else:
        no_of_iop_claims = 0
        no_of_denied_claims = 0
        
    if not rx.empty:
        no_of_rx_claims = len(set(rx['LEGACY_CLAIM_ID']))       
    else:
        no_of_rx_claims = 0
        
    no_of_claims = no_of_iop_claims+no_of_rx_claims
    pct_of_denied_claims = np.round((no_of_denied_claims/no_of_claims),2)
    
    return no_of_claims,pct_of_denied_claims

def amount_features(claims,rx):
    '''
    Returns features related to claim amount

    Parameters
    ----------
    claims : Claims data of a given member 
    rx : Pharmacy data of given member
    
    Returns
    -------
    total_claim_amt
    total_oop_amt
    pct_of_claim_amt_bucket_low,pct_of_claim_amt_bucket_med,pct_of_claim_amt_bucket_high : percentage of number of claims falling under respective claims amount bucket
    '''
    if not claims.empty:
    
        # Filter claims with negative Paid amounts and denied claims
        # Aggregate on claim level
        claims_amt = claims[(claims['CLAIM_PAYMENT_STATUS_CODE']=='P') & (claims['PAID_AMT']>0)].groupby(by=['LEGACY_CLAIM_ID'])['PAID_AMT'].unique().reset_index()
        
        cols = ['LEGACY_CLAIM_ID','COINSURANCE_AMT','COPAY_AMT','DEDUCTIBLE_AMT']
        df_oop = claims[cols].groupby(by=['LEGACY_CLAIM_ID']).max().reset_index()
        cols = ['COINSURANCE_AMT','COPAY_AMT','DEDUCTIBLE_AMT']
        for col in cols:
            df_oop[col] = df_oop[col].apply(lambda x: x if x>0 else 0)
        df_oop['total_oop'] = df_oop[['COINSURANCE_AMT','COPAY_AMT','DEDUCTIBLE_AMT']].sum(axis=1)
        
        
    if not rx.empty:
        rx_amt = rx[(rx['PAID_AMT']>0)].groupby(by=['LEGACY_CLAIM_ID'])['PAID_AMT'].unique().reset_index()
        
        df_oop_rx = rx[['LEGACY_CLAIM_ID','COPAY_AMT','DEDUCTIBLE_AMT']].groupby(by=['LEGACY_CLAIM_ID']).max().reset_index()
        cols = ['COPAY_AMT','DEDUCTIBLE_AMT']
        for col in cols:
            df_oop_rx[col] = df_oop_rx[col].apply(lambda x: x if x>0 else 0)
        df_oop_rx['total_oop'] = df_oop_rx[['COPAY_AMT','DEDUCTIBLE_AMT']].sum(axis=1)
        
    if (not claims.empty) and (not rx.empty):
        
        amt = pd.concat([claims_amt,rx_amt]).reset_index(drop=True)
        oop = pd.concat([df_oop[['total_oop']],df_oop_rx[['total_oop']]]).reset_index(drop=True)
        
    elif claims.empty:
        amt = rx_amt
        oop = df_oop_rx[['total_oop']]
    else:
        amt = claims_amt
        oop = df_oop[['total_oop']] 
    
    
    if not amt.empty:
        amt['claim_amt'] = [sum(i) for i in amt['PAID_AMT']]

        total_claim_amt = amt['claim_amt'].sum()
        
        ### threshold for claim amount bucket
        # claim_amt_bucket_low : claims amount <= $20
        # claim_amt_bucket_low : $20 < claims amount <= $132
        # claim_amt_bucket_low : claims amount > $132
        # threshold are decided based on the distribution of claim amount in claims devlopment modeling data

        no_of_claim_amt_bucket_low = len(amt[amt['claim_amt']<=20]['LEGACY_CLAIM_ID'].unique())
        no_of_claim_amt_bucket_med = len(amt[(amt['claim_amt']>20)&(amt['claim_amt']<=132)]['LEGACY_CLAIM_ID'].unique())
        no_of_claim_amt_bucket_high = len(amt[amt['claim_amt']>132]['LEGACY_CLAIM_ID'].unique())

        total_no = no_of_claim_amt_bucket_low + no_of_claim_amt_bucket_med + no_of_claim_amt_bucket_high

        pct_of_claim_amt_bucket_low = np.round((no_of_claim_amt_bucket_low/total_no),2)
        pct_of_claim_amt_bucket_med = np.round((no_of_claim_amt_bucket_med/total_no),2)
        pct_of_claim_amt_bucket_high = np.round((no_of_claim_amt_bucket_high/total_no),2)
    else:
        total_claim_amt = 0
        pct_of_claim_amt_bucket_low = 0
        pct_of_claim_amt_bucket_med = 0
        pct_of_claim_amt_bucket_high = 0    
    
    if not oop.empty:
        total_oop_amt = oop['total_oop'].sum()
    else:
        total_oop_amt = 0
    
    return total_claim_amt,total_oop_amt,pct_of_claim_amt_bucket_low,pct_of_claim_amt_bucket_med,pct_of_claim_amt_bucket_high   
    

def time_features(claims):
    '''
    Returns features related to duration

    Parameters
    ----------
    claims : Claims data of a given member
    
    Returns
    -------
    avg_time_between_claims : Average No of days between consecutive claims over the feature creation period
    '''   
    if not claims.empty:
        df_time = claims[['LEGACY_CLAIM_ID','FROM_DATE']].drop_duplicates()
        df_time = df_time.sort_values(by=['FROM_DATE']).reset_index(drop=True)

        df_time['from_date_shift'] = df_time.FROM_DATE.shift(-1)
        df_time['time_b/w_claims'] = df_time['from_date_shift'] - df_time['FROM_DATE']
        df_time['time_b/w_claims'] = [x.days for x in df_time['time_b/w_claims']]

        avg_time_between_claims = np.round((df_time['time_b/w_claims'].mean()),2)
    else:
        avg_time_between_claims = -999
    return avg_time_between_claims


def diag_code_features(df):
    '''
    Returns features related to dignosis codes

    Parameters
    ----------
    df : Claims data of a given member
    
    Returns
    -------
    diag_dict : dictionary containing no of occurances of each dignosis code for given member in feature creation period
    '''
    
    # most prevailing dignosis codes, obtained from diag code distribution in claims devlopment modeling data
    top_codes = ['FAC016','MUS010', 'MBD005', 'MUS011','CIR007', 'END010', 'MBD002', 'FAC008']
    if not df.empty:
    
        df = df[['LEGACY_CLAIM_ID','PRINCIPAL_DIAG_CODE',
                            'SECONDARY_DIAG1_CODE','SECONDARY_DIAG2_CODE','SECONDARY_DIAG3_CODE']].drop_duplicates()

        df = df.groupby(by=['LEGACY_CLAIM_ID'])[['PRINCIPAL_DIAG_CODE','SECONDARY_DIAG3_CODE','SECONDARY_DIAG2_CODE',
                                                 'SECONDARY_DIAG1_CODE']].agg(list).reset_index()

        df['diag_list'] = df.apply(lambda x : list(set(x['PRINCIPAL_DIAG_CODE'] + x['SECONDARY_DIAG1_CODE']+ 
                                                       x['SECONDARY_DIAG2_CODE']+x['SECONDARY_DIAG3_CODE'])),axis=1)

        diag_list = []
        for row in df.itertuples():
            diag_list.extend(row.diag_list)

        count_dict = Counter(diag_list)
    
        diag_dict={}
        for code in top_codes:
            if code in count_dict.keys():
                diag_dict['diag_'+code] = count_dict[code]
            else:
                diag_dict['diag_'+code] = 0
    else:
        diag_dict={}
        for code in top_codes:
            diag_dict['diag_'+code] = 0
    
    return diag_dict


def feature_engineering(cmid_list, claims_data, rx_data, start):
    '''
    Returns the data conataining member level features calculated over lookback window (1 year time_period)

    Parameters
    ----------

    cmid_list : List of CMIDs of members
    claims_data : Entire Claims data(IP,OP,Prof) of members
    rx_data : Entire RX data(IP,OP,Prof) of members
    start : start point of current chunk in multiprocessing 

    Returns
    -------
    Saves the feature engineered data for current chunk
    '''

    cmid_list  = cmid_list[start:start+BUCKET_SIZE]

    # dataframe for saving member wise feature results
    result_data = pd.DataFrame()
    
    # For loops at different places in the script will be required
    # to iterate over each CMID for data preparation, feature construction for each CMID, etc.
    # For loops here will not be inefficient as we're already using multiprocessing (running multiple processes simultaneously for different CMID)
    for member in cmid_list:
        claims = claims_data[claims_data['CONSISTENT_MEMBER_ID']==member].reset_index(drop=True)
        rx = rx_data[rx_data['CONSISTENT_MEMBER_ID']==member].reset_index(drop=True)
        
        # Get No of claims related features
        no_of_claims,pct_of_denied_claims = no_of_claims_features(claims,rx)
        
        # Get Claim Amount related features
        total_claim_amt,total_oop_amt,pct_of_claim_amt_bucket_low,pct_of_claim_amt_bucket_med,pct_of_claim_amt_bucket_high = amount_features(claims,rx)
        
        # Get Duration related features
        avg_time_between_claims = time_features(claims)
        
        # Get dignosis code features
        diag_dict = diag_code_features(claims)
        
        # preparing data to added up as row to result Dataframe
        feature_dict ={'CONSISTENT_MEMBER_ID':member,'no_of_claims':no_of_claims,'pct_of_denied_claims':pct_of_denied_claims,
                       'total_claim_amt':total_claim_amt,'total_oop_amt':total_oop_amt,
                       'pct_of_claims_amt_bucket_low':pct_of_claim_amt_bucket_low,
                       'pct_of_claims_amt_bucket_med':pct_of_claim_amt_bucket_med,
                       'pct_of_claims_amt_bucket_high':pct_of_claim_amt_bucket_high,
                       'avg_time_between_claims':avg_time_between_claims
                      }
        
        # adding dignosis code features
        feature_dict.update(diag_dict)
        
        # Appending the row of features in result data for current member
        result_data = result_data.append(feature_dict,ignore_index=True)

        # Save feature engineered data for current chunk in multiprocessing
        result_data.to_csv(CONSTANTS.feature_engg_temp_path+'claims_feature_engg_chunk_'+str(start)+'.csv',index=False)
    

def clean_folder(folder_path):
    for file_name in glob.glob(folder_path+'*.csv'):
        os.remove(file_name)


def main():
    try: 
        # Read ICD mapping files that maps ICD codes to CCSR codes
        # Reading Static File 
        # CONSTANTS.icd_ccsr_codemapping_file file must be present at location : /data/Static_Data/
        ICD_Mappings = pd.read_excel(CONSTANTS.icd_ccsr_codemapping_file, sheet_name='in')
        logger.info('Read ICD_mapping file')
        
        ICD_Mappings = ICD_Mappings[['ICD-10-CM CODE','ICD-10-CM CODE DESCRIPTION','Default CCSR CATEGORY','Default CCSR CATEGORY DESCRIPTION']]
        ICD_Mappings['ICD-10-CM CODE'] = ICD_Mappings['ICD-10-CM CODE'].map(lambda x : x.replace('\'',''))
        ICD_Mappings['Default CCSR CATEGORY'] = ICD_Mappings['Default CCSR CATEGORY'].map(lambda x : x.replace('\'',''))
        ICD_Mappings_dict = dict(ICD_Mappings[['ICD-10-CM CODE','Default CCSR CATEGORY']].values)
        
        logger.info('Reading Claims raw data')
        claims_data = pd.read_csv(CONF.claims_acquisition_file)
        logger.info('Reading Claims data completed')
        rx_data = pd.read_csv(CONF.pharmacy_data_file)
        logger.info('Reading RX data completed')
        
        # Get member list for feature creation
        claims_cmid = list(claims_data['CONSISTENT_MEMBER_ID'].unique())
        rx_cmid = list(rx_data['CONSISTENT_MEMBER_ID'].unique())
        all_cmid = list(set(rx_cmid)|set(claims_cmid))
        
        # Convert date columns to datetime datatype
        claims_data['FROM_DATE'] = pd.to_datetime(claims_data['FROM_DATE'])
        claims_data['THRU_DATE'] = pd.to_datetime(claims_data['THRU_DATE'])
        claims_data['ADMIT_DATE'] = pd.to_datetime(claims_data['ADMIT_DATE'])
        claims_data['DISCHARGE_DATE'] = pd.to_datetime(claims_data['DISCHARGE_DATE'])
        
        rx_data['PRESCRIPTION_FILLED_DATE'] = pd.to_datetime(rx_data['PRESCRIPTION_FILLED_DATE'])

        # Remove initial spaces in column values from data
        for column in claims_data.columns:
            if claims_data[column].dtype == object:
                claims_data[column] = claims_data[column].str.strip()
        
        for column in rx_data.columns:
            if rx_data[column].dtype == object:
                rx_data[column] = rx_data[column].str.strip()
        
        # Replacing blank values in data with np.nan 
        claims_data = claims_data.replace(r'^\s*$',np.nan,regex=True)
        
        # Replacing values having string 'NA' with np.nan
        claims_data = claims_data.replace('NA',np.nan)
        
        # Mapping ICD dignosis codes to CCSR codes
        claims_data['SECONDARY_DIAG1_CODE'] = claims_data['SECONDARY_DIAG1_CODE'].apply(lambda x: ICD_Mappings_dict.get(str(x).replace('.','').strip()))
        claims_data['SECONDARY_DIAG2_CODE'] = claims_data['SECONDARY_DIAG2_CODE'].apply(lambda x: ICD_Mappings_dict.get(str(x).replace('.','').strip()))
        claims_data['SECONDARY_DIAG3_CODE'] = claims_data['SECONDARY_DIAG3_CODE'].apply(lambda x: ICD_Mappings_dict.get(str(x).replace('.','').strip()))
        claims_data['PRINCIPAL_DIAG_CODE'] = claims_data['PRINCIPAL_DIAG_CODE'].apply(lambda x: ICD_Mappings_dict.get(str(x).replace('.','').strip()))
        
        ## combining all the temporary files generated.
        logger.info('Combining all the temporary files generated in multiprocessing')
        glued_data = pd.DataFrame()
        
        for file_name in glob.glob(CONSTANTS.feature_engg_temp_path+'*.csv'):
            x= pd.read_csv(file_name)
            glued_data = pd.concat([glued_data,x],axis=0)
            os.remove(file_name) 
        glued_data.to_csv(CONSTANTS.claims_modeling_data,index=False)

        #Cleaning folder
        #clean_folder(CONSTANTS.feature_engg_temp_path)
        #logger.info('Cleaning folder : '+str(CONSTANTS.feature_engg_temp_path))

        # Multiprocessing started and mapping members to feature_engineering function
        logger.info('Claims Feature Creation started in Multiprocessing')
        chunks  = [x for x in range(0,len(all_cmid),BUCKET_SIZE)]
        pool = multiprocessing.Pool()
        func = partial(feature_engineering, all_cmid, claims_data, rx_data)
        l = pool.map(func, chunks)
        pool.close()
        pool.join()
        
        logger.info('Claims Feature creation completed')
        logger.info('Claims modeling data saved at location : '+str(CONSTANTS.claims_modeling_data))
        
    except Exception as my_error:
        _, _, exc_tb = sys.exc_info()
        logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')
        logger.info('Claims feature engineering failed, exiting')
        sys.exit(1)
