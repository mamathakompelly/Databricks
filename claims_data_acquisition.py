# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:11:49 2021

@author: satyam.kumar
"""

'''
Import necessary packages
'''
from lib2to3.pgen2.pgen import DFAState
import os
import gc
import sys
import glob
import snowflake.connector
import datetime
import pandas as pd
import logging.config
import multiprocessing
from functools import partial

from data_prep import sql_queries as q
from data_prep.modeling_data import claims_feature_engineering

# Load configurations (hyperparameters & constants)
from config import Config
conf = Config()
import constants as c

# Create an object for FileHandler logger
# Use logger to log DEBUG, INFO, ERROR comments in file
logging.config.fileConfig(disable_existing_loggers=False, fname='../misc/logging.conf')
logger = logging.getLogger(__name__)


class DataExtraction:
    
    def __init__(self, server, database, username, password):
        try:
            # Establish connection with the SQL DB
            self.cnxn = snowflake.connector.connect(
            insecure_mode=True,
            user='BIE_DEVELOPER',
            password='BIE2021',
            account='tl53752.east-us-2.azure',
            warehouse='BIE_WH',
            database='BIE',
            schema='DATAHUB'
        )
            self.cursor = self.cnxn.cursor()
        except Exception as e:
            logger.info(__name__ + ' : ' + ' Connection to SQL database failed with error: '+ str(e))
            sys.exit(1)

        
    def extractData(self, startDate, endDate, FileInfo, cmid_list, start):
        '''
        Extract Claims data and save as csv File
        
        Parameters
        ----------
        startDate : Start date of Inpatient data
        endDate : End date of Inpatient data
        FileInfo : File Name

        Returns
        -------
        None.

        '''
        saveFileName = self.saveAs(FileInfo, startDate, endDate, start)
        DateMonthWise = self.getDateList(FileInfo, startDate, endDate, splitInDays=30)
        
        for i in range(len(DateMonthWise)-1):
            
            st = DateMonthWise[i]
            end = DateMonthWise[i+1]
            
            query = self.queryFunc(FileInfo, st, end, cmid_list)
            
            df = pd.read_sql(query, self.cnxn)
            logger.debug('File Extracted from '+str(st)+' to '+str(end))

            if FileInfo=='Inpatient':
                self.saveClaimsFile(df, c.claims_path + conf.temp_ip_path + saveFileName)
            elif FileInfo=='Outpatient':
                self.saveClaimsFile(df, c.claims_path + conf.temp_op_path + saveFileName)
            elif FileInfo=='Prof':
                self.saveClaimsFile(df, c.claims_path + conf.temp_prof_path + saveFileName)
            elif FileInfo=='RX':
                self.saveClaimsFile(df, c.claims_path + conf.temp_rx_path + saveFileName)
            else:
                pass

        logger.info('File extracted from '+str(startDate)+' to '+str(endDate)+' & saved at '+ c.claims_path)

        
    def saveClaimsFile(self, df, filePath):
        '''
        Save Claims File at filePath location
        
        Parameters
        ----------
        df : claims dataframe to save
        filePath : save path location
        
        Returns
        -------
        None.
        '''
        #check if file present at filePath location
        if os.path.exists(filePath):
            # If file present, then append the dataframe
            with open(filePath, 'a', newline='') as f:
                df.to_csv(f, header=False, index=False)
        else:
            # If file not present, then save new file
            df.to_csv(filePath, index=False)        

    
    def getDateList(self, FileInfo, startDate, endDate, splitInDays=30):
        
        '''
        Returns list of dates from startDate till endDate in split of splitInDays days
        
        Parameters
        ----------
        FileInfo : IP/OP/Prof
        startDate : start date
        endDate : end date
        splitInDays : split in days
        
        Returns
        -------
        DateMonthWise : List of dates from startDate till endDate
        '''
        
        # Convert string dates to standard datetime format
        stDate = datetime.datetime(year=int(startDate[:4]), month=int(startDate[4:6]), day=int(startDate[6:]))
        endDate = datetime.datetime(year=int(endDate[:4]), month=int(endDate[4:6]), day=int(endDate[6:]))
            
        tempDate = stDate
        DateMonthWise = []
        while(tempDate<endDate):
            DateMonthWise.append(tempDate.strftime("%Y")+""+tempDate.strftime("%m")+""+tempDate.strftime("%d"))
            tempDate = tempDate+datetime.timedelta(days=splitInDays)
        DateMonthWise.append(endDate.strftime("%Y")+""+endDate.strftime("%m")+""+endDate.strftime("%d"))
        
        return DateMonthWise


    def queryFunc(self, FileInfo, st, end, cmid_list):
        
        if FileInfo=='Inpatient':
            return q.get_inpatient_query(cmid_list)
        elif FileInfo=='Outpatient':
            return q.get_outpatient_query(cmid_list)
        elif FileInfo=='Prof':
            return q.get_prof_query(cmid_list)
        elif FileInfo=='RX':
            return q.get_pharmacy_query(cmid_list)
        else:
            Exception("Error in getting SQL Queries, Claims Type can be [Inpatient, Outpatient, Prof, RX]")
        return ""
        

    def saveAs(self, filename, startDate, endDate, start):
        return filename+'_'+startDate+'_to_'+endDate+'_chunk_'+start+'.csv'



def data_extraction_multiprocessing(cmid_list, initialDate, finalDate, start):
    '''
    Extract and save IP, OP, Prof Claims data for CMIDs in cmid_list in date range initialDate and finalDate

    Parameters
    ----------
    cmid_list : List of CMID for which data to be extracted
    initialDate : start date
    finalDate : end date
    '''

    # Create an instance of Data Extraction class
    dataExt = DataExtraction(c.server, c.database, c.username, c.password)
    logger.info('Data Extraction Instance Created')

    cmid_list = str(tuple(cmid_list[start:start + conf.bucket_size_claims]))

    # Extract Claims data from SQL database
    logger.info("Inpatient Data Extraction Instance Created for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))
    dataExt.extractData(initialDate, finalDate, 'Inpatient', cmid_list, str(start))
    logger.info("Inpatient Data Extraction Completed for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))

    logger.info("Outpatient Data Extraction Instance Created for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))
    dataExt.extractData(initialDate, finalDate, 'Outpatient', cmid_list, str(start))
    logger.info("Outpatient Data Extraction Completed for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))

    logger.info("Prof Data Extraction Instance Created for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))
    dataExt.extractData(initialDate, finalDate, 'Prof', cmid_list, str(start))
    logger.info("Prof Data Extraction Completed for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))

    logger.info("RX Data Extraction Instance Created for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))
    dataExt.extractData(initialDate, finalDate, 'RX', cmid_list, str(start))
    logger.info("RX Data Extraction Completed for idx " + str(start) + " to idx " + str(start + conf.bucket_size_claims))


def create_member_sets():
    try:
        save_path = c.claims_path
        usecols_IPOP = c.usecols_IPOP
        usecols_Prof = c.usecols_Prof
        usecols_RX = c.usecols_RX

        logger.info("Reading IP Data...")
        dfInPatient = pd.read_csv(save_path + 'inpatient.csv', usecols=usecols_IPOP, dtype=str,
                                  skipinitialspace=True).drop_duplicates()
        logger.info("Reading OP Data...")
        dfoutPatient = pd.read_csv(save_path + 'outpatient.csv', usecols=usecols_IPOP, dtype=str,
                                   skipinitialspace=True).drop_duplicates()
        logger.info("Reading Prof Data...")
        dfProf = pd.read_csv(save_path + 'prof.csv', dtype=str, usecols=usecols_Prof,
                             skipinitialspace=True).drop_duplicates()
        logger.info("Reading Rx Data...")
        dfRX = pd.read_csv(save_path + 'RX.csv', dtype=str, usecols=usecols_RX,
                             skipinitialspace=True).drop_duplicates()

        # Adding Claims Type Column
        dfInPatient['Type'] = 'IP'
        dfoutPatient['Type'] = 'OP'
        dfProf['Type'] = 'Prof'
        dfRX['Type'] = 'RX'

        dfProf = dfProf.rename(columns={'HCPCS_PROC_CODE': 'DETAIL_PROC_CODE'})
        data = pd.concat([dfInPatient, dfoutPatient, dfProf, dfRX])
        logger.info("Concatenating IP, OP, Prof Claims Data Completed")

        del dfProf; del dfInPatient; del dfoutPatient
        gc.collect()
        logger.info('Deleting IP, OP, Prof Claims Data Frames and collecting garbage memory')

        data.to_csv(conf.claims_acquisition_file, index=False)
    except Exception as my_error:
        _, _, exc_tb = sys.exc_info()
        logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')
        sys.exit(1)


def clean_folder(folder_path):
    for file_name in glob.glob(folder_path+'*.csv'):
        os.remove(file_name)



def main(pipeline):
    try:
        logger.info('Cleaning folder : '+ c.claims_path)
        clean_folder(c.claims_path)
        
        if pipeline=='training':
            logger.info("Claims data extraction for training")
            initial_date = conf.initial_date
            final_date = conf.final_date
        
            # Get member CMIDs list 
            df = pd.read_csv(conf.engagement_score_filename)
            print(conf.engagement_score_filename)
        else:
            logger.info("Claims data extraction for inference")
            initial_date = str(datetime.date.today() - datetime.timedelta(days=conf.lookback_window)).replace('-','')
            final_date = str(datetime.date.today()).replace('-','')
            print(conf.inference_input_file)
            # Reading inference_input_file from location '../data/Inference_Input_File.csv'
            # File must have columns : ['CONSISTENT_MEMBER_ID']         
            df = pd.read_csv(conf.inference_input_file)
                   

        df['CONSISTENT_MEMBER_ID'] = df['CONSISTENT_MEMBER_ID']
        cmid_list = list(df['CONSISTENT_MEMBER_ID'].unique())
        # Multiprocessing: Claims data extraction
        logger.info('Multiprocessing started')
        chunks = [x for x in range(0, len(cmid_list), conf.bucket_size_claims)]
        pool = multiprocessing.Pool()
        func = partial(data_extraction_multiprocessing, cmid_list, initial_date, final_date)
        l = pool.map(func, chunks)
        pool.close()
        pool.join()

        logger.info('Merging Claims Files')
        folder_dict = {c.temp_ip_path: 'inpatient', c.temp_op_path: 'outpatient',
                       c.temp_prof_path: 'prof', c.temp_rx_path: 'RX'}
        for key in folder_dict.keys():
            glued_data = pd.DataFrame()
            for file_name in glob.glob(str(key) + '*.csv'):
                x = pd.read_csv(file_name)
                glued_data = pd.concat([glued_data, x], axis=0).reset_index(drop=True)
                os.remove(file_name)

            glued_data.to_csv(c.claims_path + str(folder_dict[key]) + '.csv', index=False)
            logger.info('Merging Claims files completed for :' + str(folder_dict[key]))


        # Call helper methods to combine member file creation for IP, OP, Prof data
        create_member_sets()
        # Feature Engineer claims data to prepare NPS modeling data for claims
        claims_feature_engineering.main()
    
    except Exception as my_error:
        _, _, exc_tb = sys.exc_info()
        logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')
        sys.exit(1)
