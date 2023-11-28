# -*- coding: utf-8 -*-
"""
Data Preparation & Engagement Score calculation corresponding to
members digital behavior data aggregated over a given time period
"""

__author__ = "Rohit Kr Singh"

# Import Python libraries
import os
import glob
import sys
import time
import numpy as np
import pandas as pd
import logging.config

# Load user defined helper methods
from data_prep.engagement_data.engagement_channels import portal_data, wellness_event_data, smt_used_data, text_data, email_data, comm_pref_data, app_utilisation_data
from data_prep.engagement_data import aggregate_engagement_data

# Load configurations (parameters & constants)
import constants as CONSTANTS
from config import Config
CONF = Config()

# Create an object for FileHandler logger
# Use logger to log DEBUG, INFO, ERROR comments in file
logging.config.fileConfig(disable_existing_loggers=False, fname='../misc/logging.conf')
logger = logging.getLogger(__name__)



def clean_folder(folder_path):
    for file_name in glob.glob(folder_path + '*.csv'):
        os.remove(file_name)

        
def main():
    
    try:
        # Clean data directory 
        logger.info('Cleaning data/target_data/ data directory for .csv files')
        clean_folder(CONF.cur_dir + 'target_data/')

        ## Get Aggregated Portal, Virgin Pulse Wellness App Event, Text,
        ## E-mail, Communication Preference & Mobile App Utilisation data

        # Calculation for portal features is done at a daily
        # level & then aggregated over a given time period
        portal_data.main()

        # Calculation for Wellness Event features is done at monthly
        # level & then aggregated over a given time period
        wellness_event_data.main()
        
        # Calculate Secure Messaging Tool usage features
        smt_used_data.main()

        # Text & E-mail data is provided as well as
        # feature engineered at an yearly level
        text_data.main()
        email_data.main()

        # Mobile App Utilisation & Communication Preference data are 
        # one-time dataset, no aggregation over a  period is needed
        app_utilisation_data.main()
        comm_pref_data.main()
        
        ## Merge data acrooss engagement channels & calculate
        ## engagement score corresponding to members digital
        ## behavior aggregated over a given time period
        aggregate_engagement_data.main()
        
    except Exception as my_error:
        _, _, exc_tb = sys.exc_info()
        logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')
        logger.info('Failed to locate engagement channels .py failed, please keep .py file names & location unchanged, exiting')
        sys.exit(1)
