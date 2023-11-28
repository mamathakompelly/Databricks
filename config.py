# 
"""
load the configurations & other parameters
"""

# Import basic Python libraries
import os
import pickle
import logging.config
from ast import literal_eval as le
from configparser import ConfigParser

# Create an object for FileHandler logger
# Use logger to log DEBUG, INFO, ERROR comments in file
#logging.config.fileConfig(disable_existing_loggers=False, fname='../misc/logging.conf')
#logger = logging.getLogger(__name__)

# Read configurations from config.ini file
conf = ConfigParser()
conf.read('../misc/config.ini')


class Config:
    """
        Class contains utility functions to check for & create required directories,
        load configurations & other parameters
    """

    def __init__(self):
        """" Initialize required constants & data here """

        os.chdir("../")
        # Create the logs directory for saving Time-based logs
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Create the data directory, put all your data here
        if not os.path.exists('data/Raw'): 
            os.makedirs('data/Raw')
        
        # Create the model_repo directory to maintain model versions
        if not os.path.exists('model_artifacts'):
            os.makedirs('model_artifacts')
        
        if not os.path.exists('model_artifacts/models'):
            os.makedirs('model_artifacts/models')
        if not os.path.exists('model_artifacts/modelling_data'):
            os.makedirs('model_artifacts/modelling_data')
        if not os.path.exists('model_artifacts/encoders'):
            os.makedirs('model_artifacts/encoders')

        # Create the results directory to store Inference results
        if not os.path.exists('predictions'):
            os.makedirs('predictions')

        # Create the report directory if to maintain evaluation metrics report
        if not os.path.exists('reports'):
            os.makedirs('reports')
        if not os.path.exists('reports/plots'):
            os.makedirs('reports/plots')
        
   
    modeling_data_folder = 'model_artifacts/modelling_data/'
    raw_data_folder = 'data/Raw/'

   # Folder location to save model artifacts
    model_filename    = 'model_artifacts/models/xgb_model1.pkl'
    encoder_filename  = 'model_artifacts/encoders/onehotencoding.pkl'
   # Model classifier names
    model_name = 'randomforest'



    # Input file names
    # Input file to Inference pipeline 
    # File must have columns : ['CONSISTENT_MEMBER_ID']
    inference_input_file = 'data/Inference_Input_File.csv'


    # Output file names

    # Inference prediction output folder
    inference_output_folder = 'predictions/'

    # Inference final output file with predictions
    # final_prediction_file = Inference_Predictions.csv

if __name__ == '__main__':
    conf = Config()

    print(Config.inference_input_file)

