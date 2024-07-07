import sys
import os

from contourpy import contour_generator
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)

from constants import *
from config.entity import DataIngestionConfig
from loggingInfo.loggingInfo import logging
import gdown
from utils.utils import read_yaml, create_directories

import logging
import zipfile

from exception.exception import customException

# class ConfigurationManger:
#     def __init__(self, config_file_path = CONFIG_FILE_PATH, params_file_path = PARAMS_FILE_PATH):
#         self.config_filePath = config_file_path
#         self.params_filePath = params_file_path
        
#         self.config = read_yaml(config_file_path)
#         self.params = read_yaml(params_file_path)
    
#     def get_data_ingestion_config(self):
#         config = self.config.data_ingestion
        
#         create_directories([config.root_dir])
        
#         data_ingestion_config = DataIngestionConfig(
#             root_dir = config.root_dir,
#             source_URL = config.source_URL,
#             local_data_file = config.local_data_file,
#             unzip_dir = config.unzip_dir
#         )
        
        # return(data_ingestion_config)
    

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise customException(e, sys)
    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            
            
# if __name__ == "__main__":
#     from config.configuration import ConfigurationManager
    
#     config = ConfigurationManager()
#     get_data_ingestion_con = config.get_data_ingestion_config()
    
#     dataCon = DataIngestion(config=get_data_ingestion_con)
#     dataCon.download_file()
#     dataCon.extract_zip_file()
    