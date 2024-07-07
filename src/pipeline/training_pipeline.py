# import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir,"..")
sys.path.append(src_dir)

from components.data_ingestion import DataIngestion
from config.configuration import ConfigurationManager
from loggingInfo.loggingInfo import logging
from exception.exception import customException

class TrainingPipeline:
    
    def __init__(self):
        pass
    
    def main(self):
        try:            
            config = ConfigurationManager()
            get_data_ingestion_con = config.get_data_ingestion_config()
            
            dataCon = DataIngestion(config=get_data_ingestion_con)
            dataCon.download_file()
            dataCon.extract_zip_file()
            
        except Exception as e:
            raise customException(e, sys)
        
# if __name__ == "__main__":
#     try:            
#         logging.info("<STAGE-01> Training Pipeline started.")
#         training_pipe = TrainingPipeline()
#         training_pipe.main()
#         logging.info("Training Pipeline successfully executed.")
        
#     except Exception as e:
#             raise customException(e, sys)