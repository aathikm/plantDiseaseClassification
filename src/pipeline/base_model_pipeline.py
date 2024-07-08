import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir,"..")
sys.path.append(src_dir)

from components.base_model_dev import PrepareBaseModel
from config.configuration import ConfigurationManager
from loggingInfo.loggingInfo import logging
from exception.exception import customException

class BaseModelPipeline:
    
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            get_base_model_config = config.get_prepare_base_model_config()
            
            base_model_pipe = PrepareBaseModel(get_base_model_config)
            base_model_pipe.get_base_model()
            base_model_pipe.update_base_model()
        
        except Exception as e:
            customException(e, sys)
            
STAGE_NAME2 = "Base Model Preparation"
if __name__ == "__main__":
    try:
        logging.info(f"######## <{STAGE_NAME2}> started ##########")
        base_model_pipe = BaseModelPipeline()
        base_model_pipe.main()
        logging.info(f"######## <{STAGE_NAME2}> Completed ##########")
    
    except Exception as e:
        customException(e, sys)