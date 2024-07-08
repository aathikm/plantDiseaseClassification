from src.pipeline.data_ingestion_pipeline import TrainingPipeline
from src.pipeline.base_model_pipeline import BaseModelPipeline
from src.loggingInfo.loggingInfo import logging
from src.exception.exception import customException
import sys
import os

STAGE_NAME = "TRAININ_PIPELINE"
try:            
    logging.info(f"<{STAGE_NAME}> Training Pipeline started.")
    training_pipe = TrainingPipeline()
    training_pipe.main()
    logging.info(f"<{STAGE_NAME}> successfully executed.")
        
except Exception as e:
    raise customException(e, sys)

STAGE_NAME2 = "Base Model Preparation"
try:
    logging.info(f"######## <{STAGE_NAME2}> started ##########")
    base_model_pipe = BaseModelPipeline()
    base_model_pipe.main()
    logging.info(f"######## <{STAGE_NAME2}> Completed ##########")
    
except Exception as e:
    customException(e, sys)