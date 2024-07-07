from src.pipeline.training_pipeline import TrainingPipeline
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