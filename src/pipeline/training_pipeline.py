# import logging
import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir,"..")
sys.path.append(src_dir)

from components.model_training import Training
import tensorflow as tf
from config.configuration import ConfigurationManager
from loggingInfo.loggingInfo import logging
from exception.exception import customException
from components.model_whole_trian import PrepareBaseModel

STAGE_NAME3 = "Trianing_Pipeline"

class TrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            get_base_model_config = config.get_prepare_base_model_config()
            get_model_train_config = config.get_training_config()
                    
            base_model_pipe = PrepareBaseModel(get_base_model_config, get_model_train_config)
            # model = base_model_pipe.get_base_model()
            base_model_pipe.train(
                model = tf.keras.applications.InceptionV3,
                    classes=get_base_model_config.params_classes,
                    freeze_all=True,
                    freeze_till=None,
                    learning_rate=get_base_model_config.params_learning_rate
            )
        
        except Exception as e:
            customException(e, sys)
            
if __name__ == "__main__":
    logging.info(f"######## <{STAGE_NAME3}> started ###########")
    training_pipe = TrainingPipeline()
    training_pipe.main()
    logging.info(f"######## <{STAGE_NAME3}> completed ###########")