import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import urllib.request as request
from zipfile import ZipFile
from numpy import full
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from pathlib import Path
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir,"..")
sys.path.append(src_dir)

from config.configuration import PrepareBaseModelConfig, ConfigurationManager


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            classes = self.config.params_classes
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024,activation='relu')(x) 
        x = Dense(512,activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        prediction= Dense(38, activation = 'softmax')(x)
        full_model = Model(inputs= model.input, outputs= prediction)
        
        full_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy']
            # optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            # loss=tf.keras.losses.CategoricalCrossentropy(),
            # metrics=["accuracy"]
        )

        full_model.summary()
        logging.info(full_model.summary())
        return full_model
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        
if __name__ == "__main__":
    config = ConfigurationManager()
    get_base_model_config = config.get_prepare_base_model_config()
            
    base_model_pipe = PrepareBaseModel(get_base_model_config)
    base_model_pipe.get_base_model()
    base_model_pipe.update_base_model()
 