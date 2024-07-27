import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pyexpat import model
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

from config.configuration import PrepareBaseModelConfig, ConfigurationManager, TrainingConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig, config1: TrainingConfig):
        self.config = config
        self.config1 = config1

    
    def get_base_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            classes = self.config.params_classes
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
        
        return model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    
    def train(self, model, classes, freeze_all, freeze_till, learning_rate):
        model = model(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            classes = self.config.params_classes
        )
        
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024,activation='relu')(x) 
        x = Dense(512,activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        prediction= Dense(38, activation = 'softmax')(x)
        model = Model(inputs= model.input, outputs= prediction)
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # full_model.summary()
        img_preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.5, 1.5],
            channel_shift_range=10,
            vertical_flip=True,
            validation_split=0.2,
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=True,   # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=True,   # divide each input by its std
        )

        train_data_gen = img_preprocessor.flow_from_directory(self.config1.training_data,
                                                              target_size=self.config1.params_image_size[:-1],
                                                              batch_size=self.config1.params_batch_size,
                                                              subset='training',
                                                              class_mode='categorical', shuffle=True)

        val_data_gen = img_preprocessor.flow_from_directory(self.config1.training_data,
                                                            target_size=self.config1.params_image_size[:-1],
                                                            batch_size=self.config1.params_batch_size,
                                                            subset='validation', 
                                                            class_mode='categorical', shuffle=True)

        self.steps_per_epoch = train_data_gen.samples // train_data_gen.batch_size
        self.validation_steps = val_data_gen.samples // val_data_gen.batch_size
        
        logging.info(f"steps per epoch: {self.steps_per_epoch}")
        logging.info(f"updated base model path: {self.config1.updated_base_model_path}")
        logging.info(f"The train label indices of each classes: {train_data_gen.class_indices}")
        logging.info(f"The val label indices of each classes: {val_data_gen.class_indices}")
        
        # model = tf.keras.models.load_model(
        #     self.config1.updated_base_model_path
        # )

        logging.info(f"model summary: {model.summary()}")
        history = model.fit(
            x = train_data_gen,
            validation_data = val_data_gen,
            epochs = self.config1.params_epochs,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps
        )
        logging.info(f"history stored: ", history)
        self.save_model(
            path=self.config1.trained_model_path,
            model=model
        )
        logging.info(model.summary())
        return model
    
    
    # def update_base_model(self):
    #     self.full_model = self._prepare_full_model(
    #         model=self.model,
    #         classes=self.config.params_classes,
    #         freeze_all=True,
    #         freeze_till=None,
    #         learning_rate=self.config.params_learning_rate
    #     )

    #     self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        
if __name__ == "__main__":
    config = ConfigurationManager()
    get_base_model_config = config.get_prepare_base_model_config()
    get_model_train_config = config.get_training_config()
            
    base_model_pipe = PrepareBaseModel(get_base_model_config, get_model_train_config)
    # model = base_model_pipe.get_base_model()
    base_model_pipe.train(
        model = tf.keras.applications.MobileNetV2,
            classes=get_base_model_config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=get_base_model_config.params_learning_rate
    )
    # base_model_pipe.update_base_model()
 