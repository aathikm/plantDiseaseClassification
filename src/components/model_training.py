import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir,"..")
sys.path.append(src_dir)

from config.configuration import TrainingConfig
from config.configuration import ConfigurationManager
from loggingInfo.loggingInfo import logging

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        return self.model

    # def train_valid_generator(self):

        # datagenerator_kwargs = dict(
        #     rescale=1./255
        #     # shear_range=0.2,
        #     # zoom_range=0.2,
        #     # horizontal_flip=True
        #     # rotation_range=40,
        #     # width_shift_range=0.2,
        #     # height_shift_range=0.2,
        #     # brightness_range=[0.5, 1.5],
        #     # channel_shift_range=10,
        #     # vertical_flip=True,
        #     # validation_split=0.2,
        #     # featurewise_center=True,  # set input mean to 0 over the dataset
        #     # samplewise_center=True,   # set each sample mean to 0
        #     # featurewise_std_normalization=True,  # divide inputs by std of the dataset
        #     # samplewise_std_normalization=True,   # divide each input by its std
        # )

        # dataflow_kwargs = dict(
        #     target_size=self.config.params_image_size[:-1],
        #     batch_size=self.config.params_batch_size,
        #     interpolation="bilinear",
        #     class_mode='categorical',
        # )

        # valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        #     **datagenerator_kwargs
        # )
        
        # self.train_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset="training",
        #     shuffle=True,
        #     **dataflow_kwargs
        # )

        # self.valid_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset="validation",
        #     shuffle=True,
        #     **dataflow_kwargs
        # )
        
        # print(self.valid_generator.__getitem__()[1])
        # print(self.train_generator.__getitem__()[1])

        # if self.config.params_is_augmentation:
        #     train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        #         rotation_range=40,
        #         horizontal_flip=True,
        #         width_shift_range=0.2,
        #         height_shift_range=0.2,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        #         **datagenerator_kwargs
        #     )
        # else:
        #     train_datagenerator = valid_datagenerator

        # self.train_generator = train_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset="training",
        #     shuffle=True,
        #     **dataflow_kwargs
        # )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def train(self):
    #     datagenerator_kwargs = dict(
    #         rescale=1./255
    #    )

    #     dataflow_kwargs = dict(
    #         target_size=self.config.params_image_size[:-1],
    #         batch_size=self.config.params_batch_size,
    #         interpolation="bilinear",
    #         class_mode='categorical',
    #     )

    #     valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #         **datagenerator_kwargs
    #     )
        
        # self.train_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset="training",
        #     shuffle=True,
        #     **dataflow_kwargs
        # )

        # self.valid_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset="validation",
        #     shuffle=True,
        #     **dataflow_kwargs
        # )
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

        train_data_gen = img_preprocessor.flow_from_directory(self.config.training_data,
                                                              target_size=self.config.params_image_size[:-1],
                                                              batch_size=self.config.params_batch_size,
                                                              subset='training',
                                                              class_mode='categorical', shuffle=True)

        val_data_gen = img_preprocessor.flow_from_directory(self.config.training_data,
                                                            target_size=self.config.params_image_size[:-1],
                                                            batch_size=self.config.params_batch_size,
                                                            subset='validation', 
                                                            class_mode='categorical', shuffle=True)

        self.steps_per_epoch = train_data_gen.samples // train_data_gen.batch_size
        self.validation_steps = val_data_gen.samples // val_data_gen.batch_size
        
        logging.info(f"steps per epoch: {self.steps_per_epoch}")
        logging.info(f"updated base model path: {self.config.updated_base_model_path}")
        
        model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        logging.info(f"model summary: {model.summary()}")
        history = model.fit(
            x = train_data_gen,
            validation_data = val_data_gen,
            epochs = self.config.params_epochs,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps
        )
        logging.info(f"history stored: ", history)
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
        
if __name__ == "__main__":
        
    config = ConfigurationManager()
    get_training_conf = config.get_training_config()
    logging.info(f"training config - {get_training_conf}")
    training_pipe = Training(get_training_conf)
    training_pipe.get_base_model()
    # training_pipe.train_valid_generator()
    training_pipe.train()
        