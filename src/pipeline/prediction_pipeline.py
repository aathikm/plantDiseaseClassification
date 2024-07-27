import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir,"..")
sys.path.append(src_dir)

from loggingInfo.loggingInfo import logging

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
    
    def predict(self):
        # load model
        model = tf.keras.models.load_model(os.path.join("models", "model.h5"))
        labels_df_path = os.path.join("models", "labels.csv")

        imagename = self.filename
        logging.info(imagename)
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size = (224,224))
        logging.info(test_image)
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        logging.info(result)
        labels_df = pd.read_csv(labels_df_path)
        logging.info(labels_df)
        labels_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        logging.info(labels_df)
        result_val = labels_df["Disease"][labels_df["Code"] == result[0]].values[0]

        logging.info(result_val)
        print(result_val)
        return(result_val)

# if __name__ == "__main__":
#     obj = PredictionPipeline(filename= "testImage/image12.JPG")
#     obj.predict()
    