import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
    
    def predict(self):
        # load model
        model = tf.keras.models.load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        print(imagename)
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size = (224,224))
        print(test_image)
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

if __name__ == "__main__":
    obj = PredictionPipeline(filename= "testImage/image12.JPG")
    obj.predict()
    