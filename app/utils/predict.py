import tensorflow as tf #type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image #type: ignore

# Load the model once
import os
model_path = os.path.join(os.path.dirname(__file__), '..','models', 'cat_dog_classifier.h5')
model = tf.keras.models.load_model(model_path)

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "Dog ", float(prediction)
    else:
        return "Cat ", float(1 - prediction)
