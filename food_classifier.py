import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model


def food_classifier(img):
    model = load_model(os.path.join(
        'models', 'functional_food_type_classifer.h5'))
    # img = cv2.imread(image)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    resize = tf.image.resize(img, (256, 256))
    # plt.imshow(resize.numpy().astype(int))
    # plt.show()

    yhat = model.predict(np.expand_dims(resize/255, 0))
    # print(yhat)
    output = np.argmax(yhat)

    return output
