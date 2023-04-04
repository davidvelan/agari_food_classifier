from flask import Flask, request
import numpy as np
import food_classifier as Classifier
import cv2
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return 'food processing api'


@app.route("/upload", methods=['POST'])
def upload_image():
    file = request.files['image']
    image_file = cv2.imdecode(np.frombuffer(
        file.read(), np.uint8), cv2.IMREAD_COLOR)
    output = Classifier.food_classifier(image_file)
    food_type = ''

    if output == 0:
        food_type = 'chicken'
    elif output == 1:
        food_type = 'salmon'
    elif output == 2:
        food_type = 'steak'
    else:
        food_type = 'not identified'

    return food_type


if __name__ == '__main__':
    app.run(debug=True)
