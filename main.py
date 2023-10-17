from flask import Flask, render_template, request

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = '/Users/haris/PycharmProjects/Flask/my_model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")


def pred_cotton(cotton):
    test_image = load_img(cotton, target_size=(224, 224))
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)
    print(pred)
    if pred == 0:
        return "Diseased Cotton Leaf", 'DiseasedCottonLeaf.html'

    elif pred == 1:
        return "Diseased Cotton Plant", 'DiseasedCottonPlant.html'

    elif pred == 2:
        return "Cotton Leaf is not Diseased", 'Healthyleaf.html'
    else:
        return "Cotton Plant is not Diseased", 'healthyplant.html'



app = Flask(__name__)


# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join(
            'C:/Users/haris/PycharmProjects/Flask/static/uploads',
            filename)
        file.save(file_path)

        print("@@ Predicting class")
        pred, output_page = pred_cotton(cotton=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)


# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=8080)