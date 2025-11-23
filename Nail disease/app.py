from flask import Flask, render_template, request
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
class_names = ['Acral_Lentiginous_Melanoma', 'Onychogryphosis', 'beau_s lines', 'blue_finger', 'clubbing', 'healthy', 'koilonychia', 'onychomycosis', 'pitting', 'psoriasis', 'terry_s nail', 'white nail', 'yellow nails']
# Load the saved model
model = load_model('naildisease.h5')


def predict_img(fpath):
    
    image=cv2.imread(fpath)
    example = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(example)
    image_resized= cv2.resize(image, (256,256))
    image=np.expand_dims(image_resized,axis=0)
    pred=model.predict(image)
    output=class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    print(confidence)
    print(output)
    return(output,confidence)


# Create flask instance
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index1.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds="load"
    confid="load"
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds,confid = predict_img(file_path)
        result = preds
        return render_template('index1.html',preds=preds,conf=confid,fname=f.filename)
    return render_template('index1.html',preds=preds,conf=confid,fname=f.filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=5000)
