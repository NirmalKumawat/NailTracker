import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image

import boto3, botocore

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_restful import Api,Resource
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define a flask app
app = Flask(__name__)
api=Api(app)
# Model saved with Keras model.save()

app.config['S3_BUCKET'] = "doctodo-lab"
app.config['S3_KEY'] = "AKIA2OWK2G6V7WXG4BL7"
app.config['S3_SECRET'] = "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"
app.config['S3_LOCATION'] = 'http://{}.s3.amazonaws.com/'.format('doctodo-lab')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =tf.keras.models.load_model('model.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

client = boto3.client(
   "s3",
   region_name='ap-south-1',
   aws_access_key_id=app.config['S3_KEY'],
   aws_secret_access_key=app.config['S3_SECRET']
)


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(244, 244))
    show_img = image.load_img(img_path, grayscale=False, target_size=(244, 244))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        #Save File to s3 bucket
        if f:
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            print(file_path)
            # Make prediction
            preds = model_predict(file_path, model)
            # print(preds[0])

            # x = x.reshape([64, 64]);
            disease_class = ['beau_s line','black line','clubbing', 'mees_ line','normal','onycholysis','terry_s nail0', 'white spot']
            a = preds[0]
            ind=np.argmax(a)
            print('Prediction:', disease_class[ind])
            result=disease_class[ind]
            
            client.upload_file(
                file_path,app.config['S3_BUCKET'],app.config['S3_KEY']
            )
            print("Upload Done !")

            os.remove(file_path)
            print("deleted from locally!")

            return result

        
        
    return None








if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    
    app.run(debug=True)