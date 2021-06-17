from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import numpy as np
import cv2
import os
from glob import glob
from PIL import Image 
from skimage import transform
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

'''UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER'''

# Define a flask app
app = Flask(__name__,static_url_path='/',
            static_folder='./',template_folder='./')

# Model saved with Keras model.save()
MODEL_VGG16_PATH = 'models\model_vgg16.h5'
#MODEL_INCEPTION3_PATH = 'models\model_inception.h5'
#MODEL_RESNET50_PATH = 'models\model_resnet50.h5'

# Load your trained model
MODEL_VGG16 = load_model(MODEL_VGG16_PATH)
MODEL_VGG16.make_predict_function()          # Necessary

#MODEL_INCEPTION3 = load_model(MODEL_INCEPTION3_PATH)
#MODEL_INCEPTION3.make_predict_function()          # Necessary

#MODEL_RESNET50 = load_model(MODEL_RESNET50_PATH)
#MODEL_RESNET50.make_predict_function()          # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def model_predict(img_path, model):
    image = load(img_path)
    preds = model.predict(image,batch_size=32)
    return preds

@app.route('/',methods=['GET'])
def root():
   return render_template("index.html")

@app.route('/index.html')
def index():
   return render_template("index.html")

@app.route('/upload.html')
def upload():
   return render_template("upload.html")

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,"uploads/upload_img.png")
        f.save("uploads/upload_img.png")

        #Labels
        lst = ["Covid-19","Normal","Tuberculosis","Viral Pneumonia"]

        # Make prediction VGG16
        preds_VGG16 = model_predict(file_path, MODEL_VGG16)
        pred_class_VGG16 = np.argmax(preds_VGG16)            
        result_VGG16 = str(lst[pred_class_VGG16])

        # Make prediction InceptionV3
        #preds_InceptionV3 = model_predict(file_path, MODEL_INCEPTION3)
        #pred_class_InceptionV3 = np.argmax(preds_InceptionV3)            
        #result_InceptionV3 = str(lst[pred_class_InceptionV3]) 

        # Make prediction ResNet50
        #preds_ResNet50 = model_predict(file_path, MODEL_RESNET50)
        #pred_class_ResNet50 = np.argmax(preds_ResNet50)            
        #result_ResNet50 = str(lst[pred_class_ResNet50])   

        return render_template('results_chest.html',result=result_VGG16)

   return None

if __name__ == '__main__':
   #app.secret_key = ".."
   app.run()