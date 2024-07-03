from flask import Flask, request, jsonify,Blueprint,send_file
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense

import tensorflow.keras.layers as L
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import librosa
import pickle
import os
import base64
from io import BytesIO
import uuid
import tempfile

# Custom object dictionary
custom_objects = {
    'Sequential': Sequential,
    'Conv1D': Conv1D,
    'BatchNormalization': BatchNormalization,
    'MaxPooling1D': MaxPooling1D,
    'Dropout': Dropout,
    'Flatten': Flatten,
    'Dense': Dense
}
# Create a Blueprint for deployment
deploy_bp = Blueprint('Deploy', __name__, url_prefix='/Deploy')


json_file = open(r'SpeechEmotion\app\kaggle\CNN_model.json', 'r')
loaded_model_json = json_file.read()
#print(loaded_model_json)
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
# load weights into new model
loaded_model.load_weights(r"SpeechEmotion\app\kaggle\best_model1_weights.h5")


# Load the scaler and encoder
with open(r'SpeechEmotion\app\kaggle\scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open(r'SpeechEmotion\app\kaggle\encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

# Define feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                    ))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 2376))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result



def prediction(path):
    res = get_predict_feat(path)
    predictions = loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]



        
# Prediction endpoint form voice recording. 
@deploy_bp.route('/predict-emotion',methods=['POST'])
def predict():
    print("Predict endpoint accessed") 
    file = request.files['file']
    
    if not file:
        return jsonify({'error': 'Empty file'})

    prediction_result = prediction(file)

    return jsonify({

        'emotion': prediction_result

    })





