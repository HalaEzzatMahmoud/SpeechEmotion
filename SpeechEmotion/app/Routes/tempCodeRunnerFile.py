from tensorflow.keras.models import Sequential, model_from_json
from flask import Flask, request, jsonify,Blueprint,json
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

deploy_bp = Blueprint('Deploy',__name__, url_prefix='/model')

json_file = open(r'D:\GPDOC\GP_BackEnd\SpeechEmotion\app\kaggle\CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"D:\GPDOC\GP_BackEnd\SpeechEmotion\app\kaggle\best_model1_weights.h5")
#print("Loaded model from disk")

with open(r'D:\GPDOC\GP_BackEnd\SpeechEmotion\app\kaggle\scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
    
with open(r'D:\GPDOC\GP_BackEnd\SpeechEmotion\app\kaggle\encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)


def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                    zcr(data,frame_length,hop_length),
                    rmse(data,frame_length,hop_length),
                    mfcc(data,sr,frame_length,hop_length)
                    ))
    return result

res=get_predict_feat(r"D:\GPDOC\RecordsTest\03-01-03-01-01-01-02.wav")
print(res.shape)



    
