# example of using Flask to deploy a fastai deep learning model trained on a tabular dataset
import json
import os
import urllib.request
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# carry over imports from custom layer version
import time
import seaborn as sns
# import datetime, timedelta
import datetime
import pydotplus
from datetime import datetime, timedelta
from datetime import date
from dateutil import relativedelta
from io import StringIO
import pandas as pd
import pickle
from pickle import dump
from pickle import load
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# DSX code to import uploaded documents
from io import StringIO
import requests
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
import os
import yaml
import math
from flask import Flask, render_template, request

# load config gile
current_path = os.getcwd()
print("current directory is: "+current_path)
path_to_yaml = os.path.join(current_path, 'flask_web_deploy_config.yml')
print("path_to_yaml "+path_to_yaml)
try:
    with open (path_to_yaml, 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print('Error reading the config file')

# build the path for the trained model
rawpath = os.getcwd()
# models are in a directory called "models" in the same directory as this module
model_path = os.path.abspath(os.path.join(rawpath, 'models'))

print("path is:",rawpath)
print("model_path is: ",model_path)
# load the model

model_file_name = os.path.join(model_path,config['file_names']['saved_model'])

loaded_model = tf.keras.models.load_model(model_file_name)

def get_model_path():
    '''get the path for data files
    
    Returns:
        path: path for model files
    '''
    rawpath = os.getcwd()
    # models are in a directory called "models" in the same directory as this module
    path = os.path.abspath(os.path.join(rawpath, 'models'))
    return(path)

app = Flask(__name__)


@app.route('/')
def home():   
    ''' render home.html - page that is served at localhost that allows user to enter model scoring parameters'''
    title_text = "fastai deployment"
    title = {'titlename':title_text}
    return render_template('home.html',title=title) 
    
@app.route('/show-prediction/')
def show_prediction():
    ''' 
    get the scoring parameters entered in home.html and render show-prediction.html
    '''
    # the scoring parameters are sent to this page as parameters on the URL link from home.html
    # load the scoring parameter values into a dataframe
    # create and load scoring parameters dataframe (containing the scoring parameters)that will be fed into the pipelines
    score_df = pd.DataFrame(columns=scoring_columns)
    print("score_df before load is "+str(score_df))
    for col in scoring_columns:
        print("value for "+col+" is: "+str(request.args.get(col)))    
        score_df.at[0,col] = request.args.get(col)
   # print details about scoring parameters
    print("score_df: ",score_df)
    print("score_df.dtypes: ",score_df.dtypes)
    print("score_df.iloc[0]",score_df.iloc[0])
    print("shape of score_df.iloc[0] is: ",score_df.iloc[0].shape)
    pred_class,pred_idx,outputs = learner.predict(score_df.iloc[0])
    for col in scoring_columns:
        print("pred_class "+str(col)+" is: "+str(pred_class[col]))
    print("pred_idx is: "+str(pred_idx))
    print("outputs is: "+str(outputs))
    # get a result string from the value of the model's prediction
    if outputs[0] >= outputs[1]:
        predict_string = "Prediction is: individual has income less than 50k"
    else:
        predict_string = "Prediction is: individual has income greater than 50k"
    # build parameter to pass on to show-prediction.html
    prediction = {'prediction_key':predict_string}
    # render the page that will show the prediction
    return(render_template('show-prediction.html',prediction=prediction))
    
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')