import os 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.models import load_model
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import json
import struct
import tempfile
import shutil
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-params', action='store', dest='params_file',
                    help='params file')
args = parser.parse_args()
args = parser.parse_args([
    '-params', './leonard/data/vertex200m.params.json'
])
with open(args.params_file, 'r') as f:
    params = json.load(f)
params['re_values_dict'].pop('hash')  # 移除params 中're_values_dict'中'hash'条目

with open(args.params_file+'_s', 'w') as f:  # 换名保存，删除原来的json
    json.dump(params, f, indent=4)