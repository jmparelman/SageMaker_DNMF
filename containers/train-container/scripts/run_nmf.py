import joblib, os, json
from sklearn import decomposition

base_path = '/opt/ml'
input_path = os.path.join(base_path,'input/data')
output_path = os.path.join(base_path,'output')
model_path = os.path.join(base_path,'model')
hyperparams = json.load(os.path.join(base_path,'input/conf/hyperparameters.json'))
