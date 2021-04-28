import joblib, os, json, argparse
from sklearn import decomposition

base_path = '/opt/ml/'
input_path = os.path.join(base_path,'input/data')
output_path = os.path.join(base_path,'output')
model_path = os.path.join(base_path,'model')
with open(os.path.join(base_path,'input/config/hyperparameters.json')) as F:
    trainingParams = json.load(F)

channel = 'training'
training_path = os.path.join(input_path,channel)

def train():
    
    # get K and convert to int
    k = trainingParams.get('k',None)
    if k is not None:
        k = int(k)
    else:
        k = 50
    
    # read in the data, should be a single .pkl file
    training_file = os.listdir(training_path)[0]
    input_dict = joblib.load(os.path.join(training_path,training_file))
    X = input_dict['dtm']
    
    # get the chamber name
    chamber = training_file.split('.')[0]
    
    # fit model
    model = decomposition.NMF(n_components=k,init='nndsvd',max_iter=200,random_state=1234)
    W = model.fit_transform(X)
    H = model.components_
    model_dict = {"W":W,'H':H,"model":model}
    
    with open(os.path.join(model_path,f'NMF_{chamber}_{k}.pkl'),'wb') as out:
        joblib.dump(model_dict,out)

if __name__ == "__main__":
    train()
