import joblib, os, json
from sklearn import decomposition
from optparse import OptionParser

base_path = '/opt/ml/train'
input_path = os.path.join(base_path,'input')
output_path = os.path.join(base_path,'output')
hyperparameters = json.load(os.path.join(base_path,'input/data','hyperparameters.json'))
k = hyperparameters['k']

def train(X,k,max_iter=300,random_state=1234):
    model = decomposition.NMF(n_components=k,init='nndsvd',max_iter=max_iter,random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    return W,H,model

if __name__ == "__main__":
    parser = OptionParser(usage="usage: %prog [options] chamber")
    parser.add_option('--miter',action='store',type='int', dest='max_iter')
    parser.add_option('--r',action='store',type='int', dest='random_state',default=1234)
    (options,args) = parser.parse_args()
    chamber = args[0]

    dtm_dict = joblib.load(os.path.join(input_path,f'{chamber}.pkl'))
    dtm = dtm_dict['dtm']

    W,H,model = train(dtm,k,options.max_iter,options.random_state)
    joblib.dump({"W":W,"H":H,"model":model},os.path.join(output_path,f'{chamber}.pkl'))
