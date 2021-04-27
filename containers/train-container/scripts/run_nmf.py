import joblib, os, json
from sklearn import decomposition
from optparse import OptionParser

base_path = '/opt/ml'
input_path = os.path.join(base_path,'input/data')
output_path = os.path.join(base_path,'output')
model_path = os.path.join(base_path,'model')
hyperparameters = json.load(os.path.join(base_path,'input/data','hyperparameters.json'))
k = hyperparameters['k']


dict_ = joblib.load(input_path)


def train(X,k,max_iter=300,random_state=1234):
    model = decomposition.NMF(n_components=k,init='nndsvd',max_iter=max_iter,random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    return W,H,model

if __name__ == "__main__":
    parser = OptionParser(usage="usage: %prog [options] chamber")
    parser.add_option('--k',action='store',type='int',dest='k')
    parser.add_option('--miter',action='store',type='int', dest='max_iter')
    parser.add_option('--r',action='store',type='int', dest='random_state',default=1234)
    (options,args) = parser.parse_args()
    k = args[0]
    
