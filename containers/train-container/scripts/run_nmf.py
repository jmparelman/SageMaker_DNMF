import joblib, os, json, argparse
from sklearn import decomposition

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
    parser = argparse.ArgumentParser(description='run NMF training')
    parser.add_argument('chamber',type=int,help='chamber of congress to train on')
    parser.add_argument('-m','--max_iter',help='maximum NMF iterations',type=int,default=300)
    parser.add_argument('-r','random_state',help='random state for NMF',type=int,default=1234)
    args = parser.parse_args()
    
    dtm_dict = joblib.load(os.path.join(input_path,f'{args.chamber}.pkl'))
    dtm = dtm_dict['dtm']

    W,H,model = train(dtm,k,args.max_iter,args.random_state)
    joblib.dump({"W":W,"H":H,"model":model},os.path.join(output_path,f'{args.chamber}.pkl'))
