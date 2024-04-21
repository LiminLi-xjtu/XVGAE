import opt
import torch
import numpy as np
import random
import scanpy as sc
from sklearn.decomposition import PCA
from utils import normalize_adj,get_process
from XVGAE import XVGAE
from pretrain import pretrain
from train import train
from adj import adj


# seedALL
torch.manual_seed(opt.args.seed)
torch.cuda.manual_seed(opt.args.seed)
torch.cuda.manual_seed_all(opt.args.seed)
np.random.seed(opt.args.seed)
random.seed(opt.args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if opt.args.cuda else "cpu")
dataset_name=opt.args.name
load_path = "data/" + dataset_name + "/"

# load data
adata = sc.read_visium(path=load_path, count_file=opt.args.name+'_filtered_feature_bc_matrix.h5')
X=get_process(adata,pca_n=50)

# adj
A = adj(adata,view='gene',model='Radius')
A = normalize_adj(A)
A = torch.FloatTensor(A).to(device)

# adj image
datai = np.loadtxt( load_path+opt.args.name+"128_0.5_200_12_simCLR_reprensentation2.csv",delimiter=",")
pca = PCA(n_components=100)
image_spatial=pca.fit_transform(datai)
adata.obsm['image_spatial']=image_spatial
Ai = adj(adata,view='image',model='KNN')
Ai = normalize_adj(Ai)
Ai = torch.FloatTensor(Ai).to(device)

if opt.args.name == 'MBC' or opt.args.name == 'MBP' :
    y=None
else:
    y= np.loadtxt(load_path+opt.args.name+"truth.csv",delimiter=",")


if opt.args.name == '151677':
    opt.args.n_clusters = 20
    opt.args.n_input = 50
    opt.args.alpha_value = 0.2
    opt.args.lambda_value = 10
    opt.args.gamma_value = 1e3
elif opt.args.name == "151669" or opt.args.name == "151670" or opt.args.name == "151671" or opt.args.name == "151672":
    opt.args.n_clusters = 5
    opt.args.n_input = 50
    opt.args.alpha_value = 0.2
    opt.args.lambda_value = 10
    opt.args.gamma_value = 1e3
elif opt.args.name == "151678":
    opt.args.n_clusters = 8
    opt.args.n_input = 50
    opt.args.alpha_value = 0.2
    opt.args.lambda_value = 10
    opt.args.gamma_value = 1e3
elif opt.args.name == "MBP" or opt.args.name == 'MBC':
    opt.args.n_clusters = 15
    opt.args.n_input = 50
    opt.args.alpha_value = 0.2
    opt.args.lambda_value = 10
    opt.args.gamma_value = 1e3
else:
    opt.args.n_clusters = 7
    opt.args.n_input = 50
    opt.args.alpha_value = 0.2
    opt.args.lambda_value = 10
    opt.args.gamma_value = 1e3


opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
model = XVGAE(n_node=X.shape[0]).to(opt.args.device)
pretrain(model, X, y, A, Ai)
train(model,X, y, A, Ai)
