# XVGAE
## Installation
```bash
pip install git+https://github.com/LiminLi-x]Tu/XVGAE.git
```
## Requirements
* Python 3.8
* torch 2.2.2
* Scanpy 1.9.1
* anndata 0.8.0
* louvain 0.8.0
* numpy 1.22.4
* scikit-learn 1.0.2
* pandas 1.3.5
* munkres 1.1.4
* tqdm 4.64.1
## Usage
```python
# load data
adata = sc.read_visium(path=load_path, count_file=opt.args.name+'_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
sc.pp.filter_genes_dispersion(adata, n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# pca x
X = adata.X.toarray()
pca_x = PCA(n_components=50)
X=pca_x.fit_transform(X)
X = torch.FloatTensor(X).to(device)

# adj spatial
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

# Create Model and Train
model = XVGAE(n_node=X.shape[0]).to(opt.args.device)
pretrain(model, X, y, A, Ai)
train(model,X, y, A, Ai)
```
