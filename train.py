import tqdm
from utils import *
from torch.optim import Adam
import scanpy as sc
import matplotlib.pyplot as plt
import opt
from sklearn.metrics.cluster import adjusted_rand_score
loss_list = []
x = []
def train(model, X, y, A, Ai):

    print("Training…")
    load_path = "data/" + opt.args.name + "/"
    model.load_state_dict(torch.load(load_path+"pretrain.pkl", map_location='cpu'))
    # calculate embedding similarity and cluster centers
    centers = model_init(model, X, y, A)
    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)
    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    for epoch in tqdm.tqdm(range(opt.args.epoch+1)):
        # input & output
        X_hat,  Q, Z= model(X, A, Ai)
        # loss: L_{REC} and L_{KL}
        L_REC = reconstruction_loss(X, X_hat)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss =  L_REC +opt.args.gamma_value * L_KL
        # loss
        x.append(epoch)
        loss_list.append(loss.cpu().detach().numpy())
        if epoch == opt.args.epoch:
            plt.plot(x, loss_list, 'r', lw=5)
            plt.title("loss")
            plt.xlabel("steps")
            plt.ylabel("loss")
            plt.legend("train_loss")
            load_path = "data/" + opt.args.name + "/"
            load_path1 = load_path + "losstrain.png"
            plt.savefig(load_path1)
        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # clustering & evaluation
        acc, nmi, ari, f1, _ ,ypre= clustering(Z, y)
        print("epoch",epoch,"ari",ari)

        # plot
        if epoch == opt.args.epoch:
            adata = sc.read_visium(path=load_path, count_file=opt.args.name + '_filtered_feature_bc_matrix.h5')
            adata.var_names_make_unique()
            sc.pp.filter_genes_dispersion(adata, n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            ypre = [str(i) for i in ypre]
            adata.obs['y'] = ypre
            adata.obs['y'] = adata.obs['y'].astype('category')
            adata.obs['Ground Truth'] = y
            adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.spatial(adata, color=["y"], title=[' (ARI=%.2f)' % ari],size=1.3)
            sc.pl.spatial(adata, color=["Ground Truth"], title=[' (ARI=%.2f)' % ari], size=1.3)



