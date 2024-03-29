import tqdm
from utils import *
from torch.optim import Adam
import scanpy as sc
import matplotlib.pyplot as plt
import opt
loss_list = []
x = []
def pretrain(model, X, y, A, Ai):

    print("preTraining…")
    # calculate embedding similarity and cluster centers
    centers = model_init(model, X, y, A)
    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)

    optimizer = Adam(model.parameters(), lr=opt.args.lr_pr)

    for epoch in tqdm.tqdm(range(opt.args.epoch_pre+1)):
        # input & output
        X_hat, Q, Z = model(X, A, Ai)
        # loss: L_{REC}
        L_REC = reconstruction_loss(X, X_hat)
        loss =  L_REC
        # loss图
        x.append(epoch)
        loss_list.append(loss.cpu().detach().numpy())
        if epoch == opt.args.epoch_pre:
            plt.plot(x, loss_list, 'r', lw=5)
            plt.title("loss")
            plt.xlabel("steps")
            plt.ylabel("loss")
            plt.legend("pretrain_loss")
            load_path = "data/" + opt.args.name + "/"
            load_path = load_path + "losspretrain.png"
            plt.savefig(load_path)

        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # clustering & evaluation
        acc, nmi, ari, f1, _ ,ypre= clustering(Z, y)
        print("epoch",epoch,"ari",ari)
        # plot
        if epoch == opt.args.epoch_pre:
            # print(model.state_dict())
            load_path = "data/" + opt.args.name + "/"
            torch.save(
                model.state_dict(), load_path+"pretrain.pkl"
            )
            adata = sc.read_visium(path=load_path, count_file=opt.args.name + '_filtered_feature_bc_matrix.h5')
            adata.var_names_make_unique()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            print(adata)
            print(len(ypre))
            ypre=[str(i) for i in ypre]
            adata.obs['y'] = ypre
            adata.obs['y'] = adata.obs['y'].astype('category')
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.spatial(adata, color=["y"], title=[' (ARI=%.2f)' % ari],size=1.5)

