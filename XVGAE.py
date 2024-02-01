import opt
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter


# AE encoder
class AE_encoder(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


# AE decoder
class AE_decoder(nn.Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat

####################
# AE
class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)
#########################

# GNNLayer
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if opt.args.name == "dblp":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        else:
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            if opt.args.name == "dblp":
                support = self.act(F.linear(features, self.weight))
            else:
                support = self.act(torch.mm(features, self.weight))
        else:
            if opt.args.name == "dblp":
                support = F.linear(features, self.weight)
            else:
                support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        az = torch.spmm(adj, output)
        return output, az


# GAE encoder
class GAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, n_input):
        super(GAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)

        self.s = nn.Sigmoid()

    def forward(self, x, adj1,adj2):
        #  zp:embedding of location  zi:embedding of image.

        # layer1 location
        zp_1, azp_1 = self.gnn_1(x, adj1, active=True)
        # layer1 image
        zi_1, azi_1 = self.gnn_1(x, adj2, active=True)
        # layer2 location
        zp_igae, azp_2 = self.gnn_2(zi_1, adj1, active=True)
        # layer2 image
        zi_igae, azi_2 = self.gnn_2(zp_1, adj2, active=True)

        return zp_igae,zi_igae


# GAE decoder
class GAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2,  n_input):
        super(GAE_decoder, self).__init__()

        self.gnn_5 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_6 = GNNLayer(gae_n_dec_2, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_gae, adj):
        z_1, az_1= self.gnn_5(z_gae, adj, active=True)
        z_hat, az_2 = self.gnn_6(z_1, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2], [z_1,  z_hat]


# GAE
class GAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2,  gae_n_dec_1, gae_n_dec_2,  n_input):
        super(GAE, self).__init__()
        # GAE encoder
        self.encoder = GAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,

            n_input=n_input)
        # GAE decoder
        self.decoder = GAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,

            n_input=n_input)


# readout function
class Readout(nn.Module):
    def __init__(self, K):
        super(Readout, self).__init__()
        self.K = K

    def forward(self, Z):
        # calculate cluster-level embedding
        Z_tilde = []

        # step1: split the nodes into K groups
        # step2: average the node embedding in each group
        n_node = Z.shape[0]
        step = n_node // self.K
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))

        # the cluster-level embedding
        Z_tilde = torch.cat(Z_tilde, dim=0)
        return Z_tilde.view(1, -1)



class XVGAE(nn.Module):
    def __init__(self, n_node=None):
        super(XVGAE, self).__init__()

        # Auto Encoder
        self.ae = AE(
            ae_n_enc_1=opt.args.ae_n_enc_1,
            ae_n_enc_2=opt.args.ae_n_enc_2,
            ae_n_enc_3=opt.args.ae_n_enc_3,
            ae_n_dec_1=opt.args.ae_n_dec_1,
            ae_n_dec_2=opt.args.ae_n_dec_2,
            ae_n_dec_3=opt.args.ae_n_dec_3,
            n_input=opt.args.n_input,
            n_z=opt.args.n_z)

        # GAE
        self.gae = GAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_dec_1=opt.args.gae_n_dec_1,
            gae_n_dec_2=opt.args.gae_n_dec_2,
            n_input=opt.args.n_input)

        # fusion parameter from DFCN
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.zeros(1))

        # cluster layer (clustering assignment matrix)
        self.cluster_centers = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)

        # readout function
        self.R = Readout(K=opt.args.n_clusters)

    # calculate the soft assignment distribution Q
    def q_distribute(self, Z, Z_ae, Z_gae):

        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_gae = 1.0 / (1.0 + torch.sum(torch.pow(Z_gae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_gae = (q_gae.t() / torch.sum(q_gae, 1)).t()

        return [q, q_ae, q_gae]

    def forward(self, X, A, Ai):
        # AE

        Z_ae = self.ae.encoder(X)
        # GAE
        Z_gae1,Z_gae2 = self.gae.encoder(X, A,Ai)

        # linear combination of view 1 and view 2
        Z_ae = Z_ae
        Z_gae = (Z_gae1 + Z_gae2) / 2

        # fusion
        Z_i = self.a * Z_ae + self.b * Z_gae
        Z_l = torch.spmm(A, Z_i)
        S = torch.mm(Z_l, Z_l.t())
        S = F.softmax(S, dim=1)
        Z_g = torch.mm(S, Z_l)
        Z = self.alpha * Z_g + Z_l

        # Reconstruction X
        X_hat = self.ae.decoder(Z)

        # Q
        Q = self.q_distribute(Z, Z_ae, Z_gae)


        return X_hat, Q, Z,
