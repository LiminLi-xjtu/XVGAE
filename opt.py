import random
import argparse
parser = argparse.ArgumentParser(description='XVGAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# setting
parser.add_argument('--name', type=str, default="151669")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--k', type=int, default=8)
parser.add_argument('--gamma_value', type=float, default=10)
parser.add_argument('--lr_pr', type=float, default=1e-2)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--epoch_pre', type=int, default=200)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--show_training_details', type=bool, default=False)

# AE
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

# GAE
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=128)

# clustering
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)
args = parser.parse_args()
