"""
The main file to train P3GM and construct synthetic data.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from hivae import HIVAE
from p3gm import P3GM, HIP3GM
from vae import VAE
import pathlib
filedir = pathlib.Path(__file__).resolve().parent
import my_util

parser = argparse.ArgumentParser(description='Implementation of P3GM')

parser.add_argument('--db', type=str, help="used dataset [adult, credit, mnist, fashion, esr, isolet]", default="adult")
parser.add_argument('--alg', type=str, help="used algorithm [p3gm, hip3gm, vae, hivae]", default="hip3gm")
parser.add_argument('--lot_size', type=int, default=200,
                    help='input batch size for sgd (default: 200)')
parser.add_argument('--sgd_sigma', type=float, default=1.31,
                    help='noise multiplier for sgd (default: 1.31)')
parser.add_argument('--gmm_sigma', type=float, default=100,
                    help='noise multiplier for em (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for sgd (default: 1e-3)')
parser.add_argument('--gmm_n_comp', type=int, default=1,
                    help='the number of mixture of Gaussian (default: 1)')
parser.add_argument('--pca_eps', type=float, default=1e-2,
                    help='epsilon for pca (default: 1e-2)')
parser.add_argument('--epoch', type=int, default=2,
                    help='the number of epochs (default: 2)')
parser.add_argument('--z_dim', type=int, default=20,
                    help='the latent dimensionality (=the number of pca components) (default: 20)')
parser.add_argument('--latent_dim', type=int, default=300,
                    help='the number of nodes in latent layers (default: 300)')
parser.add_argument('--n_microbatches', type=int, default=0,
                    help='the number of microbatches (default: lot_size)')
parser.add_argument('--n_iter', type=int, default=1,
                    help='the number of iterations of a set of training (default: 1)')
parser.add_argument('--no_dp', action='store_true')
args = parser.parse_args()
    
if args.n_microbatches == 0:
    args.n_microbatches = args.lot_size

def main():
    
    torch.manual_seed(0)
    random_state = np.random.RandomState(0)

    X, encoders = my_util.load_dataset(args.db)
    train_loader = my_util.make_dataloader(X, args.lot_size, random_state=random_state)
    
    dims = np.array([len(encoder.categories_[0]) for encoder in encoders])
    is_cuda = torch.cuda.is_available()
    
    dir = filedir.parent / "synthetic_data" / f"{args.db}"
    dir.mkdir(parents=True, exist_ok=True)
    img_dir = filedir.parent / "result" / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print("Algorithm:", args.alg)
    print("Data shape:",  X.shape)
    
    # choose algorithms
    # HIP3GM inheretes HI-VAE
    # P3GM inheretes VAE
    # HIVAE is not differenally private
    if args.alg == "hip3gm":
        MODEL = HIP3GM
    elif args.alg == "p3gm":
        MODEL = P3GM
    elif args.alg == "hivae":
        MODEL = HIVAE
    else:
        MODEL = VAE
        
    device = torch.device("cuda:0" if is_cuda else "cpu")

    for i in range(args.n_iter):
        # initialize the model
        model = MODEL.make_model(dims, device, z_dim=args.z_dim, latent_dim=args.latent_dim)
        
        # training
        model.train(train_loader, random_state=random_state, sgd_sigma=args.sgd_sigma, sgd_epoch=args.epoch, gmm_sigma=args.gmm_sigma, gmm_n_comp=args.gmm_n_comp, pca_eps=args.pca_eps, num_microbatches=args.n_microbatches, no_dp=args.no_dp)

        # generate synthetic data from the trained model
        syn_data = model.generate_data(len(X)).detach().cpu().numpy()
        
        # inverse the one-hot data to categorical
        inversed_data = pd.DataFrame(my_util.inverse(syn_data, encoders))
        
        # save
        inversed_data.to_csv(dir / f"out_{i}.csv", index=None)
    
    
if __name__ == "__main__":
    main()