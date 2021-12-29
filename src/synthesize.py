"""
The main file to train P3GM and construct synthetic data.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from p3gm import P3GM
from vae import VAE
import pathlib
filedir = pathlib.Path(__file__).resolve().parent
import my_util
import datetime
import json
import ml_task.exp_ml

parser = argparse.ArgumentParser(description='Implementation of P3GM')

parser.add_argument('--db', type=str, help="used dataset [adult, credit, mnist, fashion, esr, isolet]", default="adult")
parser.add_argument('--alg', type=str, help="used algorithm [p3gm, hip3gm, vae, hivae]", default="hip3gm")
parser.add_argument('--lot_size', type=int, default=200,
                    help='input batch size for sgd (default: 200)')
parser.add_argument('--sgd_sigma', type=float, default=1.31,
                    help='noise multiplier for sgd (default: 1.31)')
parser.add_argument('--sgd_epoch', type=int, default=2,
                    help='the number of epochs (default: 2)')
parser.add_argument('--clipping', type=float, default=1,
                    help='the clipping norm')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for sgd (default: 1e-3)')
parser.add_argument('--p3gm_delta', type=float, default=1e-5,
                    help='delta for P3GM (default: 1e-5)')
parser.add_argument('--gmm_sigma', type=float, default=100,
                    help='noise multiplier for em (default: 100)')
parser.add_argument('--gmm_n_comp', type=int, default=1,
                    help='the number of mixture of Gaussian (default: 1)')
parser.add_argument('--gmm_iter', type=int, default=20,
                    help='the number iterations of the EM algorithm (default: 20)')
parser.add_argument('--pca_sigma', type=float, default=1,
                    help='epsilon for pca (default: 1e-2)')
parser.add_argument('--z_dim', type=int, default=30,
                    help='the latent dimensionality (=the number of pca components) (default: 20)')
parser.add_argument('--latent_dim', type=int, default=1000,
                    help='the number of nodes in latent layers (default: 300)')
parser.add_argument('--num_microbatches', type=int, default=0,
                    help='the number of microbatches (default: lot_size)')
parser.add_argument('--n_iter', type=int, default=1,
                    help='the number of iterations of a set of training (default: 1)')
parser.add_argument('--skip_ml', action='store_true')
args = parser.parse_args()
    
if args.num_microbatches == 0:
    args.num_microbatches = args.lot_size

def main():
    
    torch.manual_seed(0)
    random_state = np.random.RandomState(0)

    X, encoders, keys, dims = my_util.load_dataset(args.db)
    X_test, _, _, _ = my_util.load_test_dataset(args.db)
    train_loader = my_util.make_dataloader(X, args.lot_size, random_state=random_state)

    dt_now = datetime.datetime.now()
    now_time = dt_now.strftime('%Y%m%d-%H%M%S')
    
    save_data_dir = pathlib.Path("/data/takagi") / "synthetic_data" / f"{args.db}" / now_time
    result_dir = filedir.parent / "result" / f"{args.db}" / now_time
    save_data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print("Algorithm:", args.alg)
    print("Data shape:",  X.shape)

    parameters = vars(args)
    parameters["p3gm_epsilon"] = P3GM.cp_epsilon(len(X), args.lot_size, args.pca_sigma, args.gmm_sigma, args.gmm_iter, args.gmm_n_comp, args.sgd_sigma, args.sgd_epoch, args.p3gm_delta)
    parameters["time"] = now_time
    with open(result_dir / 'param.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # P3GM inheretes VAE
    if args.alg == "p3gm":
        MODEL = P3GM
    else:
        MODEL = VAE
        
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    for i in range(args.n_iter):
        # initialize the model
        model = MODEL(dims, device, z_dim=args.z_dim, latent_dim=args.latent_dim).to(device)
        
        # training
        model.train(train_loader, random_state=random_state, **parameters)

        # generate synthetic data from the trained model
        syn_data = model.generate_data(len(X)).detach().cpu().numpy()
        syn_data_from_public_data = model.decode(model.encode(torch.tensor(X_test, dtype=torch.float32).to(device))[0]).detach().cpu().numpy()
        
        # inverse the one-hot data to categorical
        inversed_data = pd.DataFrame(my_util.inverse(syn_data, encoders), columns=keys)
        inversed_data_from_public_data = pd.DataFrame(my_util.inverse(syn_data_from_public_data, encoders), columns=keys)
        
        inversed_data.to_csv(save_data_dir / f"out_{i}.csv", index=None)
        inversed_data_from_public_data.to_csv(save_data_dir / f"out_public_{i}.csv", index=None)

    if not args.skip_ml:
        ml_task.exp_ml.run(parameters)
    
if __name__ == "__main__":
    main()