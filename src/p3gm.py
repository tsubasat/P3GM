import hivae
import vae
import sys
import torch
import functools
import my_util
import math
import pathlib
import numpy as np
filedir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(filedir.parent / "privacy"))
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

sys.path.append(str(filedir.parent))
import dp_utils

# The method to compute the approximate KL divergence between Gaussian and MoG.
# MoG = [pi, mu, var]
def _p3gm_kl_divergence(gmm1, gmm2):
    mu1, var1 = gmm1[0], gmm1[1]
    pr_klds = []
    for i, (pi2, mu2, var2) in enumerate(zip(*gmm2)):
        kld = _kl_divergence((mu1, var1),(mu2, var2))
        pr_kld = -kld + torch.log(pi2)
        pr_klds.append(pr_kld)
    pr_klds = torch.stack(pr_klds)
    max_values = torch.max(pr_klds, dim=0)[0].reshape(1,-1)
    pr_klds = pr_klds - max_values
    var_kld = - max_values - torch.log(torch.sum(torch.exp(pr_klds), dim=0))
    return var_kld.reshape(-1)

# The method to compute the KL divergence between two Gaussians.
# gauss = [mu, var]
def _kl_divergence(gauss1, gauss2):
    mu1, var1 = gauss1[0], gauss1[1]
    mu2, var2 = gauss2[0].reshape(1,-1), gauss2[1].reshape(1,-1)
    
    dim = mu1.shape[1]
    
    log_var2 = torch.log(var2)
    log_var1 = torch.log(var1)
    
    term1 = torch.sum(log_var2, dim=1) - torch.sum(log_var1, dim=1)
    term2 = torch.mm(1/var2, torch.t(var1))[0]
    term3 = torch.diag(torch.mm((mu2 - mu1) * (1/var2), torch.t(mu2 - mu1)))
    
    return (1/2) * (term1 - dim + term2 + term3)

# The method to compute the sum of the privacy budget for P3GM.
# This method is depending on the tensorflow.privacy library
def analysis_privacy(lot_size, data_size, sgd_sigma, gmm_sigma, gmm_iter, gmm_n_comp, sgd_epoch, pca_eps, delta=1e-5):
    q = lot_size / data_size
    sgd_steps = int(math.ceil(sgd_epoch * data_size / lot_size))
    gmm_steps = gmm_iter * (2 * gmm_n_comp + 1)
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
    pca_rdp = np.array(orders) * 2 * (pca_eps**2)
    sgd_rdp = compute_rdp(q, sgd_sigma, sgd_steps, orders)
    gmm_rdp = compute_rdp(1, gmm_sigma, gmm_steps, orders)
    
    rdp = pca_rdp + gmm_rdp + sgd_rdp
    
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    
    index = orders.index(opt_order)
    print(f"ratio(pca:gmm:sgd):{pca_rdp[index]/rdp[index]}:{gmm_rdp[index]/rdp[index]}:{sgd_rdp[index]/rdp[index]}")
    print(f"GMM + SGD + PCA (MA): {eps}, {delta}-DP")
    
    return eps, [pca_rdp[index]/rdp[index], gmm_rdp[index]/rdp[index], sgd_rdp[index]/rdp[index]]


# The method to construct the P3GM class.
# We prepare two types of P3GM which are depending on VAE and HI-VAE.
# Refer to https://arxiv.org/abs/1807.03653 for HI-VAE.
def make_P3GM(model):
    
    # The P3GM class which inherets a model class (VAE and HI-VAE)
    class P3GM(model):

        # The staticmethod to compute the sum of the privacy budget (This refers to the analysis_privacy method)
        @staticmethod
        def cp_epsilon(data_size, lot_size, pca_eps, gmm_sigma, gmm_iter, gmm_n_comp, sgd_sigma, sgd_epoch, delta=1e-5):
            return analysis_privacy(lot_size, data_size, sgd_sigma, gmm_sigma, gmm_iter, gmm_n_comp, sgd_epoch, pca_eps, delta=delta)[0]

        # The initilization method to construct networks.
        # z_dim is the number of components for PCA (the dimensionality of z)
        # latent_din is the number of nodes for hidden layers
        def __init__(self, dims, device, z_dim=10, latent_dim=1000):
            if z_dim > sum(dims):
                z_dim = sum(dims)
            super().__init__(dims, device, z_dim=z_dim, latent_dim=latent_dim)


        # The method to train P3GM.
        # 1. fit PCA 2. fit GMM, 3. train neural networks
        def train(self, train_loader, random_state, **kwargs):
            
            # hyperparameters
            pca_eps = kwargs.get("pca_eps", 1e-2)
            gmm_sigma = kwargs.get("gmm_sigma", 100)
            gmm_n_comp = kwargs.get("gmm_n_comp", 3)
            gmm_iter = kwargs.get("gmm_iter", 20)
            sgd_epoch = kwargs.get("sgd_epoch", 3)
            sgd_sigma = kwargs.get("sgd_sigma", 1.3)
            delta = kwargs.get("delta", 1e-5)
            clipping = kwargs.get("clipping", 1)
            num_microbatches = kwargs.get("num_microbatches", 3)
            no_dp = kwargs.get("no_dp", False)
            lr = kwargs.get("lr", 1e-3)

            self.sgd_sigma, self.clipping, self.num_microbatches, self.no_dp = sgd_sigma, clipping, num_microbatches, no_dp

            # compute the sum of privacy budgets using RDP
            print(P3GM.cp_epsilon(train_loader.get_data_size(), train_loader.get_batch_size(), pca_eps, gmm_sigma, gmm_iter, gmm_n_comp, sgd_sigma, sgd_epoch))

            data = train_loader.get_data()
            
            # train PCA
            self._train_pca(data, pca_eps, self.z_dim, random_state)
            feature = self.pca.transform(data)

            # train GMM
            self._train_gmm(feature, gmm_sigma, gmm_n_comp, gmm_iter, random_state)
            self._set_parameters()
            
            self.reduce = True
            
            # train neural networks
            super().train(train_loader, sigma=sgd_sigma, epoch=sgd_epoch, lr=lr)

        def _transform_pca(self, X):
            X = X - self.pca_mean_
            X_transformed = torch.mm(X, self.pca_components_)
            return X_transformed

        def encode(self, x):
            h1 = torch.nn.functional.relu(self.fc1(x))
            return self._transform_pca(x), self.fc22(h1)

        def z_sample(self, n_samples):
            return torch.tensor(self.gmm.sample(n_samples=n_samples)[0], dtype=torch.float32).to(self.device)

        def _set_parameters(self):
            self.gmm_weights_ = torch.nn.Parameter(torch.tensor(self.gmm.weights_, dtype=torch.float32), requires_grad=False).to(self.device)
            self.gmm_means_ = torch.nn.Parameter(torch.tensor(self.gmm.means_, dtype=torch.float32), requires_grad=False).to(self.device)
            self.gmm_covariances_ = torch.nn.Parameter(torch.tensor(self.gmm.covariances_, dtype=torch.float32), requires_grad=False).to(self.device)
            self.gmm_params = [self.gmm_weights_, self.gmm_means_, self.gmm_covariances_]
            self.pca_components_ = torch.nn.Parameter(torch.t(torch.tensor(self.pca.components_[:self.pca.n_components], dtype=torch.float32)), requires_grad=False).to(self.device)
            self.pca_mean_ = torch.nn.Parameter(torch.tensor(self.pca.mean_, dtype=torch.float32), requires_grad=False).to(self.device)

        def _train_pca(self, data, epsilon, n_comp, random_state):
            self.pca = dp_utils.dp_pca.DP_PCA(eps=epsilon, n_components=n_comp, random_state=random_state)
            self.pca.fit(data)

        def _train_gmm(self, feature, gmm_sigma, gmm_n_comp, gmm_iter, random_state):
            self.gmm = dp_utils.dp_gaussian_mixture.DPGaussianMixture(sigma=gmm_sigma, n_components=gmm_n_comp, n_iter=gmm_iter, random_state=random_state)
            self.gmm.fit(feature)

        def loss_function(self, data, means, logvar):
            logits = self._sample_and_recon(means, logvar)
            
            bce = torch.nn.CrossEntropyLoss(reduction='none')
            counter = 0
            n_samples = len(logits[0])
            n_batch = len(logits[0][0])
            re_loss = torch.tensor([0]*n_batch, dtype=torch.float32).to(self.device)
            for i, (dim, logit) in enumerate(zip(self.dims, logits)):
                # if the value is not categorical, we use MSE loss.
                if (i==0 and self.reduce) or dim == 1:
                    re_loss += torch.sum((logit - data[:, counter:counter+dim]) ** 2, dim=2).mean(dim=0)
                else:
                # if the value is categorical, we use BCE loss
                    logit = logit.reshape(logit.shape[0]*logit.shape[1],-1)
                    label = torch.stack([torch.max(data[:, counter:counter+dim], 1)[1] for _ in range(n_samples)]).reshape(-1)
                    re_loss += bce(logit, label).reshape(n_samples, n_batch, -1).mean(dim=0).reshape(-1)
                counter += dim
            KLD = _p3gm_kl_divergence([means, torch.exp(logvar)], self.gmm_params)
            return re_loss, KLD


        # The method to backward the neural networks
        # P3GM uses noised step for DP-SGD
        def backward(self, losses, **kwargs):
            if not self.no_dp:
                self._noised_step(losses, self.sgd_sigma, self.clipping, self.num_microbatches)
            else:
                super().backward(losses)

        # The method for the implementation of DP-SGD
        def _noised_step(self, losses, sigma, clipping, num_microbatches=3):

            saved_var = dict()
            parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
            named_parameters = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))

            for tensor_name, tensor in named_parameters:
                saved_var[tensor_name] = torch.zeros_like(tensor)

            n_batch = len(losses)
            n_loss_in_microbatch = math.ceil(n_batch / num_microbatches)

            for i in range(num_microbatches):
                loss = losses[i*n_loss_in_microbatch:(i+1)*n_loss_in_microbatch].mean()
                loss.backward(retain_graph=True)

                norm = torch.nn.utils.clip_grad_norm_(parameters, clipping)
                for tensor_name, tensor in named_parameters:
                    new_grad = tensor.grad
                    if tensor.grad is None:
                        continue
                    saved_var[tensor_name].add_(new_grad)
                self.zero_grad()

            for tensor_name, tensor in named_parameters:
                if tensor.grad is None:
                    continue
                noise = torch.FloatTensor(tensor.grad.shape).normal_(0, sigma * clipping).to(self.device)
                saved_var[tensor_name].add_(noise)
                tensor.grad = saved_var[tensor_name] / num_microbatches

        def decode_to_array(self, z):
            if hasattr(super(), "decode_to_array"):
                return super().decode_to_array(z)
            else:
                return [super().decode(z)]
            
        # The method to sample random variable for Monte Carlo iterations.
        def _sample_and_recon(self, means, logvar, n_samples=100):
            std = torch.exp(0.5*logvar)
            eps = torch.randn(n_samples, means.size(0), means.size(1)).to(self.device)
            z = eps.mul(std).add_(means)
            zz = z.view(-1, means.size(1))
            recon_samples = self.decode_to_array(zz)
            return [recon_sample.view(n_samples, means.size(0), -1) for recon_sample in recon_samples]
    
    return P3GM
        
        
HIP3GM = make_P3GM(hivae.HIVAE)
P3GM = make_P3GM(vae.VAE)
