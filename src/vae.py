import torch
import torchvision
import pathlib
import pandas as pd
import my_util
filedir = pathlib.Path(__file__).resolve().parent


class VAE(torch.nn.Module):
    
    def __init__(self, dims, device, z_dim=10, latent_dim=1000):
        super().__init__()
        
        data_dim = sum(dims)
        self.fc1 = torch.nn.Linear(data_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, z_dim)
        self.fc22 = torch.nn.Linear(latent_dim, z_dim)
        self.fc3 = torch.nn.Linear(z_dim, latent_dim)
        self.fc4 = torch.nn.Linear(latent_dim, data_dim)
        self.dims, self.data_dim, self.z_dim, self.latent_dim = dims, data_dim, z_dim, latent_dim
        self.device = device
        
    def encode(self, x):
        h1 = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(h1), self.fc22(h1)

    def reparameterize(self, means, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(means)
    
    def decode(self, z):
        h3 = torch.nn.functional.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x):
        mean, _ = self.encode(x)
        return self.decode(mean)
        
    def z_sample(self, n_samples):
        samples = torch.Tensor(torch.randn(n_samples, self.z_dim)).to(self.device)
        return samples
    
    def generate_data(self, n_data):
        z = self.z_sample(n_data)
        return self.decode(z)

    def loss_function(self, x, means, logvar):
        z = self.reparameterize(means, logvar)
        recon_x = self.decode(z)
        
        SE = torch.sum((recon_x - x) ** 2, dim=1)
        KLD = -0.5 * torch.sum(1 + logvar - means.pow(2) - logvar.exp(), dim=1)
        return SE, KLD    
    
    def backward(self, losses):
        losses.mean().backward()

    def generate_data_to_csv(self, n_data, X_test, encoders, keys, i, save_data_dir, name, public=False):
        # generate synthetic data from the trained model
        syn_data = self.generate_data(n_data).detach().cpu().numpy()
        
        # inverse the one-hot data to categorical
        inversed_data = pd.DataFrame(my_util.inverse(syn_data, encoders), columns=keys)

        inversed_data.to_csv(save_data_dir / f"{name}_{i}.csv", index=None)

        if public:
            syn_data_from_public_data = self.decode(self.encode(torch.tensor(X_test, dtype=torch.float32).to(self.device))[0]).detach().cpu().numpy()
            inversed_data_from_public_data = pd.DataFrame(my_util.inverse(syn_data_from_public_data, encoders), columns=keys)
            inversed_data_from_public_data.to_csv(save_data_dir / f"{name}_public_{i}.csv", index=None)

    def _saveimg(self, sample_data, epoch):
        (filedir.parent / "result" / "imgs").mkdir(exist_ok=True)
        torchvision.utils.save_image(sample_data.view(-1, 1, 28, 28), filedir.parent / "result" / "imgs" / f"{epoch}.png", nrow=10)

    def train(self, train_loader, **kwargs):
        lr = kwargs.get('lr', 1e-3)
        epoch = kwargs.get("sgd_epoch", 10)
        sgd_sigma = kwargs.get("sgd_sigma")
        clipping = kwargs.get("clipping")
        num_microbatches = kwargs.get("num_microbatches")
        eval_each_epoch = kwargs.get("eval_each_epoch")

        self.sgd_sigma, self.clipping, self.num_microbatches = sgd_sigma, clipping, num_microbatches

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        print(kwargs)
        
        syn_data_dir_temp = pathlib.Path("/data/takagi") / "synthetic_data" / f"{kwargs['db']}" / f"{kwargs['time']}" / "temp"
        syn_data_dir_temp.mkdir(exist_ok=True)
        result_dir = filedir.parent / "result" / f"{kwargs['db']}" / f"{kwargs['time']}"

        log_recon_loss = []

        print(f"n_epoch:{epoch}")
        for i in range(epoch):
            recon_loss, kld_loss = 0, 0
            
            for data in train_loader:

                data = data.to(self.device)
                means, logvar = self.encode(data)

                recon_losses, kld_losses = self.loss_function(data, means, logvar)
                losses = recon_losses + kld_losses
                
                optimizer.zero_grad()
                self.backward(losses)
                optimizer.step()
                
                recon_loss += recon_losses.mean().item()
                log_recon_loss.append(recon_losses.mean().item())
                kld_loss += kld_losses.mean().item()
                
                print(f"EPOCH {i+1}/{epoch}: {train_loader.get_current_index() + 1}/{train_loader.get_n_batch()} recon_loss: {recon_losses.mean().item()} kld_loss: {kld_losses.mean().item()}\r", end="")

            if self.data_dim == 794:
                self._saveimg(self.generate_data(100)[:, :-10], i)

            if eval_each_epoch:
                self.generate_data_to_csv(train_loader.get_data_size(), train_loader.get_test_data(), train_loader.encoders, train_loader.keys, i, syn_data_dir_temp, f"temp_{kwargs['current_iter']}")

            print(f"\nAVERAGE recon loss: {recon_loss/train_loader.get_n_batch()} kld_loss: {kld_loss/train_loader.get_n_batch()}")

        with (result_dir / f"log_{kwargs['current_iter']}.txt").open("w") as f:
            f.write(",".join([str(v) for v in log_recon_loss]))