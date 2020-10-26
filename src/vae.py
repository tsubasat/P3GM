import torch
import torchvision
import pathlib
filedir = pathlib.Path(__file__).resolve().parent


class VAE(torch.nn.Module):
    
    @classmethod
    def make_model(cls, dims, device, z_dim=10, latent_dim=1000):
        return cls(dims, device, z_dim, latent_dim).to(device)
    
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
        mean, logvar = self.encode(x)
        return self.decode(mean)

    def save_model(self, name):
        now = datetime.datetime.now()
        filename = now.strftime("VAE_" + '%Y%m%d_%H%M%S' + "_" + name)
        model_dir = os.path.join("models", self.db, filename)
        torch.save(self.state_dict(), model_dir)    
        
    def load_model(self, filename):
        model_dir = os.path.join("models", filename)
        self.load_state_dict(torch.load(model_dir))
        
    def z_sample(self, n_samples):
        samples = torch.Tensor(torch.randn(n_samples, self.z_dim)).to(self.device)
        return samples
    
    def generate_data(self, n_data):
        z = self.z_sample(n_data)
        data = self.decode(z)
        return data
    
    def loss_function(self, x, means, logvar):
        z = self.reparameterize(means, logvar)
        recon_x = self.decode(z)
        
        SE = torch.sum((recon_x - x) ** 2, dim=1)
        KLD = -0.5 * torch.sum(1 + logvar - means.pow(2) - logvar.exp(), dim=1)
        return SE, KLD    
    
    def backward(self, losses):
        losses.mean().backward()
        

    def _saveimg(self, sample_data, epoch):
        torchvision.utils.save_image(sample_data.view(-1, 1, 28, 28), filedir.parent / "result" / "imgs" / f"{epoch}.png", nrow=10)
    
    def train(self, train_loader, **kwargs):
        lr = kwargs.get('lr', 1e-3)
        epoch = kwargs.get("epoch", 10)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
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
                kld_loss += kld_losses.mean().item()
                
                print(f"EPOCH {i+1}/{epoch}: {train_loader.get_current_index() + 1}/{train_loader.get_n_batch()} recon_loss: {recon_losses.mean().item()} kld_loss: {kld_losses.mean().item()}\r", end="")
            
            if self.data_dim == 794:
                self._saveimg(self.generate_data(100)[:, :-10], i)
                
            print(f"\nAVERAGE recon loss: {recon_loss/train_loader.get_n_batch()} kld_loss: {kld_loss/train_loader.get_n_batch()}")