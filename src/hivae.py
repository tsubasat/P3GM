import torch
import functools
from vae import VAE


class HIVAE(VAE):

    def __init__(self, dims, device, z_dim=10, latent_dim=1000):
        super().__init__(dims, device, z_dim=z_dim, latent_dim=latent_dim)
        
        self.dims = dims
        self.reduce = dims[0] == 1
        
        if self.reduce:
            def reduce_initial(x,y):
                if y==1:
                    return [x[0] + 1]
                else:
                    return x + [y]
            self.dims = functools.reduce(reduce_initial, dims, [0])
        else:
            self.dims = dims
        
        print(f"DIMENSION: {self.dims}")
        self.fc4 = torch.nn.ModuleList([torch.nn.Linear(latent_dim, dim) for dim in self.dims])
        
    def decode_to_array(self, z):
        h3 = torch.nn.functional.relu(self.fc3(z))
        logits = []
        for layer in self.fc4:
            logits.append(layer(h3))
        return logits
    
    def decode(self, z):
        return torch.cat(self.decode_to_array(z), 1)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        return self.decode(mean)
    
    def generate_data(self, n_data):
        z = self.z_sample(n_data)
        return self.decode(z)
    
    def loss_function(self, data, means, logvar):
        z = self.reparameterize(means, logvar)
        logits = self.decode_to_array(z)
        
        bce = torch.nn.CrossEntropyLoss()
        counter = 0
        SE = torch.tensor([0]*data.shape[0], dtype=torch.float32).to(self.device)
        for i, (dim, logit) in enumerate(zip(self.dims, logits)):
            if (i==0 and self.reduce) or dim == 1:
                SE += torch.sum((logit - data[:, counter:counter+dim]) ** 2, dim=1)
            else:
                inferred_label = torch.max(data[:, counter:counter+dim], 1)[1]
                SE += bce(logit, inferred_label)
            counter += dim
        KLD = -0.5 * torch.sum(1 + logvar - means.pow(2) - logvar.exp(), dim=1)
        return SE, KLD