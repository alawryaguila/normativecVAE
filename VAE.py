import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Parameter


def compute_ll(x, x_recon):
    return x_recon.log_prob(x).sum(1, keepdims=True).mean(0)

class Encoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim, 
                variational=True, 
                non_linear=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.z_dim = hidden_dim[-1]
        self.variational = variational
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim] + hidden_dim
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:])]
               
        self.encoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.enc_mean_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)
        self.enc_logvar_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)

        mu = self.enc_mean_layer(h1)
        logvar = self.enc_logvar_layer(h1)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim,
                non_linear=False, 
                init_logvar=-3):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.layer_sizes_decoder = hidden_dim[::-1] + [input_dim]
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:])]
        self.decoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.decoder_mean_layer = nn.Linear(self.layer_sizes_decoder[-2],self.layer_sizes_decoder[-1], bias=True)
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)


    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)

        mu_out = self.decoder_mean_layer(x_rec)
        return Normal(loc=mu_out, scale=self.logvar_out.exp().pow(0.5))

class VAE(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                latent_dim, 
                learning_rate=0.001, 
                non_linear=False):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.non_linear = non_linear
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=self.hidden_dim, non_linear=non_linear) 
        self.decoder = Decoder(input_dim=input_dim, hidden_dim=self.hidden_dim, non_linear=non_linear)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate) 
    
    def encode(self, x):
        return self.encoder(x)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def calc_kl(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def forward(self, x):
        self.zero_grad()
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)
    
        total = kl - recon
        losses = {'total': total,
                'kl': kl,
                'll': recon}
        return losses

    def pred_latent(self, x, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        with torch.no_grad():
            mu, logvar = self.encode(x)   
        latent = mu.cpu().detach().numpy()
        latent_var = logvar.exp().cpu().detach().numpy()
        return latent, latent_var

    def pred_recon(self, x, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.encode(x)
            x_pred = self.decode(mu).loc.cpu().detach().numpy()
        return x_pred