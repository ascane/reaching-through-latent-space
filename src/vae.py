from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim, units_per_layer, num_hidden_layers):

        super(VAE, self).__init__()

        # joint angles (dof) + ee position (3)
        self.input_dim = input_dim  
        self.latent_dim = latent_dim

        # encoder
        self.fc1 = nn.Linear(self.input_dim, units_per_layer)
        self.fc_encoder = nn.ModuleList([nn.Linear(units_per_layer, units_per_layer) for i in range(num_hidden_layers - 1)])
        self.fc21 = nn.Linear(units_per_layer, self.latent_dim)
        self.fc22 = nn.Linear(units_per_layer, self.latent_dim)

        # decoder
        self.fc31 = nn.Linear(self.latent_dim, units_per_layer)
        self.fc_decoder = nn.ModuleList([nn.Linear(units_per_layer, units_per_layer) for i in range(num_hidden_layers - 1)])
        self.fc41 = nn.Linear(units_per_layer, self.input_dim)

    def encoder(self, x):

        h = F.elu(self.fc1(x))
        for fc in self.fc_encoder:
            h = F.elu(fc(h))
        mu = self.fc21(h)

        var = F.softplus(self.fc22(h)) + 1e-5
        std = torch.sqrt(var)
        eps = torch.randn_like(std)

        logVar = torch.log(std**2)

        z = mu + eps * std
        return z, mu, logVar

    def decoder(self, z):

        h = F.elu(self.fc31(z))
        for fc in self.fc_decoder:
            h = F.elu(fc(h))
        h = self.fc41(h)
        return h

    def forward(self, x, obs=None):

        z, mu, logVar = self.encoder(x.view(-1, self.input_dim))

        xPrime = self.decoder(z)

        return xPrime, mu, logVar

    def get_features(self, x):

        with torch.no_grad():
            (z, _, _) = self.encoder(x.view(-1, self.input_dim))
            return z

    def get_reconstruction(self, x):

        with torch.no_grad():
            (z, _, _) = self.encoder(x.view(-1, self.input_dim))
            xPrime = self.decoder(z)
            return xPrime

    def get_recon_from_latent(self, z):

        with torch.no_grad():
            xPrime = self.decoder(z)
            return xPrime
