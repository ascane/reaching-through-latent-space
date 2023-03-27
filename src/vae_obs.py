from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class VAEObstacleBCE(nn.Module):

    def __init__(self, input_dim, latent_dim, units_per_layer, num_hidden_layers):

        super(VAEObstacleBCE, self).__init__()

        # joint angles (dof) + ee position (3)
        self.input_dim = input_dim  
        self.latent_dim = latent_dim
        self.obs_dim = 4  # x y h r

        # encoder
        self.fc1 = nn.Linear(self.input_dim, units_per_layer)
        self.fc_encoder = nn.ModuleList([nn.Linear(units_per_layer, units_per_layer) for i in range(num_hidden_layers - 1)])
        self.fc21 = nn.Linear(units_per_layer, self.latent_dim)
        self.fc22 = nn.Linear(units_per_layer, self.latent_dim)

        # decoder
        self.fc31 = nn.Linear(self.latent_dim, units_per_layer)
        self.fc_decoder = nn.ModuleList([nn.Linear(units_per_layer, units_per_layer) for i in range(num_hidden_layers - 1)])
        self.fc41 = nn.Linear(units_per_layer, self.input_dim)

        # binary obstacle classifier -> probability of collision
        self.fc32 = nn.Linear(self.latent_dim + self.obs_dim, units_per_layer)
        self.fc_obs = nn.ModuleList([nn.Linear(units_per_layer, units_per_layer) for i in range(num_hidden_layers - 1)])
        self.fc42 = nn.Linear(units_per_layer, 1)

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

    def obstacle_collision_classifier(self, z, obs):

        h = self.fc32(torch.cat((z.view(-1, self.latent_dim), obs.view(-1, self.obs_dim)), dim=1))
        for fc in self.fc_obs:
            h = fc(F.elu(h))

        return self.fc42(F.elu(h)).view(-1)

    def forward(self, x, obs=None):

        z, mu, logVar = self.encoder(x.view(-1, self.input_dim))

        xPrime = self.decoder(z)

        obs_logit = None
        if obs is not None:
            obs_logit = self.obstacle_collision_classifier(z, obs)

        return xPrime, mu, logVar, obs_logit

    def get_features(self, x):

        with torch.no_grad():
            (z, _, _) = self.encoder(x.view(-1, self.input_dim))
        return z

    def get_reconstruction(self, x):

        with torch.no_grad():
            (z, _, _) = self.encoder(x.view(-1, self.input_dim))
            xPrime = self.decoder(z)

            return xPrime

    def get_reconstruction_and_pred(self, x, obs):

        with torch.no_grad():
            (z, _, _) = self.encoder(x.view(-1, self.input_dim))
            obs_logit = self.obstacle_collision_classifier(z, obs)
            xPrime = self.decoder(z)

            return xPrime, obs_logit

    def get_pred(self, z, obs):
         with torch.no_grad():
            obs_logit = self.obstacle_collision_classifier(z, obs)

            return obs_logit

    def get_recon_from_latent(self, z):

        with torch.no_grad():
            xPrime = self.decoder(z)

            return xPrime
