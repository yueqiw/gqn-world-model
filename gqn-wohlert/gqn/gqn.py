import random

import torch
import torch.nn as nn
from torch.distributions import Normal

from .representation import TowerRepresentation, PyramidRepresentation
from .generator import GeneratorNetwork, Conv2dLSTMCell


class GenerativeQueryNetwork(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12, pool=False):
        super(GenerativeQueryNetwork, self).__init__()
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim, pool=pool)
        #self.representation = PyramidRepresentation(x_dim, v_dim, r_dim)

    def forward(self, images, viewpoints):
        """
        Forward through the GQN.

        :param images: batch of images [b, m, c, h, w]
        :param viewpoints: batch of viewpoints for image [b, m, k]
        """
        # Number of context datapoints to use for representation
        batch_size, m, *_ = viewpoints.size()

        # Sample random number of views and generate representation
        n_views = random.randint(2, m-1)

        indices = torch.randperm(m)
        representation_idx, query_idx = indices[:n_views], indices[n_views]

        x, v = images[:, representation_idx], viewpoints[:, representation_idx]

        # Merge batch and view dimensions.
        _, _, *x_dims = x.size()
        _, _, *v_dims = v.size()

        x = x.view((-1, *x_dims))
        v = v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        # sum over view representations
        r = torch.sum(phi, dim=1)

        # Use random (image, viewpoint) pair in batch as query
        x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]
        x_mu, kl = self.generator(x_q, v_q, r)

        # Return reconstruction and query viewpoint
        # for computing error
        return [x_mu, x_q, r, kl]

    def sample(self, context_x, context_v, viewpoint, sigma):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param viewpoint: viewpoint to generate image from
        :param sigma: pixel variance
        """
        with torch.no_grad():
            batch_size, n_views, _, h, w = context_x.size()
            
            _, _, *x_dims = context_x.size()
            _, _, *v_dims = context_v.size()

            x = context_x.view((-1, *x_dims))
            v = context_v.view((-1, *v_dims))

            phi = self.representation(x, v)

            _, *phi_dims = phi.size()
            phi = phi.view((batch_size, n_views, *phi_dims))

            r = torch.sum(phi, dim=1)

            x_mu = self.generator.sample((h, w), viewpoint, r)
            x_sample = x_mu 
        # Due to the fact that we do not learn per-pixel variances, 
        # in figures, we show the mean value of each pixel conditioned on the sampled latent variables.
        # x_sample = Normal(x_mu, sigma).sample()
        return x_sample


class GQNTimeSeriesSum(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12, pool=False):
        super(GQNTimeSeriesSum, self).__init__()
        self.r_dim = r_dim
        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim, pool=pool)
        #self.representation = PyramidRepresentation(x_dim, v_dim, r_dim)

    def forward(self, images, viewpoints, random_n=True):
        """
        Forward through the GQN.

        :param images: batch of images [b, m, c, h, w]
        :param viewpoints: batch of viewpoints for image [b, m, k]
        """
        # Number of context datapoints to use for representation
        batch_size, m, *_ = viewpoints.size()

        # Sample random number of views and generate representation
        if random_n:
            n_views = random.randint(2, m-1)
        else:
            n_views = m-1 

        indices = torch.arange(m)
        representation_idx, query_idx = indices[:n_views], indices[n_views]

        x, v = images[:, representation_idx], viewpoints[:, representation_idx]

        # Merge batch and view dimensions.
        _, _, *x_dims = x.size()
        _, _, *v_dims = v.size()

        x = x.view((-1, *x_dims))
        v = v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        # sum over view representations
        r = torch.sum(phi, dim=1)

        # Use random (image, viewpoint) pair in batch as query
        x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]
        x_mu, kl = self.generator(x_q, v_q, r)

        # Return reconstruction and query viewpoint
        # for computing error
        return [x_mu, x_q, r, kl]

    def sample(self, context_x, context_v, viewpoint):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param viewpoint: viewpoint to generate image from
        :param sigma: pixel variance
        """
        with torch.no_grad():
            batch_size, n_views, _, h, w = context_x.size()
            
            _, _, *x_dims = context_x.size()
            _, _, *v_dims = context_v.size()

            x = context_x.view((-1, *x_dims))
            v = context_v.view((-1, *v_dims))

            phi = self.representation(x, v)

            _, *phi_dims = phi.size()
            phi = phi.view((batch_size, n_views, *phi_dims))

            r = torch.sum(phi, dim=1)

            x_mu = self.generator.sample((h, w), viewpoint, r)
            x_sample = x_mu 
        # Due to the fact that we do not learn per-pixel variances, 
        # in figures, we show the mean value of each pixel conditioned on the sampled latent variables.
        # x_sample = Normal(x_mu, sigma).sample()
        return x_sample



class GQNTimeSeriesLSTM(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12, pool=False):
        super(GQNTimeSeriesLSTM, self).__init__()
        self.r_dim = r_dim
        self.lstm_aggregator = Conv2dLSTMCell(r_dim, r_dim, kernel_size=5, stride=1, padding=2)
        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim, pool=pool)
        #self.representation = PyramidRepresentation(x_dim, v_dim, r_dim)

    def forward(self, images, viewpoints, random_n=True):
        """
        Forward through the GQN.

        :param images: batch of images [b, m, c, h, w]
        :param viewpoints: batch of viewpoints for image [b, m, k]
        """
        # Number of context datapoints to use for representation
        batch_size, m, *_ = viewpoints.size()

        # Sample random number of views and generate representation
        if random_n:
            n_views = random.randint(2, m-1)
        else:
            n_views = m-1 

        indices = torch.arange(m)
        representation_idx, query_idx = indices[:n_views], indices[n_views]

        x, v = images[:, representation_idx], viewpoints[:, representation_idx]

        # Merge batch and view dimensions.
        _, _, *x_dims = x.size()
        _, _, *v_dims = v.size()

        x = x.view((-1, *x_dims))
        v = v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        # sum over view representations
        # r = torch.sum(phi, dim=1)

        # lstm aggregator of view representations 
        hidden_a = x.new_zeros((batch_size, *phi_dims))
        cell_a = x.new_zeros((batch_size, *phi_dims))

        for i in range(n_views):
            hidden_a, cell_a = self.lstm_aggregator(phi[:, i], [hidden_a, cell_a])
        r = hidden_a

        # Use random (image, viewpoint) pair in batch as query
        x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]
        x_mu, kl = self.generator(x_q, v_q, r)

        # Return reconstruction and query viewpoint
        # for computing error
        return [x_mu, x_q, r, kl]

    def sample(self, context_x, context_v, viewpoint):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param viewpoint: viewpoint to generate image from
        :param sigma: pixel variance
        """
        with torch.no_grad():
            batch_size, n_views, _, h, w = context_x.size()
            
            _, _, *x_dims = context_x.size()
            _, _, *v_dims = context_v.size()

            x = context_x.view((-1, *x_dims))
            v = context_v.view((-1, *v_dims))

            phi = self.representation(x, v)

            _, *phi_dims = phi.size()
            phi = phi.view((batch_size, n_views, *phi_dims))

            # lstm aggregator of view representations 
            hidden_a = x.new_zeros((batch_size, *phi_dims))
            cell_a = x.new_zeros((batch_size, *phi_dims))
            for i in range(n_views):
                hidden_a, cell_a = self.lstm_aggregator(phi[:, i], [hidden_a, cell_a])
            r = hidden_a

            x_mu = self.generator.sample((h, w), viewpoint, r)
            x_sample = x_mu 
        # Due to the fact that we do not learn per-pixel variances, 
        # in figures, we show the mean value of each pixel conditioned on the sampled latent variables.
        # x_sample = Normal(x_mu, sigma).sample()
        return x_sample