
import sys, os
import random
import math
import argparse
from tqdm import tqdm
from datetime import datetime 

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.append("../gqn-wohlert")
from gqn import GenerativeQueryNetwork
from datasets import Face3D, transform_viewpoint


def sample(model, context_x, context_v, viewpoint, sigma):
    """
    Sample from the network given some context and viewpoint.

    :param context_x: set of context images to generate representation
    :param context_v: viewpoints of `context_x`
    :param viewpoint: viewpoint to generate image from, (n_batch, v_dim)
    :param sigma: pixel variance
    """
    with torch.no_grad():
        batch_size, n_views, _, h, w = context_x.size()

        _, _, *x_dims = context_x.size()
        _, _, *v_dims = context_v.size()

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = model.representation(x, v)

        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)

        x_mu = model.generator.sample((h, w), viewpoint, r)
        x_sample = x_mu 
        # Due to the fact that we do not learn per-pixel variances, 
        # in figures, we show the mean value of each pixel conditioned on the sampled latent variables.
        # x_sample = Normal(x_mu, sigma).sample()
    return x_sample # (n_batch, *img_shape)


def sample_multiview(model, context_x, context_v, viewpoint, sigma):
    """
    Sample from the network given some context and viewpoint.

    :param context_x: set of context images to generate representation
    :param context_v: viewpoints of `context_x`, (n_batch, n_viewpoints, v_dim)
    :param viewpoint: viewpoint to generate image from
    :param sigma: pixel variance
    """
    with torch.no_grad():
        batch_size, n_views, _, h, w = context_x.size()

        _, _, *x_dims = context_x.size()
        _, _, *v_dims = context_v.size()

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = model.representation(x, v)

        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)

        x_sample = []
        for i in range(viewpoint.shape[1]):
            x_mu = model.generator.sample((h, w), viewpoint[:,i,:], r)
            x_sample.append(x_mu)
        x_sample = torch.stack(x_sample)

        x_sample = x_sample.transpose(0,1)
        # Due to the fact that we do not learn per-pixel variances, 
        # in figures, we show the mean value of each pixel conditioned on the sampled latent variables.
        # x_sample = Normal(x_mu, sigma).sample()
    return x_sample  # (n_batch, n_viewpoints, *img_shape)

def render_viewpoints_1(model, x, v, sigma, img_path):
    '''
    x: (n_context, *imgshape)
    v: (n_context, *v_dim)
    '''
    x, v = x.reshape(1, *x.shape), v.reshape(1, *v.shape)
    meshgrid = np.meshgrid(np.arange(-45, 46, 15), np.arange(-45, 46, 15))
    viewpoints = [(y, x, 0) for x, y in zip(*[x.ravel() for x in meshgrid])]
    viewpoints = torch.from_numpy(np.array(viewpoints)).float()
    viewpoints = viewpoints.reshape(1, *viewpoints.shape)
    viewpoints = transform_viewpoint(viewpoints)
    out = sample_multiview(model, x, v, viewpoints, sigma)
    save_image(out[0], img_path, nrow=meshgrid[0].shape[0])

def render_viewpoints_2(model, x, v, sigma, img_path):
    '''
    x: (n_context, *imgshape)
    v: (n_context, *v_dim)
    '''
    x, v = x.reshape(1, *x.shape), v.reshape(1, *v.shape)
    meshgrid = np.meshgrid(np.arange(-45, 46, 15), np.arange(-45, 46, 15))
    viewpoints = [(0, x, -y) for x, y in zip(*[x.ravel() for x in meshgrid])]
    viewpoints = torch.from_numpy(np.array(viewpoints)).float()
    viewpoints = viewpoints.reshape(1, *viewpoints.shape)
    viewpoints = transform_viewpoint(viewpoints)
    out = sample_multiview(model, x, v, viewpoints, sigma)
    save_image(out[0], img_path, nrow=meshgrid[0].shape[0])

def find_viewpoint_input(x, v, v_target):
    transform_v = transform_viewpoint(torch.Tensor(v_target))
    mask = torch.all(v == transform_v, dim=1)
    if torch.sum(mask) == 0:
        return None, None
    elif torch.sum(mask) > 1:
        return x[mask][0], v[mask][0]
    return x[mask], v[mask]
    
def single_context_rendering(model, dataset, idx, target_v, sigma, figdir='figures', device='cpu'):
    x_all, v_all = dataset[idx]
    x_all, v_all = x_all.to(device), v_all.to(device)
    x, v = find_viewpoint_input(x_all, v_all, target_v)
    
    if x is None:
        raise ValueError("target viewpoint not found in dataset.")
        return None
    if not os.path.exists(figdir):
        os.mkdir(figdir)
    
    render_viewpoints_1(model, x, v, sigma, os.path.join(figdir, 'data{}_single_ctxt_{}_{}_{}_render1.jpg'.format(idx, *target_v)))
    render_viewpoints_2(model, x, v, sigma, os.path.join(figdir, 'data{}_single_ctxt_{}_{}_{}_render2.jpg'.format(idx, *target_v)))

def multi_context_rendering(model, dataset, idx, n_context, sigma, figdir='figures', device='cpu'):
    x_all, v_all = dataset[idx]
    x_all, v_all = x_all.to(device), v_all.to(device)
    x, v = x_all[:n_context], v_all[:n_context]
    if not os.path.exists(figdir):
        os.mkdir(figdir)
    
    render_viewpoints_1(model, x, v, sigma, os.path.join(figdir, 'data{}_multi_ctxt_{}_render1.jpg'.format(idx, n_context)))
    render_viewpoints_2(model, x, v, sigma, os.path.join(figdir, 'data{}_multi_ctxt_{}_render2.jpg'.format(idx, n_context)))
    