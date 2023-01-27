import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()


def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1,MMD_use=False,latent_size=28,device=None):
    
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    ### compute Binary Cross Entropy Loss
    x = x.long()[:,1:] - 1 #get rid of start token and shift by 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    
    ### compute predictor loss if used
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    
    ### either compute Kiebler-Lublach Divergence if Elbo-VAE or Maximum Mean Discrepency if MMD-VAE
    if MMD_use:
        MMD_loss=MMD(torch.randn(200, latent_size, requires_grad = False).to(device), mu)
        return BCE+MMD_loss+MSE,BCE,MMD_loss,MSE

    else:
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) #detleted Beta coeff - Zoe
        if torch.isnan(KLD):
            KLD = torch.tensor(0.)
        return BCE + beta * KLD + MSE, BCE, KLD, beta*KLD, MSE #changed output -Zoe

def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence + Mask Length Prediction"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCEmol + BCEmask + KLD + MSE, BCEmol, BCEmask, KLD, MSE


