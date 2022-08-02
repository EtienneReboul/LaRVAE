import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import time

from transvae.tvae_util import *
from adjMatrix import getAdjMatrixFromSelfie #added by Zoe

def vae_data_gen(mols, props, char_dict, padded_len, use_adj=False, adj_weight=0.3):
    """
    Encodes input smiles to tensors with token ids

    Arguments:
        mols (np.array, req): Array containing input molecular structures
        props (np.array, req): Array containing scalar chemical property values
        char_dict (dict, req): Dictionary mapping tokens to integer id
    Returns:
        encoded_data (torch.tensor): Tensor containing encodings for each
                                     SMILES string
    """
    #define data dimension
    selfie_encoding_len = padded_len+1 #should be 59 tokens with start token
    flat_adjmatrix_len = selfie_encoding_len**2 #should be 59x59

    selfies = mols[:,0]
    if props is None:
        props = np.zeros(selfies.shape)
    del mols
    tokenized_selfies = [tokenizer(x) for x in selfies]

    #initialize size of data matrix
    if use_adj:
        encoded_data = torch.empty((len(tokenized_selfies), selfie_encoding_len+flat_adjmatrix_len+1)) #selfie, adjmatrix, prop
    else: 
        encoded_data = torch.empty((len(tokenized_selfies), selfie_encoding_len+1))
    
    #fill in data matrix
    for j, tokenized_selfie in enumerate(tokenized_selfies):
        encoded_selfie = encode_smiles(tokenized_selfie, selfie_encoding_len-1, char_dict) #should be length 60
        encoded_selfie = [0] + encoded_selfie #adding start token (length 59)
        if use_adj: #if adj_matrix
            atom_list, adjmatrix = getAdjMatrixFromSelfie(selfies[j], selfie_encoding_len, c=adj_weight)
            encoded_data[j,:-flat_adjmatrix_len-1] = torch.tensor(encoded_selfie)
            encoded_data[j,selfie_encoding_len:-1] = torch.tensor(adjmatrix.flatten())
            encoded_data[j,-1] = torch.tensor(props[j])
        else:
            encoded_data[j,:-1] = torch.tensor(encoded_selfie)
            encoded_data[j,-1] = torch.tensor(props[j])
    return encoded_data

def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
