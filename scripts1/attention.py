import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from transvae.trans_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn

from transvae.data import vae_data_gen
from scripts1.parsers1 import attn_parser

def calc_attention(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    elif args.model == 'rnn':
        vae = RNN(load_fn=ckpt_fn)

    if args.shuffle:
        data = pd.read_csv(args.mols).sample(args.n_samples).to_numpy()
    else:
        data = pd.read_csv(args.mols).to_numpy()
        data = data[:args.n_samples,:]

    ### Load data and prepare for iteration
    data = vae_data_gen(data, props=None, char_dict=vae.params['CHAR_DICT'])
    data_iter = torch.utils.data.DataLoader(data,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=0,
                                            pin_memory=False, drop_last=True)
    save_shape = len(data_iter)*args.batch_size
    chunk_size = args.batch_size // args.batch_chunks

    ### Prepare save path
    if args.save_path is None:
        os.makedirs('attn_wts', exist_ok=True)
        save_path = 'attn_wts/{}'.format(vae.name)
    else:
        save_path = args.save_path

    ### Calculate attention weights
    vae.model.eval()
    if args.model == 'rnnattn':
        attn = torch.empty((save_shape, 1, 1, 127, 127))
        for j, data in enumerate(data_iter):
            for i in range(args.batch_chunks):
                batch_data = data[i*chunk_size:(i+1)*chunk_size,:]
                mols_data = batch_data[:,:-1]
                props_data = batch_data[:,-1]
                if vae.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()

                # Run samples through model to calculate weights
                mem, mu, logvar, attn_wts = vae.model.encoder(vae.model.src_embed(src), return_attn=True)
                start = j*args.batch_size+i*chunk_size
                stop = j*args.batch_size+(i+1)*chunk_size
                attn[start:stop,0,0,:,:] = attn_wts

        np.save(save_path+'.npy', attn.numpy())


if __name__ == '__main__':
    parser = attn_parser()
    args = parser.parse_args()
    calc_attention(args)
