import time

import os
import pickle
#import pkg_resources
import sys
import numpy as np
import pandas as pd


from torch import load, device
script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

#from transvae.trans_models import TransVAE
#from transvae.rnn_models import RNN, RNNAttn
start_time = time.time()
from scripts1.parsers1 import model_init
print("model init: " + str(time.time() - start_time))
start_time = time.time()
from scripts1.parsers1 import train_parser
print("parsers: " + str(time.time() - start_time))

def train(args):
    print("Entered train.py: " + str(time.time() - start_time))

    ### Update beta init parameter
    if args.checkpoint is not None:
        ckpt = load(args.checkpoint, map_location=device('cpu'))
        start_epoch = ckpt['epoch']
        total_epochs = start_epoch + args.epochs
        beta_init = (args.beta - args.beta_init) / total_epochs * start_epoch
        args.beta_init = beta_init

    ### Build params dict
    params = {'ADAM_LR': args.adam_lr,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'LR_SCALE': args.lr_scale,
              'WARMUP_STEPS': args.warmup_steps,
              'ADJ_MAT': args.adj_matrix,
              'ADJ_WEIGHT': args.adj_weight,
              'MMD_USE':args.mmd_use,
              'LATENT_SIZE':args.d_latent}

    ### Load data, vocab and token weights
    if args.data_source == 'custom':
        assert args.train_mols_path is not None and args.test_mols_path is not None and args.vocab_path is not None,\
        "ERROR: Must specify files for train/test data and vocabulary"
        print("start load data: " + str(time.time() - start_time))
        train_mols = pd.read_csv(args.train_mols_path).to_numpy() #just take first samples
        test_mols = pd.read_csv(args.test_mols_path).to_numpy()
        print("end load data: " + str(time.time() - start_time))
        if args.property_predictor:
            assert args.train_props_path is not None and args.test_props_path is not None, \
            "ERROR: Must specify files with train/test properties if training a property predictor"
            train_props = pd.read_csv(args.train_props_path).to_numpy()
            test_props = pd.read_csv(args.test_props_path).to_numpy()
        else:
            train_props = None
            test_props = None
        with open(args.vocab_path, 'rb') as f:
            char_dict = pickle.load(f)
        if args.char_weights_path is not None:
            char_weights = np.load(args.char_weights_path)
            params['CHAR_WEIGHTS'] = char_weights
    else:
        train_mols = pd.read_csv('data/{}_train.txt'.format(args.data_source)).to_numpy()
        test_mols = pd.read_csv('data/{}_test.txt'.format(args.data_source)).to_numpy()
        if args.property_predictor:
            assert args.train_props_path is not None and args.test_props_path is not None, \
            "ERROR: Must specify files with train/test properties if training a property predictor"
            train_props = pd.read_csv(args.train_props_path).to_numpy()
            test_props = pd.read_csv(args.test_props_path).to_numpy()
        else:
            train_props = None
            test_props = None
        with open('data/char_dict_{}.pkl'.format(args.data_source), 'rb') as f:
            char_dict = pickle.load(f)
        char_weights = np.load('data/char_weights_{}.npy'.format(args.data_source))
        params['CHAR_WEIGHTS'] = char_weights

    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k

    params['CHAR_DICT'] = char_dict
    params['ORG_DICT'] = org_dict

    ### Train model
    vae = model_init(args, params)
    if args.checkpoint is not None:
        vae.load(args.checkpoint)
    vae.train(train_mols, test_mols, train_props, test_props,
              epochs=args.epochs, save_freq=args.save_freq) #end_beta_scale=args.end_beta_scale


if __name__ == '__main__':
    parser = train_parser()
    args = parser.parse_args()
    d_args = vars(args)
    train(args)
