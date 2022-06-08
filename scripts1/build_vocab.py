import os
import pickle

import numpy as np
import pandas as pd
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

from transvae.tvae_util import *
from scripts1.parsers1 import vocab_parser

def build_vocab(args):
    ### Build vocab dictionary
    print('building dictionary...')
    char_dict = {'<start>': 0}
    char_idx = 1
    mol_toks = []
    mol_enconding=''
    max_len=int()
    with open(args.mols, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            if line.lower() in ['smile', 'smiles', 'selfie', 'selfies']:
                if line.lower() in ['smile', 'smiles']:
                    mol_enconding='smiles'
                elif line.lower() in ['selfie','selfies']:
                    mol_enconding='selfies'
                pass
        
            else:
                mol = tokenizer(line,mol_enconding)
                for tok in mol:
                    if tok not in char_dict.keys():
                        char_dict[tok] = char_idx
                        char_idx += 1
                    else:
                        pass
                mol.append('<end>')
                if len(mol)>max_len:
                    max_len=len(mol)
                mol_toks.append(mol)
    char_dict['_'] = char_idx
    char_dict['<end>'] = char_idx + 1

    ### Write dictionary to file
    with open(os.path.join(args.save_dir, args.vocab_name+'.pkl'), 'wb') as f:
        pickle.dump(char_dict, f)

    ### Set weights params
    del char_dict['<start>']
    params = {'MAX_LENGTH': max_len,
              'NUM_CHAR': len(char_dict.keys()),
              'CHAR_DICT': char_dict}

    ### Calculate weights
    print('calculating weights...')
    char_weights = get_char_weights(mol_toks, params, freq_penalty=args.freq_penalty)
    char_weights[-2] = args.pad_penalty
    np.save(os.path.join(args.save_dir, args.weights_name+'.npy'), char_weights)


if __name__ == '__main__':
    parser = vocab_parser()
    args = parser.parse_args()
    build_vocab(args)
