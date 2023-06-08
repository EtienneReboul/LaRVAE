import numpy as np
import os
import argparse
import pandas as pd
import selfies as sf
import pickle
from collections import Counter
from transvae.tvae_util import *

def clean_charge_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mols', required=True, type=str)   
    parser.add_argument('--test_mols', required=True, type=str)
    parser.add_argument('--vocab_name', required=True, type=str)
    parser.add_argument('--save_dir', default='data', type=str)
    return parser

parser = clean_charge_parser()
args = parser.parse_args()


train_data = pd.read_csv(args.train_mols).to_numpy()
test_data = pd.read_csv(args.test_mols).to_numpy()

bad_tokens = {}
token_counts = Counter()

for line in train_data:
    mol = tokenizer(line[0], mol_encoding="selfies")
    token_counts.update(mol)

for tok, count in token_counts.items():
    if count < 2000:
        bad_tokens[tok] = 1

cleaned_train_mols = []
for line in train_data:
    mol = tokenizer(line[0], mol_encoding="selfies")
    bad_selfie = False
    if len(mol)>58: 
        bad_selfie = True
    else:
        for tok in mol:
            if bool(bad_tokens.get(tok, 0)): 
                bad_selfie = True
                break
    if not bad_selfie:
        cleaned_train_mols.append(["".join(mol)])

cleaned_test_mols = []
for line in test_data:
    mol = tokenizer(line[0], mol_encoding="selfies")
    bad_selfie = False
    if len(mol)>58: 
        bad_selfie = True
    else:
        for tok in mol:
            if bool(bad_tokens.get(tok, 0)): 
                bad_selfie = True
                break
    if not bad_selfie:
        cleaned_test_mols.append(["".join(mol)])

cleaned_train_data = pd.DataFrame(cleaned_train_mols, columns =['selfies'])
cleaned_test_data = pd.DataFrame(cleaned_test_mols, columns =['selfies'])

cleaned_train_data['selfies'].to_csv("data/clean_zinc_train.txt",index=False)
cleaned_test_data['selfies'].to_csv("data/clean_zinc_test.txt",index=False)

clean_char_dict = {}
char_idx = 0
for tok in token_counts.keys():
    if not bool(bad_tokens.get(tok, 0)):
        clean_char_dict[tok] = char_idx
        char_idx += 1

with open(os.path.join(args.save_dir, args.vocab_name+'.pkl'), 'wb') as f:
        pickle.dump(clean_char_dict, f)
f.close()

#print(token_counts)
#print(bad_tokens)
#print(clean_char_dict)
#print(train_data.shape)
#print(test_data.shape)
#print(cleaned_train_data.shape)
#print(cleaned_test_data.shape)

