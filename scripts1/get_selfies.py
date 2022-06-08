# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:33:44 2020

@author: jacqu

Compute selfies for all smiles in csv 
"""
from ntpath import join
import pandas as pd
import argparse
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import sys
import selfies as sf
import json
import time
from selfies import encoder

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))


def _InitialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


def NeutraliseCharges(smiles, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions = _InitialiseNeutralisationReactions()
        reactions = _reactions
    mol = Chem.MolFromSmiles(smiles)
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (Chem.MolToSmiles(mol, True), True)
    else:
        return (smiles, False)
_reactions=None

def clean_smiles(s):
    """ Function to clean smiles; change as needed """
    s2 = NeutraliseCharges(s)
    m = AllChem.MolFromSmiles(s2[0])
    Chem.Kekulize(m)
    s = Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=True)
    return s


def process_one(s):
    clean_smile = clean_smiles(s)
    individual_selfie = encoder(clean_smile)
    return  individual_selfie

def get_selfies(smiles,processes=10):
    time1 = time.perf_counter()
    pool = Pool(processes)
    selfies_list = pool.map(process_one, smiles)
    duration=time.perf_counter()-time1
    pool.close()
    pool.join()
    print(f'It took {duration:.2f}s to process {len(selfies_list)} smiles to selfies')
    return selfies_list


def main(processes=10):

    ###loading smiles from csv  
    train = pd.read_csv("data/moses_smiles_train.csv")
    test=pd.read_csv("data/moses_smiles_test.csv") 
    
    
    ### clean the smiles and convert them to selfies 
    train_selfies=get_selfies(smiles=train["smiles"],processes=processes)
    test_selfies=get_selfies(smiles=test["smiles"],processes=processes)


    ### write a txt file with csv format (weird but expected from TransVAE script) 
    print(f'saving selfies as data/moses_test.txt and data/moses_train.txt')
    train['selfies'] = train_selfies
    test['selfies'] = test_selfies
    train['selfies'].to_csv("data/moses_train.txt",index=False)
    test['selfies'].to_csv("data/moses_test.txt",index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p','--processes', help="number of processes for multiprocessing", type=int,default=10)

    # ======================
    args, _ = parser.parse_known_args()

    # A sanity check for the 'clean_smiles' function in use : 
    print('Showing 3 sample smiles to check stereo and charges handling :\n')
    smiles = ['CC(=O)C1=CC=CC=C1CNCCS1C=NC=N1', 'C=CCN1C(=O)/C(=C/c2ccccc2F)S/C1=N\S(=O)(=O)c1cccs1',
              'N#Cc1ccnc(N2CCC([NH2+]C[C@@H]3CCCO3)CC2)c1']
    for s in smiles:
        print(f'base smile : {s}')
        s = clean_smiles(s)
        print(f'cleaned smile: {s}\n')

    print(f'>>> Computing selfies for all smiles in test and train set. May take some time.')
    main(processes=args.processes)
