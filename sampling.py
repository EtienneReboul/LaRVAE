import torch
import os
import argparse
import numpy as np
import pandas as pd
import selfies as sf
from transvae.tvae_util import calc_entropy
import seaborn as sns
import matplotlib.pyplot as plt
from transvae.rnn_models import RNNAttn
from rdkit import Chem
import regex as re


sf_tok_list =['[C]', '[S]', '[=Branch1]', '[=O]', '[=C]', '[NH1]', '[Branch2]', '[=N]', 
              '[O]', '[Ring1]', '[#Branch2]', '[Branch1]', '[Cl]', '[#Branch1]', '[N]',
              '[Ring2]', '[=Branch2]', '[#C]', '[F]', '[#N]', '[P]', '[Br]', '[=S]']
num_dict={
        "[0]" : 0,
        "[1]" : 1,
        "[2]" : 2,
        "[3]" : 3,
        "[4]" : 4,
        "[5]" : 5,
        "[6]" : 6,
        "[7]" : 7,
        "[8]" : 8,
        "[9]" : 9,
        "[10]" : 10,
        "[11]" : 11,
        "[12]" : 12,
        "[13]" : 13,
        "[14]" : 14,
        "[15]" : 15
    }

overloading_dict = {
    0 : "[C]",
    1 : "[Ring1]",
    2 : "[Ring2]",
    3 : "[Branch1]",
    4 : "[=Branch1]",
    5 : "[#Branch1]",
    6 : "[Branch2]",
    7 : "[=Branch2]",
    8 : "[#Branch2]",
    9 : "[O]",
    10 : "[N]",
    11 : "[=N]",
    12 : "[=C]",
    13 : "[#C]",
    14 : "[S]",
    15 : "[P]"
}


def metric_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--numsamples', required=True, type=int)
    parser.add_argument('--mol_encoding', required=True, type=str) #should be "selfies" or "smiles"
    parser.add_argument('--hex', default=False, action='store_true')
    parser.add_argument('--overload', default=False, action='store_true')
    parser.add_argument('--savename', required=True, type=str)

    return parser

def tokenizer(mol,mol_encoding='selfies'):
    if mol_encoding=='smiles':
        "Tokenizes SMILES string"
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regezz = re.compile(pattern)
        tokens = [token for token in regezz.findall(mol)]
    elif mol_encoding=='selfies':
        tokens=[block+"]" for block in mol.split("]")][:-1]
    else:
        raise NameError('expected mol encoding to be smiles or selfies')
    assert mol == ''.join(tokens), ("{} could not be joined".format(mol))
    return tokens


def unHex(hexed):
    selfie_tokens = []
    tokens = tokenizer(hexed)
    for token in tokens:
        num_toks = []
        if token not in sf_tok_list:
            hex_num = hex(int(token[1:-1]))[2:] #delete "0x" from beginning
            for digit in hex_num:
                num_toks.append(overloading_dict[int(digit, 16)])
            selfie_tokens += num_toks
        else: selfie_tokens.append(token)
    
    return ''.join(selfie_tokens)

def unOverload(overloaded):
    selfie_tokens = []
    tokens = tokenizer(overloaded)
    for token in tokens:
        if token in num_dict.keys():
            token = overloading_dict[num_dict[token]]
        selfie_tokens.append(token)
    return ''.join(selfie_tokens)


def reconstruction_rates(reconstructed_mols, true_mols):
    perfect_recon = 0 #percent of molecules which were perfectly reconstructed
    recon_rate = 0 #average percent of each molcule which is correctly reconstructed
    perf_prefix_lens = [] #length of prefixes which are correctly reconstructed
    reconstructed_mols = list(reconstructed_mols)
    for i in range(len(reconstructed_mols)):
        mol = true_mols[i][0]
        recon_mol = reconstructed_mols[i]

        if recon_mol == mol: 
            perfect_recon +=1
            recon_rate += 1
        else:
            recon_mol = tokenizer(recon_mol, mol_encoding=mol_encoding)
            mol = tokenizer(mol, mol_encoding=mol_encoding)
            min_len = min(len(recon_mol), len(mol))
            same_token = [1 if recon_mol[i]==mol[i] else 0 for i in range(min_len)]
            if 0 in same_token: idx = same_token.index(0) #matches for entire shorter selfie
            else: idx = min_len
            recon_rate += idx/len(mol)
            perf_prefix_lens.append(idx)
    
    recon_rate = recon_rate / len(reconstructed_mols)
    perfect_recon_rate = perfect_recon / len(reconstructed_mols)
    avg_prefix_len = np.mean(np.array(perf_prefix_lens))
    

    return perfect_recon_rate, recon_rate, avg_prefix_len

def assess_stability_selfie(selfie):
    if selfie.find("_")!=-1:
        return "", "", False, False
    if hexed: selfie = unHex(selfie)
    elif overloaded: selfie = unOverload(selfie)
    stable_smile=sf.decoder(selfie)
    if len(stable_smile) != 0:
        stable_selfie=sf.encoder(stable_smile)
    else:
        stable_selfie = ""
    return stable_smile,stable_selfie,selfie==stable_selfie,len(selfie)==len(stable_selfie)

def assess_SELFIES(selfies):
    ### initialize variable 
    validity= True
    stability=False
    token_loss=float(1)
    changed_tokens=False

    
    ### test if undertermined token present
    if selfies.find("_")!=-1:
        validity=False
        return validity,stability,token_loss,changed_tokens
    else:
        ### test if selfies is stable 
        _,stable_selfies,stability,_=assess_stability_selfie(selfies)
        if stability:
            return validity,stability,token_loss,changed_tokens
        else:
            ### compute the percentage of token missing
            blocks_original=tokenizer(selfies, mol_encoding=mol_encoding)
            blocks_stable=tokenizer(stable_selfies, mol_encoding=mol_encoding)
            token_loss= (len(blocks_original)-len(blocks_stable))/len(blocks_original)*100
            if token_loss==0.0 :
                ### find what happened to the mismatched token 
                changed_tokens=[f"token n°{i}/{len(blocks_original)}:{block} => {blocks_stable[i]}" for i,block in enumerate(blocks_original) if blocks_stable[i]!= block]
            else:
                pass

            return validity,stability,token_loss,changed_tokens

def assess_stability_and_validity_smile(smile):
    valid = True
    stable = True
    m = Chem.MolFromSmiles(smile,sanitize=False)
    if m is None:
        valid = False
        stable = False
    else:
        Chem.SanitizeMol(m)
        #try:
            #Chem.SanitizeMol(m)
        #except:
            #stable = False
    return valid, stable


def validity_rate(sampled_mols):
    avg_validity = 0
    avg_stability = 0
    avg_token_loss = 0
    avg_len = 0
    #token_losses = []
    changed_tokens_list = []
    changed_tokens = None
    token_loss = 0
    for selfie in sampled_mols:
        if mol_encoding == "selfies":
            validity, stability, token_loss, changed_tokens = assess_SELFIES(selfie)
        elif mol_encoding == "smiles":
            validity, stability = assess_stability_and_validity_smile(selfie)
        else:
            print("bad mol encoding")
            return
        changed_tokens_list.append(changed_tokens)
        avg_validity += int(validity)
        avg_stability += int(stability)
        avg_token_loss += token_loss
        avg_len += len(tokenizer(selfie, mol_encoding=mol_encoding))
        #token_losses.append(token_loss)
        changed_tokens_list.append(changed_tokens)
        
    avg_validity = avg_validity/len(sampled_mols)
    avg_stability = avg_stability/len(sampled_mols)
    avg_token_loss = avg_token_loss/len(sampled_mols)
    avg_len = avg_len/len(sampled_mols)

    return avg_validity, avg_stability, avg_token_loss, avg_len, changed_tokens_list



#get input
parser = metric_parser()
args = parser.parse_args()
model_path = args.model
numsamples = int(args.numsamples)
mol_encoding = args.mol_encoding
hexed = args.hex
overloaded = args.overload
savename = args.savename

#split model name so can adjust epochs
vae = RNNAttn(params={}, load_fn=model_path)

#Samples Molecule Metrics
mean_validities = []
mean_stabilities = []
mean_token_losses = []
mean_lens = []
changed_token_list = []


#Loop through each checkpoint  (5 epochs apart from 5 --> max_epoch)
print(model_path)
cols = ["sampled selfie, smile, stable selfie, augmented, truncated"]
for i in range(30):
    print("Iteration = " + str(i))

    #Sampled Metrics
    print("begin sample")
    sampled_mols = vae.sample(numsamples)
    print("end sample")
    data =[[mol] + list(assess_stability_selfie(mol)) for mol in sampled_mols]
    df = pd.DataFrame(data)
    df.to_csv(savename,  mode='a', index=False, header=True)
    del(df)
    del(data)
    del(sampled_mols)
    #validity, stability, token_loss, length, changed_tokens = validity_rate(sampled_mols)
    #print(stability)

