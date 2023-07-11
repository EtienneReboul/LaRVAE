import numpy as np
import argparse
import pandas as pd
import regex as re
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger   


def quality_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True, type=str)
    parser.add_argument('--trainset_dict', required=True, type=str)
    parser.add_argument('--sample_counter', required=True, type=str)
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


def get_ring_info(smiles):
    """
    use to compute  the number of ring and the size of each ring
    input : smiles (string)
    output: nb_ring (integer), ring_size_list (list of integers)
    """
    ### instantiate local variable 
    nb_ring=0
    ring_size_list=[]

    ### compute 2D structure from SMILES
    mol= Chem.MolFromSmiles(smiles)

    ### search for the smallest set of symmetrical rings
    ssr=Chem.GetSymmSSSR(mol)
    if ssr:
        nb_ring=len(ssr)
        for ring in ssr:
             ring_size_list.append(len(ring))
    return nb_ring,ring_size_list



############################## MAIN SCRIPT ######################################

#get input
parser = quality_parser()
args = parser.parse_args()

samples = pd.read_csv(args.samples)

#samples = samples[samples["augmented"] == False] #get only unstable selfies
#samples = samples[samples["truncated"] == False] #get only trunctated selfies

if "smile" in samples.columns:
    samples = samples["smile"]
else:
    samples = samples["smiles"]

with open(args.trainset_dict, "rb") as f:
    moses_dict = pkl.load(f)
with open(args.sample_counter, "rb") as f:
    sampled_dict = pkl.load(f)
invalid = 0
unique_thresh = 2
novel_thresh = 1

RDLogger.DisableLog('rdApp.*')   
qeds = []
sas = []
lens = []
weights = []
num_rings = []
ring_sizes = []
logps = []
novelty = 0
unique = 0
for smile in samples:
    try:
        mol = Chem.MolFromSmiles(smile,sanitize=True)
        smile = Chem.CanonSmiles(smile)
        QED_score=QED.qed(mol)
        length = len(tokenizer(smile, mol_encoding="smiles"))
        weight = Descriptors.MolWt(mol)
        sa = sascorer.calculateScore(mol)
        nb_ring,ring_size_list = get_ring_info(smile)
        logp = Descriptors.MolLogP(mol)
        qeds.append(QED_score)
        sas.append(sa)
        lens.append(length)
        weights.append(weight)
        num_rings.append(nb_ring)
        ring_sizes.append(ring_size_list)
        logps.append(logp)
        if sampled_dict.get(smile, 0) < unique_thresh:
            unique +=1
        if moses_dict.get(smile, 0) < novel_thresh:
            novelty +=1

    except Exception as e:
        invalid +=1


df = pd.DataFrame(np.array([qeds, sas, lens, weights, logps, num_rings]).T, columns=["QED", "SA", "len", "weight", "logp", "num rings"])
df["ring sizes"] = np.nan
df['ring sizes'] = df['ring sizes'].astype('object')
df['ring sizes'] = ring_sizes
df.to_csv(args.savename, index=False)

print(args.sample_counter)
print("QED mean: " + str(np.mean(qeds)))
print("QED std: " + str(np.std(qeds)))
print("SAS mean: " + str(np.mean(sas)))
print("SAS std: " + str(np.std(sas)))
print("len mean: " + str(np.mean(lens)))
print("len std: " + str(np.std(lens)))     
print("weight mean: " + str(np.mean(weights)))
print("weight std: " + str(np.std(weights)))
print("logp mean: " + str(np.mean(logps)))
print("logp std: " + str(np.std(logps)))
print("num rings mean: " + str(np.mean(num_rings)))
print("num rings std: " + str(np.std(num_rings)))

########## flattening list of ring size lists ###########
flat_ring_sizes = []
for sizes in ring_sizes:
    for size in sizes:
        flat_ring_sizes.append(size)

print("ring size mean: " + str(np.mean(flat_ring_sizes)))
print("ring size std: " + str(np.std(flat_ring_sizes)))
#########################################################

print("Invalid rate: " + str(invalid/len(samples)))
num_valid = len(samples) - invalid
print("Novelty : " + str(novelty/num_valid))
print("Uniqueness: " + str(unique/num_valid))
    