import os
import time
import numpy as np
import pandas as pd
import sys
import json #added by zoe
script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))


from transvae.rnn_models import RNN, RNNAttn
from transvae.tvae_util import calc_entropy
from scripts1.parsers1 import sample_parser

from rdkit import Chem
from rdkit.Chem import RDConfig
import selfies as sf
import pickle 
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
from rdkit.Chem import QED,AllChem
try:
    from rdkit.Chem import MolToInchi
except:
    from rdkit.Chem.rdinchi import MolToInchi

from multiprocessing import Pool
from functools import partial

### Assess if selfie should be retained as a good drug candidate 
def selfie_check(selfie,max_ringsize=6,SA_threshold=4.5,QED_threshold=0.5,alphabet={}):
    good_selfie=None
    # eliminate Selfies with non attributed token
    if selfie.find("_")!=-1:
        return good_selfie
    else:
        smi=sf.decoder(selfie)
        mol=Chem.MolFromSmiles(smi)
        try:
            s = sascorer.calculateScore(mol)
            QED_score=QED.qed(mol)
        except:
            return good_selfie 
        # reject selfies that do not meet the Synthetic Accessibility criteria (SA) and the Drug-likeness criterion (QED)
        if s >SA_threshold or QED_score< QED_threshold:
            return good_selfie
        else:
            good_selfie= selfie
            # Check for ring 
            ssr=Chem.GetSymmSSSR(mol)
            if ssr:
                for ring in ssr:
                    if len(list(ring))>max_ringsize:
                        good_selfie=None
                        break
                    else:
                        pass
            
            if good_selfie:
                ### sanitize the smiles and convert it back to a stable selfie 
                Chem.Kekulize(mol)
                smi = Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=True)
                good_selfie=sf.encoder(smi)
                ### assert that no new token appeared, rare but truncation can create new chemical function => new token 
                selfies_tokens_list=list(sf.split_selfies(good_selfie))
                if not all(token in alphabet for token in selfies_tokens_list):
                    good_selfie=None
                else:
                    pass
            return good_selfie

def get_Inchy_from_selfie(selfie):
    smi=sf.decoder(selfie)
    mol=Chem.MolFromSmiles(smi)
    Inchi=MolToInchi(mol)
    return Inchi

def add_unique_mol(samples_list,samples,cores=10):
    # translating selfies to Inchis
    pool=Pool(cores)
    Inchis=pool.map(get_Inchy_from_selfie,samples_list)
    pool.close()
    pool.join()
    cnt=0
    for i,Inchi in enumerate(Inchis):
        if Inchi in samples:
            pass
        else:
            samples[Inchi]=samples_list[i]
            cnt+=1
    
    return samples,cnt






def sample(args):
    
    ### retrieve model path from dict of previous iteration
    if args.iteration>0:
        prior_model_dict_path=os.path.join(script_dir,f'data/{args.name}/iteration_{args.iteration-1}/dict_path_to_model.json')
        with open(prior_model_dict_path) as json_file:
            model_path_dict = json.load(json_file)
        ckpt_fn=model_path_dict['PATH_MODEL']
    else:
        ckpt_fn = args.model_ckpt
    
    ### Load model
    if args.model == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    elif args.model == 'rnn':
        vae = RNN(load_fn=ckpt_fn)

    ### Parse conditional string
    if args.condition == '':
        condition = []
    else:
        condition = args.condition.split(',')

    ### Calculate entropy depending on sampling mode
    if args.sample_mode == 'rand':
        sample_mode = 'rand'
        sample_dims = None
    else:
        entropy_data = pd.read_csv(args.mols).to_numpy()
        _, mus, _ = vae.calc_mems(entropy_data, log=False, save=False)
        vae_entropy = calc_entropy(mus)
        entropy_idxs = np.where(np.array(vae_entropy) > args.entropy_cutoff)[0]
        sample_dims = entropy_idxs
        if args.sample_mode == 'high_entropy':
            sample_mode = 'top_dims'
        elif args.sample_mode == 'k_high_entropy':
            sample_mode = 'k_dims'

    ### Generate samples
    samples = {} # Dict that contains inCHi as keys and SELFIES as values 
    n_gen = args.n_samples
    total_time = time.time()
    ### retrieve selfies alphabet 
    vocab_path=os.path.join(script_dir,f'../data/{args.name}_char_dict.pkl')
    with open(vocab_path, 'rb') as f:
            char_dict = pickle.load(f)
    while n_gen > 0:
        # sample generation
        current_samples = vae.sample(args.n_samples_per_batch, sample_mode=sample_mode,
                                     sample_dims=sample_dims, k=args.k, condition=condition)
        # quality check   
        pool=Pool(args.cores)
        selfies=pool.map(partial(selfie_check,max_ringsize=6,SA_threshold=4.5,QED_threshold=0.5,alphabet=char_dict),current_samples)
        pool.close()
        pool.join()
        good_samples=[selfie for selfie in selfies if selfie]
        perct_filtered=len(good_samples)/len(current_samples)*100
        
        # adding unique molecule to pool
        samples,nb_added_mol=add_unique_mol(good_samples,samples,cores=args.cores)
        perct_added= nb_added_mol/len(good_samples)*100

        
        # Update number of molecule to sample
        if nb_added_mol >0:
            n_gen -= nb_added_mol
        else:
            print("unable to add new molecules to our current pool of samples,stoping now")
            break
        print(f"Batch of size:{args.n_samples_per_batch},{perct_filtered:.2f}% of the batch has passed the filters,{perct_added:.2f}% of the filtered molecules were added to the pool")
        print(f"Our pool currently hold {len(samples)} samples ,{n_gen} remained to be sampled")
    
    print(f'Total time: {time.time()-total_time:.2f}s')
    samples = pd.DataFrame(samples.values(), columns=['mol'])
    dirname=os.path.join(script_dir,'../docking/data_docking/VAE_generated_samples/')
    os.makedirs(dirname, exist_ok=True)
    save_path = os.path.join(dirname, f'{args.name}_sampled_selfies.csv')
    samples.to_csv(save_path,index=False)


if __name__ == '__main__':
    parser = sample_parser()
    args = parser.parse_args()
    sample(args)
