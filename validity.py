import pandas as pd
import numpy as np
import argparse
from transvae.rnn_models import RNNAttn
from transvae.tvae_util import *
import seaborn as sns
import matplotlib.pyplot as plt
import random
import selfies as sf

def tokenizer(selfie):
    return [block+"]" for block in selfie.split("]")][:-1]

def assess_stability(selfie):
    stable_smile=sf.decoder(selfie)
    if len(stable_smile) != 0:
        stable_selfie=sf.encoder(stable_smile)
    else:
        stable_selfie = ""
    return stable_selfie,selfie==stable_selfie

def assess_SELFIES(selfies):
    ### initialize variable 
    validity= True
    stability=False
    token_loss=float(-1)
    changed_tokens=False

    
    ### test if undertermined token present
    if selfies.find("_")!=-1:
        validity=False
        return validity,stability,token_loss,changed_tokens
    else:
        ### test if selfies is stable 
        stable_selfies,stability=assess_stability(selfies)
        if stability:
            token_loss=float(0)
            return validity,stability,token_loss,changed_tokens
        else:
            ### compute the percentage of token missing
            blocks_original=tokenizer(selfies)
            blocks_stable=tokenizer(stable_selfies)
            token_loss= (len(blocks_original)-len(blocks_stable))/len(blocks_original)*100
            if token_loss==0.0 :
                ### find what happened to the mismatched token 
                changed_tokens=[f"token n°{i}/{len(blocks_original)}:{block} => {blocks_stable[i]}" for i,block in enumerate(blocks_original) if blocks_stable[i]!= block]
            else:
                pass

            return validity,stability,token_loss,changed_tokens

def recon_rate(reconstructed_mols, true_mols):
    perfect_recon = 0
    perf_prefix_lens = []
    for i in range(len(reconstructed_mols)):
        mol = true_mols[i][0]
        recon_mol = reconstructed_mols[i]

        if recon_mol == mol: perfect_recon +=1
        else:
            recon_mol = tokenizer(recon_mol)
            mol = tokenizer(mol)
            min_len = min(len(recon_mol), len(mol))
            same_token = [1 if recon_mol[i]==mol[i] else 0 for i in range(min_len)]
            if 0 in same_token: idx = same_token.index(0) #matches for entire shorter selfie
            else: idx = min_len
            perf_prefix_lens.append(idx)
    
    perfect_recon_rate = perfect_recon / len(reconstructed_mols)
    perf_prefix_len = np.mean(np.array(perf_prefix_lens))
    

    return perfect_recon_rate, perf_prefix_len

def validity_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)   
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--num_samples', required=True, type=int)
    parser.add_argument('--log_name', required=True, type=str)       
    return parser

def validity_rate(sampled_mols):
    avg_validity = 0
    avg_stability = 0
    avg_token_loss = 0
    avg_len = 0
    token_losses = []
    for selfie in sampled_mols:
        validity, stability, token_loss, changed_tokens = assess_SELFIES(selfie)
        avg_validity += int(validity)
        avg_stability += int(stability)
        avg_token_loss += token_loss
        avg_len += len(tokenizer(selfie))
        token_losses.append(token_loss)
        
    avg_validity = avg_validity/len(sampled_mols)
    avg_stability = avg_stability/len(sampled_mols)
    avg_token_loss = avg_token_loss/len(sampled_mols)
    avg_len = avg_len/len(sampled_mols)

    return avg_validity, avg_stability, avg_token_loss, avg_len, token_losses


def latentCovariance(mems, log_name):
    cov_name = log_name + "_cov"
    entropy_name = log_name + "_ent"
    cov_mat = np.cov(mems.T)
    entropies = calc_entropy(mems)
    entropies = np.sort(entropies)

    #save plot of covariance matrix
    sns.set(rc = {'figure.figsize':(23,20)})
    cov_plot = sns.heatmap(cov_mat, annot=False)
    fig = cov_plot.get_figure()
    fig.savefig("analysis/" + cov_name) 

    #save plot of entropy
    fig = plt.figure(figsize=(12,6))
    plt.bar(range(len(entropies)), entropies)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Entropy (bits)')
    fig.savefig("analysis/" + entropy_name)

    #save csv file
    pd.DataFrame(cov_mat).to_csv("analysis/" + cov_name  + ".csv")
    pd.DataFrame(entropies).to_csv("analysis/" + entropy_name  + ".csv")
    return cov_mat
    


parser = validity_parser()
args = parser.parse_args()
data_file = args.data
model_name = args.model
num_samples = args.num_samples
log_name = args.log_name

#torch.cuda.empty_cache()

vae = RNNAttn(params={}, load_fn=model_name)

test_data = pd.read_csv(data_file).to_numpy()

log_file = open("Validity Files/" + log_name + ".txt", "w")
log_file.write(str(vae.params) + "\n")

with torch.no_grad():
    perfect_recon_rates = []
    perf_prefix_lens = []
    for i in range(4):
        subset_data = random.sample(list(test_data), 1000)
        subset_data = np.array(subset_data)
        reconstructed_mols = vae.reconstruct(subset_data, log=False, return_mems=False)
        #reconstructed_mols = vae.reconstruct(test_data[0:500], log=False, return_mems=False)
        perfect_recon_rate, perf_prefix_len = recon_rate(reconstructed_mols, subset_data)
        #perfect_recon_rate, perf_prefix_len = recon_rate(reconstructed_mols, test_data[0:500])
        perfect_recon_rates.append(perfect_recon_rate)
        perf_prefix_lens.append(perf_prefix_len)
        del(subset_data)
        del(reconstructed_mols)
    perfect_recon_rate = sum(perfect_recon_rates)/4
    perf_prefix_len = sum(perf_prefix_lens)/4
    del(perfect_recon_rates)
    del(perf_prefix_lens)

    log_file.write("Perfect Reconstruction Rate (w/o teacher forcing): " + str(perfect_recon_rate))
    log_file.write("\nAverage Perfect Prefix Length: " + str(perf_prefix_len))

    sum_validity = 0
    sum_stability = 0
    sum_token_loss = 0
    sum_avg_len = 0
    token_density = []
    n = 4
    for i in range(n):
        sampled_mols = vae.sample(num_samples)
        validity, stability, token_loss, avg_len, token_losses = validity_rate(sampled_mols)
        sum_validity += validity
        sum_stability += stability
        sum_token_loss += token_loss
        sum_avg_len += avg_len
        token_density+=token_losses
        del(sampled_mols)
        del(token_losses)
    total_validity = sum_validity/n
    total_stability = sum_stability/n
    total_token_loss = sum_token_loss/n
    total_avg_len = sum_avg_len/n
    log_file.write("\nSampled " + str(n*num_samples) + " latent vectors...")
    log_file.write("\nValidity Rate: " + str(total_validity))
    log_file.write("\nStability Rate: " + str(total_stability))
    log_file.write("\nAverage Token Loss: " + str(total_token_loss))
    log_file.write("\nAverage length of selfie: " + str(total_avg_len))

    density_name = log_name + "_tokenDensity"
    density_plot = sns.displot(token_density,kind="kde").set(title='Token Loss Density')
    density_plot.set(xlabel='Token Loss', ylabel='Probability')
    density_plot.savefig("analysis/" + density_name)

    _, mems, _ = vae.calc_mems(test_data, log=False, save=False)

    cov_mat = latentCovariance(mems, log_name)
    del(mems)
    log_file.write("\nCovariance Matrix of Latent Dimensions saved ")
    log_file.write("\nEntropy of Latent Dimensions saved")

    log_file.close()