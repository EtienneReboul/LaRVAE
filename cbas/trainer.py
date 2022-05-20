import json
import os
import sys
from os import listdir
from os.path import isfile, join
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import math

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

from scripts.parsers import cbas_trainer_parser,model_init


def gather_scores(iteration, name):

    ### find the csv file containing the docking results correspponding to each process 
    dirname = os.path.join(script_dir, f'../docking/docking_results/{name}/')
    csv_files_list=[join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f)) and f.endswith(".csv")]
    
    ### concatenated csv file ,remove unsuccessful docking, and sort by values 
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files_list]
    merged = pd.concat(dfs)
    merged = merged[merged['scores']!=0]
    merged.sort_values(by=['scores'])
    

    
    ### setup permanent record dir for docking score if needs be and save record csv file  
    output_dir=os.path.join(script_dir, f'../docking/docking_results/{name}/permanent_results/')
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{name}_iter_{iteration}_docking_results.csv')
    merged.to_csv(filename)


    ### clean up intermediary docking results csv files 
    for csv_file in csv_files_list:
        if os.path.isfile(csv_file):
            os.remove(csv_file)
        else:
            print(f'was unable to remove {csv_file}, no longer exists or is not there ')

    return merged


def data_prepare(data=pd.DataFrame(),quantile=0.5,name="Cbas_VAE",iteration=1):

    ###  retrieves x% best selfies based on quantile, default  is take the best half of selfies
    idx_stop=int(math.floor(data.shape[0]*quantile))
    # print(idx_stop)
    good_selfies=data[:idx_stop]
    
    ### generate test and train set 
    msk = np.random.rand(len(good_selfies)) < 0.8
    train_set=good_selfies[msk]
    test_set=good_selfies[msk]
    

    ### keeps record of the training set and test set 
    output_dir=os.path.join(script_dir,f'data/{name}/iteration_{iteration}/')
    os.makedirs(output_dir,exist_ok=True)
    train_set_path=os.path.join(output_dir,'train_set.csv')
    test_set_path=os.path.join(output_dir,'test_set.csv')
    train_set['selfies'].to_csv(train_set_path,index=False)
    test_set['selfies'].to_csv(test_set_path,index=False)

    ### create numpy array needed for model training and testing   
    # train_mols = (train_set['selfies']).to_numpy()
    # test_mols = test_set['selfies'].to_numpy()

    # return train_mols,test_mols
    return train_set_path,test_set_path


def retrieve_beta_epoch_init(args):

    ### retrieve last beta from previous model 
    ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    start_epoch = ckpt['epoch']
    total_epochs = start_epoch + args.epochs
    beta_init = (args.beta - args.beta_init) / total_epochs * start_epoch

    return beta_init,start_epoch


def model_init_argument_prepare(name,iteration):

    ### create an args parser with default settings for model 
    parser = argparse.ArgumentParser()

    ### Architecture Parameters
    parser.add_argument('--model', type=str, default="rnnattn")
    parser.add_argument('--d_model', type=int, default=126)
    parser.add_argument('--data_source', type=str, default="custom")
    parser.add_argument('--d_feedforward', default=128, type=int)
    parser.add_argument('--d_latent', default=128, type=int)
    parser.add_argument('--property_predictor', default=False, action='store_true')
    parser.add_argument('--d_property_predictor', default=256, type=int)
    parser.add_argument('--depth_property_predictor', default=2, type=int)

    ### Hyperparameters
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--batch_chunks', default=5, type=int)
    parser.add_argument('--beta', default=0.05, type=float)
    parser.add_argument('--beta_init', default=1e-8, type=float) # initialized but will be updated 
    parser.add_argument('--anneal_start', default=0, type=int)
    parser.add_argument('--adam_lr', default=3e-4, type=float)
    parser.add_argument('--lr_scale', default=1, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int) # Differs from original default 
    parser.add_argument('--eps_scale', default=1, type=float)
    parser.add_argument('--epochs', default=5, type=int) # Differs from original default 

    ### Save Parameters
    parser.add_argument('--save_name', default="CbAS_VAE", type=str)# initialized but will be updated
    parser.add_argument('--save_freq', default=5, type=int)

    ### Load Parameters
    parser.add_argument('--checkpoint', default="path_somewhere", type=str)

    
    ### pseudo-parsing and args update 
    args, _ = parser.parse_known_args()

    if iteration==0:
        prior_model_path=os.path.join(script_dir,f'../checkpoints/000_{name}.ckpt')
    else:
        prior_model_dir=os.path.join(script_dir,f'data/{name}/iteration_{iteration-1}/')
        prior_model_dict_path=os.path.join(prior_model_dir,'dict_path_to_model.json')
        with open(prior_model_dict_path) as json_file:
            model_path_dict = json.load(json_file)
        prior_model_path=model_path_dict['PATH_MODEL']



    args.checkpoint=prior_model_path
    args.save_name=f'{name}_iter_{iteration}'

    args.beta,epoch_previous=retrieve_beta_epoch_init(args=args)

    ### guess the path to the new model that will be generated  and write it as a dict in json file
    finish_epoch= epoch_previous + args.save_freq * (args.epochs//args.save_freq)
    path_model=os.path.join(script_dir,f'../checkpoints/{finish_epoch:03d}_{args.save_name}.ckpt')
    model_path_dict={"PATH_MODEL":path_model}
    output_dir=os.path.join(script_dir,f'data/{name}/iteration_{iteration}/')
    save_path_dict=os.path.join(output_dir,'dict_path_to_model.json')

    with open(save_path_dict,'w') as dict_file:
        json.dump(model_path_dict,dict_file)

    ### setup parameters dict
    params = {'ADAM_LR': args.adam_lr,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'LR_SCALE': args.lr_scale,
              'WARMUP_STEPS': args.warmup_steps}
    
    ### adding  weights to parameters
    char_weights_path=os.path.join(script_dir,f'../data/{name}_char_weights.npy')
    char_weights = np.load(char_weights_path)
    params['CHAR_WEIGHTS'] = char_weights
    
    ### adding vocabularies to parameters
    vocab_path=os.path.join(script_dir,f'../data/{name}_char_dict.pkl')
    with open(vocab_path, 'rb') as f:
            char_dict = pickle.load(f)
    
    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k
    
    params['CHAR_DICT'] = char_dict
    params['ORG_DICT'] = org_dict


    return args,params


def main(iteration, quantile, name):

    ### Aggregate docking results 
    data = gather_scores(iteration, name)

    ### prepare the data  and save train and test set as csv 
    
    train_set_path,test_set_path=data_prepare(data=data,quantile=quantile,name=name,iteration=iteration)
    train_mols=pd.read_csv(train_set_path).to_numpy()
    test_mols=pd.read_csv(test_set_path).to_numpy()
    train_props=None
    test_props=None

    ### set up argument and parameters for model training 
    args,params=model_init_argument_prepare(name=name,iteration=iteration)

    ### ReTrain model
    vae = model_init(args, params)
    vae.load(args.checkpoint)
    vae.train(train_mols, test_mols, train_props, test_props,epochs=args.epochs, save_freq=args.save_freq)


if __name__ == '__main__':

    parser2 = cbas_trainer_parser()
    args_main, _ = parser2.parse_known_args()

    main(name=args_main.name,iteration=args_main.iteration,quantile=args_main.quantile)
