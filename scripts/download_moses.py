# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:13:51 2020

@author: jacqu

Download moses datasets and saves them 
"""
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '../'))

print(f'working dir is {os.getcwd()}')
import moses
import pandas as pd



def download_moses():
    ### load moses test and train set 
    print('>>> Loading data from moses')
    train = moses.get_dataset('train')
    test = moses.get_dataset('test')
    
    ### rename colums
    train = pd.DataFrame(train).rename(columns={0: 'smiles'})
    test = pd.DataFrame(test).rename(columns={0: 'smiles'})
        
    ### save as csv file 
    print('>>> Saving data to csv files in data dir')
    train.to_csv('data/moses_smiles_train.csv',index=False)
    test.to_csv('data/moses_smiles_test.csv',index=False)

download_moses()
