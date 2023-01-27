import os
import json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

from transvae.tvae_util import *
from transvae.opt import NoamOpt
from transvae.data import vae_data_gen
from transvae.loss import vae_loss
#import dataloader


####### MODEL SHELL ##########

class VAEShell():
    """
    VAE shell class that includes methods for parameter initiation,
    data loading, training, logging, checkpointing, loading and saving,
    """
    def __init__(self, params, name=None):
        self.params = params
        self.name = name
        if 'BATCH_SIZE' not in self.params.keys():
            print("done")
            self.params['BATCH_SIZE'] = 500
        if 'BATCH_CHUNKS' not in self.params.keys():
            self.params['BATCH_CHUNKS'] = 5
        if 'BETA_INIT' not in self.params.keys():
            self.params['BETA_INIT'] = 1e-8
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.05
        if 'ANNEAL_START' not in self.params.keys():
            self.params['ANNEAL_START'] = 0
        if 'LR' not in self.params.keys():
            self.params['LR_SCALE'] = 1
        if 'WARMUP_STEPS' not in self.params.keys():
            self.params['WARMUP_STEPS'] = 10000
        if 'EPS_SCALE' not in self.params.keys():
            self.params['EPS_SCALE'] = 1
        if 'CHAR_DICT' in self.params.keys():
            self.vocab_size = len(self.params['CHAR_DICT'].keys())
            self.pad_idx = self.params['CHAR_DICT']['_']
            if 'CHAR_WEIGHTS' in self.params.keys():
                self.params['CHAR_WEIGHTS'] = torch.tensor(self.params['CHAR_WEIGHTS'], dtype=torch.float)
            else:
                self.params['CHAR_WEIGHTS'] = torch.ones(self.vocab_size, dtype=torch.float)
        self.loss_func = vae_loss
        self.data_gen = vae_data_gen

        ### Sequence length hard-coded into model
        self.src_len = 58 #Zoe changed from 126 to 58 (length of input with start token will be 59)
        self.tgt_len = 57 #Zoe changed from 125 to 57, don't know why ouput is shorter

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'params': self.params}
        self.loaded_from = None

    def save(self, state, fn, path='checkpoints', use_name=True):
        """
        Saves current model state to .ckpt file

        Arguments:
            state (dict, required): Dictionary containing model state
            fn (str, required): File name to save checkpoint with
            path (str): Folder to store saved checkpoints
        """
        os.makedirs(path, exist_ok=True)
        if use_name:
            if os.path.splitext(fn)[1] == '':
                if self.name is not None:
                    fn += '_' + self.name
                fn += '.ckpt'
            else:
                if self.name is not None:
                    fn, ext = fn.split('.')
                    fn += '_' + self.name
                    fn += '.' + ext
            save_path = os.path.join(path, fn)
        else:
            save_path = fn
        torch.save(state, save_path)

    def load(self, checkpoint_path):
        """
        Loads a saved model state

        Arguments:
            checkpoint_path (str, required): Path to saved .ckpt file
        """
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.loaded_from = checkpoint_path
        for k in self.current_state.keys():
            try:
                self.current_state[k] = loaded_checkpoint[k]
            except KeyError:
                self.current_state[k] = None

        if self.name is None:
            self.name = self.current_state['name']
        else:
            pass
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        for k, v in self.current_state['params'].items():
            if k in self.arch_params or k not in self.params.keys():
                self.params[k] = v
            else:
                pass
        self.vocab_size = len(self.params['CHAR_DICT'].keys())
        self.pad_idx = self.params['CHAR_DICT']['_']
        self.build_model()
        self.model.load_state_dict(self.current_state['model_state_dict'])
        self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])

    def train(self, train_mols, val_mols, train_props=None, val_props=None,
              epochs=100, save=True, save_freq=None, log=True, log_dir='trials'): #, end_beta_scale=20
        
        # with open("logs/betaTestFile.txt", "w") as f:
        #     f.write("end_beta_scale is 20")
        #     f.close()
        """
        Train model and validate

        Arguments:
            train_mols (np.array, required): Numpy array containing training
                                             molecular structures
            val_mols (np.array, required): Same format as train_mols. Used for
                                           model development or validation
            train_props (np.array): Numpy array containing chemical property of
                                   molecular structure
            val_props (np.array): Same format as train_prop. Used for model
                                 development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions of model
            save_freq (int): Frequency with which to save model checkpoints
            log (bool): If true, writes training metrics to log file
            log_dir (str): Directory to store log files
        """
        ### Prepare data iterators
        #train_data = dataloader.VAE_Dataset(train_mols, train_props, self.params['CHAR_DICT'], self.src_len, self.params['ADJ_MAT'], self.params['ADJ_WEIGHT'])
        #val_data = dataloader.VAE_Dataset(val_mols, val_props, self.params['CHAR_DICT'], self.src_len, self.params['ADJ_MAT'], self.params['ADJ_WEIGHT'])

        train_data = self.data_gen(train_mols, train_props, self.params['CHAR_DICT'], self.src_len, self.params['ADJ_MAT'], self.params['ADJ_WEIGHT'])
        val_data = self.data_gen(val_mols, val_props, self.params['CHAR_DICT'], self.src_len, self.params['ADJ_MAT'], self.params['ADJ_WEIGHT'])


        train_iter = torch.utils.data.DataLoader(train_data,
                                                 batch_size=self.params['BATCH_SIZE'],
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False, drop_last=True)
        val_iter = torch.utils.data.DataLoader(val_data,
                                               batch_size=self.params['BATCH_SIZE'],
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS']


        torch.backends.cudnn.benchmark = True

        ### Determine save frequency
        if save_freq is None:
            save_freq = epochs

        ### Setup log file
        if log:
            os.makedirs(log_dir, exist_ok=True)
            if self.name is not None:
                log_fn = '{}/log{}.txt'.format(log_dir, '_'+self.name)
            else:
                log_fn = '{}/log.txt'.format(log_dir)
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
            except FileNotFoundError:
                already_wrote = False
            log_file = open(log_fn, 'a')
            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,recon_loss,pred_loss,kld_loss,beta_kld,mmd_loss,prop_mse_loss,perfect_recon_acc,run_time\n')
            log_file.close()
        # tensorboard_dir=log_fn[:-4]
        # os.makedirs(tensorboard_dir,exist_ok=True)
        # writer = SummaryWriter(tensorboard_dir)

        ### Initialize Annealer
        kl_annealer = KLAnnealer(self.params['BETA_INIT'], self.params['BETA'], 
                                    40, self.params['ANNEAL_START'])#stop increasing beta at 40 epochs

        #print("stop increasing beta at 10 epochs")
        ### Epoch loop
        train_step = 0
        for epoch in range(epochs):
            start_time = time.time()
            ### Train Loop
            self.model.train()
            losses = []
            beta = kl_annealer(epoch)
            for j, data in enumerate(train_iter):
                train_step += 1
                avg_losses = []
                avg_bce_losses = []
                avg_kld_losses = []
                avg_beta_kld_losses = []
                avg_MMD_losses=[]
                avg_prop_mse_losses = []
                avg_perf_recon_accs = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    input_len = self.src_len+1 #input length including padding and start token
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data[:,:input_len] #changed by zoe
                    if self.params['ADJ_MAT']:
                        adjMat_data = batch_data[:, input_len:-1] #added by zoe
                        adjMat_data = torch.reshape(adjMat_data, (self.chunk_size, input_len, input_len))
                    else:
                        adjMat_data = None
                    props_data = batch_data[:,-1]
                    if self.use_gpu:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()
                        if self.params['ADJ_MAT']:
                            adjMat_data = adjMat_data.cuda()

                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()
                    true_prop = Variable(props_data)

                    
                    if self.params['ADJ_MAT']:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, adjMat_data) #Zoe Added AdjMatrix ", adjMat_data"
                        
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt) #Zoe Added AdjMatrix ", adjMat_data"

                    if self.params['MMD_USE']:
                        loss, bce,MMD_loss,prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                true_prop, pred_prop,
                                                                self.params['CHAR_WEIGHTS'],
                                                                beta,
                                                                MMD_use=True,
                                                                latent_size=self.params['LATENT_SIZE'],
                                                                device=self.device)
                        kld=torch.tensor(0.)
                        beta_kld=torch.tensor(0.)
                    else:
                        loss, bce, kld, beta_kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                true_prop, pred_prop,
                                                                self.params['CHAR_WEIGHTS'],
                                                                beta)
                        MMD_loss=torch.tensor(0.)

                    #acc = self.perfect_recon_acc(src, x_out, self.tgt_len, self.params["CHAR_DICT"], self.params['ORG_DICT'])
                    acc=0

                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_beta_kld_losses.append(beta_kld.item())
                    avg_MMD_losses.append(MMD_loss.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                    avg_perf_recon_accs.append(acc)
                    loss.backward()



                self.optimizer.step()
                self.model.zero_grad()
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                avg_perf_recon_acc = np.mean(avg_perf_recon_accs)
                avg_MMD=np.mean(avg_MMD_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_beta_kld = np.mean(avg_beta_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                avg_perf_recon_acc = np.mean(avg_perf_recon_accs)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                         j, 'train',
                                                                         avg_loss,
                                                                         avg_bce,
                                                                         avg_kld,
                                                                         avg_beta_kld,
                                                                         avg_MMD,
                                                                         avg_prop_mse,
                                                                         avg_perf_recon_acc,
                                                                         run_time))
                    log_file.close()
            train_loss = np.mean(losses)

            print("train time for one epoch: " + str(time.time() - start_time))
            start_time = time.time()
            
            
            ### Val Loop
            self.model.eval()
            losses = []
            for j, data in enumerate(val_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_kld_losses = []
                avg_beta_kld_losses = []
                avg_MMD_losses=[]
                avg_prop_mse_losses = []
                avg_perf_recon_accs = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    input_len = self.src_len+1 #input length including padding and start token
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data[:,:input_len] #changed by zoe
                    if self.params['ADJ_MAT']:
                        adjMat_data = batch_data[:, input_len:-1] #added by zoe
                        adjMat_data = torch.reshape(adjMat_data, (self.chunk_size, input_len, input_len))
                    else:
                        adjMat_data = None
                    if self.use_gpu:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()
                        if self.params['ADJ_MAT']:
                            adjMat_data = adjMat_data.cuda()

                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()
                    true_prop = Variable(props_data)
                    scores = Variable(data[:,-1])

                    if self.params['ADJ_MAT']:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, adjMat_data) #Zoe Added AdjMatrix ", adjMat_data"
                        
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt) #Zoe

                    if self.params['MMD_USE']:
                        loss, bce,MMD_loss,prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                true_prop, pred_prop,
                                                                self.params['CHAR_WEIGHTS'],
                                                                beta,
                                                                MMD_use=True,
                                                                latent_size=self.params['LATENT_SIZE'],
                                                                device=self.device)
                        kld=torch.tensor(0.)
                        beta_kld=torch.tensor(0.)
                    else:
                        loss, bce, kld, beta_kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                true_prop, pred_prop,
                                                                self.params['CHAR_WEIGHTS'],
                                                                beta)
                        MMD_loss=torch.tensor(0.)

                    #acc = self.perfect_recon_acc(src, x_out, self.tgt_len, self.params["CHAR_DICT"], self.params['ORG_DICT'])
                    acc=0

                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_beta_kld_losses.append(beta_kld.item())
                    avg_MMD_losses.append(MMD_loss.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                    avg_perf_recon_accs.append(acc)
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                avg_MMD=np.mean(avg_MMD_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_beta_kld = np.mean(avg_beta_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                avg_perf_recon_acc = np.mean(avg_perf_recon_accs)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'test',
                                                                avg_loss,
                                                                avg_bce,
                                                                avg_kld,
                                                                avg_beta_kld,
                                                                avg_MMD,
                                                                avg_prop_mse,
                                                                avg_perf_recon_acc,
                                                                run_time))
                    log_file.close()

            self.n_epochs += 1
            val_loss = np.mean(losses)
            if self.params['MMD_USE']:
                print('Epoch - {} Train - {} Val - {}'.format(self.n_epochs, train_loss, val_loss))
            else:
                print('Epoch - {} Train - {} Val - {} KLBeta - {}'.format(self.n_epochs, train_loss, val_loss, beta))

            

            ### Update current state and save model
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.current_state['best_loss'] = self.best_loss
                if save:
                    self.save(self.current_state, 'best')

            if (self.n_epochs) % save_freq == 0:
                epoch_str = str(self.n_epochs)
                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str
                if save:
                    self.save(self.current_state, epoch_str)
        
        print("val time for one epoch: " + str(time.time() - start_time))

    ### Sampling and Decoding Functions
    def sample_from_memory(self, size, mode='rand', sample_dims=None, k=5):
        """
        Quickly sample from latent dimension

        Arguments:
            size (int, req): Number of samples to generate in one batch
            mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
        Returns:
            z (torch.tensor): NxD_latent tensor containing sampled memory vectors
        """
        if mode == 'rand':
            z = torch.randn(size, self.params['d_latent'])
        else:
            assert sample_dims is not None, "ERROR: Must provide sample dimensions"
            if mode == 'top_dims':
                z = torch.zeros((size, self.params['d_latent']))
                for d in sample_dims:
                    z[:,d] = torch.randn(size)
            elif mode == 'k_dims':
                z = torch.zeros((size, self.params['d_latent']))
                d_select = np.random.choice(sample_dims, size=k, replace=False)
                for d in d_select:
                    z[:,d] = torch.randn(size)
        return z

    
    def greedy_decode(self, mem, condition=[]):
        """
        Greedy decode from model memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
        Returns:
            decoded (torch.tensor): Tensor of predicted token ids
        """
        torch.cuda.empty_cache()

        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        decoded = torch.ones(mem.shape[0],1).fill_(start_symbol).long()
        for tok in condition:
            condition_symbol = self.params['CHAR_DICT'][tok]
            condition_vec = torch.ones(mem.shape[0],1).fill_(condition_symbol).long()
            decoded = torch.cat([decoded, condition_vec], dim=1)
        tgt = torch.ones(mem.shape[0],max_len+2).fill_(start_symbol).long() #add start token for teacher forcing
        #above +1  hanged to +2 by zoe 
        tgt[:,:len(condition)+1] = decoded

        if self.use_gpu:
            decoded = decoded.cuda()
            tgt = tgt.cuda()

        self.model.eval()
        for i in range(len(condition), max_len+1): #+1 added by zoe
            out, _ = self.model.decode(tgt[:, :-1], mem) #last token in tgt not needed for teacher forcing
            out = self.model.generator(out) #shape is (100, 58, 25) ie no start token
            prob = F.softmax(out[:,i,:], dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word += 1 #convert to alphabet with start token
            tgt[:,i+1] = next_word
        decoded = tgt[:,1:] #delete the start token
        return decoded #shape (100, 58) no start token, but uses 26-alphabet


    def perfect_recon_acc(self, x, mem, tgt_len, char_dict, org_dict):
        #x is (100, 59) includes start token encoded with 26-alphabet
        x = x.long()[:,1:] #get rid of start token
        start_symbol = char_dict['<start>']
        predicted = torch.ones(x.shape[0],tgt_len+2).fill_(start_symbol).long() #shape (100, 58), starts token, missing last token
        for i in range(tgt_len+1): #i ranges from 0 to 56
            out, _ = self.model.decode(predicted[:, :-1], mem) #last token in tgt not needed for teacher forcing
            out = self.model.generator(out)
            prob = F.softmax(out[:,i,:], dim=-1) #x_out has length 58, so last column is never considered
            _, next_word = torch.max(prob, dim=1)
            next_word += 1
            predicted[:,i+1] = next_word
        predicted = predicted[:,1:]
        predicted = decode_mols(predicted, org_dict)
        x = decode_mols(x, org_dict)
        zipped = zip(predicted, x)
        acc = 0
        for recon_mol, mol in zipped:
            if recon_mol == mol: 
                acc += 1
        return acc/len(x)
    
    
    def reconstruct(self, mols, method='greedy', log=True, return_mems=True, return_str=True):
        """
        Method for encoding input smiles into memory and decoding back
        into smiles

        Arguments:
            mols (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            log (bool): If true, tracks reconstruction progress in separate log file
            return_mems (bool): If true, returns memory vectors in addition to decoded SMILES
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded_smiles (list): Decoded smiles data - either decoded SMILES strings or tensor of
                                   token ids
            mems (np.array): Array of model memory vectors
        """
        data = vae_data_gen(mols, None, self.params['CHAR_DICT'], self.src_len, use_adj=self.params["ADJ_MAT"], adj_weight=self.params["ADJ_WEIGHT"])
        #data = dataloader.VAE_Dataset(data, None, self.params['CHAR_DICT'], self.src_len, self.params['ADJ_MAT'], self.params['ADJ_WEIGHT'])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False,num_workers=0,
                                                pin_memory=False, drop_last=True)
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']

        input_len = self.src_len+1 #added by Zoe

        self.model.eval()
        decoded_smiles = []
        mems = torch.empty((data.shape[0], self.params['d_latent'])).cpu()
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('calcs/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                mols_data = batch_data[:,:input_len] #changed by zoe
                if self.params['ADJ_MAT']:
                    adjMat_data = batch_data[:, input_len:-1] #added by zoe
                    adjMat_data = torch.reshape(adjMat_data, (self.chunk_size, input_len, input_len))
                else:
                    adjMat_data = None
                props_data = batch_data[:,-1]
                if self.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()
                    if self.params['ADJ_MAT']:
                        adjMat_data = adjMat_data.cuda()


                src = Variable(mols_data).long()

                ### Run through encoder to get memory
                _, mem, _ = self.model.encode(src, adjMatrix=adjMat_data)
                start = j*self.batch_size+i*self.chunk_size
                stop = j*self.batch_size+(i+1)*self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()

                ### Decode logic
                if method == 'greedy':
                    decoded = self.greedy_decode(mem) #
                    #outputs seqs of length 57
                else:
                    decoded = None

                if return_str:
                    decoded = decode_mols(decoded, self.params['ORG_DICT'])
                    decoded_smiles += decoded
                else:
                    decoded_smiles.append(decoded)


        if return_mems:
            return decoded_smiles, mems.detach().numpy()
        else:
            return decoded_smiles

    def sample(self, n, method='greedy', sample_mode='rand',
                        sample_dims=None, k=None, return_str=True,
                        condition=[]):
        """
        Method for sampling from memory and decoding back into SMILES strings

        Arguments:
            n (int): Number of data points to sample
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            sample_mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded (list): Decoded smiles data - either decoded SMILES strings or tensor of
                            token ids
        """
        mem = self.sample_from_memory(n, mode=sample_mode, sample_dims=sample_dims, k=k)

        if self.use_gpu:
            mem = mem.cuda()

        ### Decode logic
        if method == 'greedy':
            decoded = self.greedy_decode(mem, condition=condition)
        else:
            decoded = None

        if return_str:
            decoded = decode_mols(decoded, self.params['ORG_DICT'])
        return decoded

    def calc_mems(self, data, log=True, save_dir='memory', save_fn='model_name', save=True):
        """
        Method for calculating and saving the memory of each neural net

        Arguments:
            data (np.array, req): Input array containing SMILES strings
            log (bool): If true, tracks calculation progress in separate log file
            save_dir (str): Directory to store output memory array
            save_fn (str): File name to store output memory array
            save (bool): If true, saves memory to disk. If false, returns memory
        Returns:
            mems(np.array): Reparameterized memory array
            mus(np.array): Mean memory array (prior to reparameterization)
            logvars(np.array): Log variance array (prior to reparameterization)
        """

        data = self.data_gen(data, None, self.params['CHAR_DICT'], self.src_len, self.params['ADJ_MAT'], self.params['ADJ_WEIGHT'])
        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=True)
        save_shape = len(data_iter)*self.params['BATCH_SIZE']
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']
        mems = torch.empty((save_shape, self.params['d_latent'])).cpu()
        mus = torch.empty((save_shape, self.params['d_latent'])).cpu()
        logvars = torch.empty((save_shape, self.params['d_latent'])).cpu()

        self.model.eval()
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('memory/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                input_len = self.src_len+1 #input length including padding and start token
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                mols_data = batch_data[:,:input_len] #changed by zoe
                if self.params['ADJ_MAT']:
                    adjMat_data = batch_data[:, input_len:-1] #added by zoe
                    adjMat_data = torch.reshape(adjMat_data, (self.chunk_size, input_len, input_len))
                else:
                    adjMat_data = None
                props_data = batch_data[:,-1]
                if self.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()
                    if self.params['ADJ_MAT']:
                        adjMat_data = adjMat_data.cuda()

                src = Variable(mols_data).long()

                ### Run through encoder to get memory
                mem, mu, logvar = self.model.encode(src, adjMat_data)
                start = j*self.batch_size+i*self.chunk_size
                stop = j*self.batch_size+(i+1)*self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()
                mus[start:stop, :] = mu.detach().cpu()
                logvars[start:stop, :] = logvar.detach().cpu()

        if save:
            if save_fn == 'model_name':
                save_fn = self.name
            save_path = os.path.join(save_dir, save_fn)
            np.save('{}_mems.npy'.format(save_path), mems.detach().numpy())
            np.save('{}_mus.npy'.format(save_path), mus.detach().numpy())
            np.save('{}_logvars.npy'.format(save_path), logvars.detach().numpy())
        else:
            return mems.detach().numpy(), mus.detach().numpy(), logvars.detach().numpy()


####### Encoder, Decoder and Generator ############

class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return self.proj(x)


############## BOTTLENECKS #################

class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    """
    def __init__(self, size):
        super().__init__()
        conv_layers = []
        in_d = size
        
        first = True
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 4 #used to be 9 now 4
                first = False
            else:
                kernel_size = 3 #used to be 8 now 3
            if i == 2:
                out_d = 64
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(2)))
            in_d = out_d
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x

class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """
    def __init__(self, size):
        super().__init__()
        deconv_layers = []
        in_d = 64
        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 4 - i
            kernel_size = 7 #was 11
            if i == 2:
                out_d = size
                stride = 1
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                                  stride=stride, padding=2)))
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x

############## Property Predictor #################

class PropertyPredictor(nn.Module):
    "Optional property predictor module"
    def __init__(self, d_pp, depth_pp, d_latent):
        super().__init__()
        prediction_layers = []
        for i in range(depth_pp):
            if i == 0:
                linear_layer = nn.Linear(d_latent, d_pp)
            elif i == depth_pp - 1:
                linear_layer = nn.Linear(d_pp, 1)
            else:
                linear_layer = nn.Linear(d_pp, d_pp)
            prediction_layers.append(linear_layer)
        self.prediction_layers = ListModule(*prediction_layers)

    def forward(self, x):
        for prediction_layer in self.prediction_layers:
            x = F.relu(prediction_layer(x))
        return x

############## Embedding Layers ###################

class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


############## Utility Layers ####################

class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2