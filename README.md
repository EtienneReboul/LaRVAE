# LaRVAE

i) Installation
Your first time using LaRVAE do the following:  

1) Log into compute canada cluster  

2) Load the correct modules with the command  
  $ module load gcc rdkit/2021.09.3 openbabel/3.1.1 python/3.8  

3) Save the modules in a collection  
  $ module save larvaeModules  

4) Clone the git repository into your working directory with the command  
  $ git clone https://github.com/EtienneReboul/LaRVAE.git  

5) After cloning the LaRVAE repository you should set up a virtual environment using the following commands:  
  $ python3 -m pip install --user virtualenv  
  $ python3 -m venv /pathToNewEnv/EnvName  
  $ source /pathToNewEnv/EnvName/bin/active  

6) Then install the packages with the command  
  $ pip install -r /pathToLaRVAE/requirements_3.8.10.txt  

7) To get data run the following scripts
  $
8) Once you are done running LaRVAE deactivate the virtual environment with the command  
  $ deactivate  
  

Subsequent times using LaRVAE do the following:  

1) Restore the module collection  
  $ module restore larvaeModules  

2) Activate the virtual environment  
  $ source /pathToNewEnv/EnvName/bin/active  

3) Once you are done running LaRVAE deactivate the virtual environment with the command  
  $ deactivate  

ii) Download Data  
  1) Download smile data from moses  
    $ python3 scripts1/download_moses.py  
    
  2) Covert smile data into seflie data  
    $ python3 scripts1/get_selfies.py  

iii) Train Initial Model   
  1) Use tranvae to train an initial vae: 
    $python scripts1/train.py --model rnnattn --data_source custom --train_mols_path my_train_data.txt --test_mols_path my_test_data.txt --vocab_path my_vocab.pkl --char_weights_path my_char_weights.npy --save_name my_model  
    
  2) You can use "$ scripts1/train.py --help" to see all command options  

iv) Launch CBAS (note: conditioning by adaptive sampling is not implemented in the current version of LaRVAE)
  1) Launch slurm_master.py for 3 iterations with a command of the form
    $salloc --time=0:30:0 --ntasks=1 --cpus-per-task=10 --mem-per-cpu=2048M --account=def-jeromew srun python cbas/slurm_master.py --name launchName --iters 3  
    
  2) Look in scripts1/parsers1.py to see all options for the sampler, docker and train scripts
