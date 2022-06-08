# LaRVAE

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

7) Once you are done running LaRVAE deactivate the virtual environment with the command  
  $ deactivate  
  

Subsequent times using LaRVAE do the following:  

1) Restore the module collection  
  $ module restore larvaeModules  

2) Activate the virtual environment  
  $ source /pathToNewEnv/EnvName/bin/active  

3) Once you are done running LaRVAE deactivate the virtual environment with the command  
  $ deactivate  
