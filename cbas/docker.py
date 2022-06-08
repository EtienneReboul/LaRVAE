"""
Script to be called by a job array slurm task.
Takes the path to a csv and annotate it with docking scores
"""
import os
import sys
import time
import pandas as pd
from multiprocessing import Pool
import selfies as sf
from functools import partial

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

print(f'working in directory:{os.getcwd()}')

from docking.docking import SMINA_dock, set_path, ligand_prep,conformers_prep,RLDOCK_dock,rmdir_dock,target_prep
from scripts1.parsers1 import docker_parser


def write_results_to_csv(name,unique_id,scores_list,selfies_list):
    ### setting path for csv file 
    dirname = os.path.join(script_dir, f'../docking/docking_results/{name}/')
    os.makedirs(dirname, exist_ok=True)
    file_path = os.path.join(dirname, f"{unique_id}.csv")
    
    ### using pandas for practical reason 
    results=pd.DataFrame()
    results['selfies']=selfies_list
    results['scores']=scores_list
    try:
        results.to_csv(file_path,index=False)
        print(f'sucessfully saved docking resulsts of process n°{unique_id} to {file_path} ')
    except:
        print(f'was unable to save results :(')


def SMINA_pipeline(selfies_list, server, unique_id, name, target='drd3',  exhaustiveness=16,cores=10):

    ### recorded start time
    start_time=time.time()

    ### path for software 
    obabel_path,smina_path,_ = set_path(server)

    ### generate ligands 
    list_smiles=[sf.decoder(selfies) for selfies in selfies_list]
    list__locations_ligand=[ligand_prep(smile,idx=i,obabel=obabel_path,format="pdbqt",unique_id=unique_id) for i,smile in enumerate(list_smiles)] # obabel already multiprocessed

    ### Docking 
    ### multiprocessing if more than 1 core
    if cores==1:
        list_scores=[SMINA_dock(ligand, exhaustiveness=exhaustiveness,smina=smina_path,target=target) for ligand in list__locations_ligand]
    else:
        pool=Pool(cores)
        list_scores=pool.map(partial(SMINA_dock,exhaustiveness=exhaustiveness,smina=smina_path,target=target),list__locations_ligand)
        pool.close()
        pool.join()

    ### write results to csv
    write_results_to_csv(name=name,unique_id=unique_id,scores_list=list_scores,selfies_list=selfies_list)

    ### clean up docking directory
    rmdir_dock(unique_id)

    ### print duration 
    duration=time.time()-start_time
    print(f'It took {duration:.2f}s to dock {len(selfies_list)} molecule(s) with {cores} core(s) ')


def RLDOCK_pipeline(selfies_list, server, unique_id, name, target='6UC9'):

    ### recorded start time
    start_time=time.time()

    ### set path for software 
    obabel_path,_,RLDOCK_path = set_path(server)
    
    ### generate ligands and their conformers 
    list_smiles=[sf.decoder(selfies) for selfies in selfies_list]
    list_locations_ligand=[ligand_prep(smile,idx=i,obabel=obabel_path,format="mol2",unique_id=unique_id) for i,smile in enumerate(list_smiles)] 
    list_locations_conformers=[conformers_prep(ligand_location,obabel=obabel_path,conformers=30) for ligand_location in list_locations_ligand]

    ### receptor preparation 
    processed_target_location=target_prep(target=target,computer=server)
    
    ###docking
    list_scores=[RLDOCK_dock(conformer=conformer_location,RLDOCK=RLDOCK_path,target=processed_target_location,unique_id=unique_id) for conformer_location in list_locations_conformers]

    ### write results to csv
    write_results_to_csv(name=name,unique_id=unique_id,scores_list=list_scores,selfies_list=selfies_list)

    ### clean up docking directory
    rmdir_dock(unique_id)

    ### print duration 
    duration=time.time()-start_time
    print(f'It took {duration:.2f}s to dock {len(selfies_list)} molecule(s) with allocated core(s) ')



def main(proc_id, num_procs, server,nb_cores, exhaustiveness, name, oracle, target):
    # parse the docking task of the whole job array and split it
    dirname=os.path.join(script_dir,'../docking/data_docking/VAE_generated_samples/')
    csv_path = os.path.join(dirname, f'{name}_sampled_selfies.csv')
    data=pd.read_csv(csv_path)
    list_selfies = list(data['mol'])

    n = len(list_selfies)
    chunk_size, rab = n // num_procs, n % num_procs
    chunk_min, chunk_max = proc_id * chunk_size, min((proc_id + 1) * chunk_size, n)
    list_data = list_selfies[chunk_min:chunk_max]
    # N = chunk_size*num_procs + rab
    # Share rab between procs
    if proc_id < rab:
        list_data.append(list_selfies[-(proc_id + 1)])


    # Do the docking and dump results
    if oracle == 'smina':
        SMINA_pipeline(list_data,
                  target=target,
                  name=name,
                  server=server,
                  cores=nb_cores,
                  unique_id=proc_id,
                  exhaustiveness=exhaustiveness)
    elif oracle == 'RLDOCK':
        RLDOCK_pipeline(list_data,
                  target=target,
                  name=name,
                  server=server,
                  unique_id=proc_id)
    else:
        raise ValueError(f'oracle {oracle} not implemented')


if __name__ == '__main__':
    pass

    parser = docker_parser()
    args, _ = parser.parse_known_args()

    try:
        proc_id, num_procs = int(sys.argv[1]), int(sys.argv[2])
    except IndexError:
        print('We are not using the args as usually in docker.py')
        proc_id, num_procs = 2, 10
    except ValueError:
        print('We are not using the args as usually in docker.py')
        proc_id, num_procs = 2, 10

    main(proc_id=proc_id,
         num_procs=num_procs,
         server=args.server,
         exhaustiveness=args.exhaustiveness,
         name=args.name,
         nb_cores=args.cores,
         oracle=args.oracle,
         target=args.target)
