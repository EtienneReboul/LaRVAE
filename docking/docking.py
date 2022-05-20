from genericpath import isdir, isfile
import random
import sys
import subprocess
import os
from time import time
import numpy as np
import shutil
from docking.pdb2pqr_example.biomol2pqr import generate_target_H
import Bio.PDB as PDB

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))



class RNASelect(PDB.Select):
    def accept_residue(self, residue):
        print(dir(residue))
        if residue.get_resname() in "AGCU":
            return 1
        else:
            return 0




def set_path(computer):
    if computer == 'etienne-reboul':
        RLDOCK = '/home/etienne-reboul/Canada_internship/RLDOCK-master/RLDOCK'
        OBABEL='/home/etienne-reboul/anaconda3/bin/obabel'
        SMINA='/home/etienne-reboul/smina.static'
    elif computer=='cedar':
        RLDOCK = '/home/retienne/projects/def-jeromew/retienne/RLDOCK/RLDOCK'
        OBABEL='obabel'
        SMINA='/home/retienne/projects/def-jeromew/retienne/smina.static'        
    return OBABEL,SMINA,RLDOCK



def rmdir_dock(proc_id):

    ### use to remvove docking dir after results are saved
    dirpath=os.path.join(script_dir, f'tmp/output_{proc_id}')
    if isdir(dirpath):
        shutil.rmtree(dirpath)
        print(f'process n°{proc_id} has been completed, deleting corresponding dir')
    else:
        print(f'could not find dir: {dirpath}')



def ligand_prep(smile,idx,obabel="obabel",quality="best",pH=7,format="pdbqt",unique_id=1):

    ### create the dir for the current process, needed for serial jobs
    output_dir=os.path.join(script_dir, f'tmp/process_{unique_id}/ligand_{idx}/')
    os.makedirs(output_dir,exist_ok=True)

    ### ligand 3D structure generation 
    output_name = os.path.join(output_dir, f'ligand_{idx}.{format}')
    cmd=f'{obabel} -:{smile} -O {output_name} --gen3D --{quality} -p {pH}'
    subprocess.run(cmd.split())

    return output_name

 

def conformers_prep(ligand_location,obabel="obabel",conformers=30):

    ### if no conformers is needed directly returns ligands names    
    if conformers==1:    
        return ligand_location
    else:

        ### paths setup       
        ligand_name=ligand_location.split("/")[-1]
        conformers_name=ligand_name.replace("ligand_","conformers_ligand_")
        output_filename=ligand_location.rstrip(ligand_name)+conformers_name

        ### conformers 3D structures generation 
        print(f"calculating {conformers} conformers for {ligand_name}")
        cmd=f"{obabel} {ligand_location} -O {output_filename} --conformer --nconf {conformers} --score rmsd --writeconformers"
        subprocess.run(cmd.split())

        return output_filename

        


def target_prep(target='6UC9',obabel="obabel"):

    ### Check if file exists if not download from the PDB
    if isfile(target):
        target_name=target.split("/")[-1].replace(".pdb","") # extract filename and remove .pdb suffix
        filename=os.path.join(script_dir,f'data_docking/RNA_only_{target_name}')
    elif len(target)==4:
        filename=os.path.join(script_dir,f"data_docking/RNA_only_{target}")
        pdbl = PDB.PDBList()
        pdbl.retrieve_pdb_file(target,pdir=os.path.join(script_dir,"data_docking/"))
        target=os.path.join(script_dir,f'data_docking/{target}.cif')
    else:
        print("unable to find a matching file or PDB code")
        NameError

    ###Check if  the target was already prepared 
    processed_target_location=f'{filename}.mol2'
    if isfile(processed_target_location):
        return processed_target_location
    else:
        ### keep only nucleic acids
        model = PDB.PDBParser().get_structure("temp_file", target)
        io = PDB.PDBIO()
        io.set_structure(model)
        io.save(f"{filename}.pdb", RNASelect())
        
        ### set the receptor charges with dedicated software pdb2pqr 
        generate_target_H(PDB_FILE=f"{filename}.pdb",PQR_OUTPUT=f"{filename}.pqr",force_field="AMBER")

        ### convert to mol2 format with openbabel 
        subprocess.run(f'{obabel} {filename}.pqr -O {processed_target_location}'.split())

        return processed_target_location



def SMINA_dock(ligand=str(),  target='drd3', smina="smina", exhaustiveness=16,score_function='vinardo'):

    ### set path for files    
    receptor_file_path= os.path.join(script_dir, f'data_docking/{target}.pdbqt')
    config_file_path= os.path.join(script_dir, f'data_docking/{target}_conf.txt')
    ligand_name=ligand.split("/")[-1].replace(".pdbqt","")
    output_path=ligand.replace(".pdbqt","_out.pdbqt")

    try:
        ### docking 
        print(f"docking {ligand_name} on {target}")
        cmd = f'{smina} --receptor {receptor_file_path} --ligand {ligand}' \
                f' --out {output_path} --config {config_file_path} --exhaustiveness {exhaustiveness} --scoring {score_function} --cpu 1 --quiet'
        subprocess.run(cmd.split(), timeout=1200)

        ### retrive docking scores and compute mean 
        with open(output_path, 'r') as f:
            lines = f.readlines()
            slines = [l for l in lines if l.startswith('REMARK minimizedAffinity')]
            values = [l.split() for l in slines]
            ### In each split string, item with index 2 should be the kcal/mol energy.
            score = [float(v[2]) for v in values]
            score = np.mean(score)
            print(f'score average for {ligand_name} is {score:.2f} kJ/mol')
    except:
        score = 0

    output_dir=ligand.replace(ligand.split("/")[-1],"")

    ### remove dir containing docking results
    try:
        pass
        shutil.rmtree(output_dir)
        print(f"successfully removed dir: {output_dir}")
    except FileNotFoundError:
        pass
        print(f"couldn't found dir: {output_dir}")

    return score

def RLDOCK_dock(conformer=str(),core=8,poses=10,RLDOCK=None,target=str(),unique_id=1):

    try:
        ### creat a copy of sphere dat to be used for this process
        src_location=RLDOCK.rstrip("RLDOCK")+"src/sphere.dat"
        conformer_name=conformer.rstrip(".mol2").split("/")[-1]
        src_copy=src_location.replace(".dat",f'_process_{unique_id}_{conformer_name}.dat')
        shutil.copy(src_location,src_copy)
        output_prefix=conformer.rstrip(".mol2")

        ### docking 
        cmd=f"{RLDOCK} -i {target} -l {conformer} -o {output_prefix} -c {poses} -n {core} -s {src_copy}"
        # print(cmd)

        ### move the sphere.dat to docking directory 
        subprocess.run(cmd.split())
        src_new_location=conformer.rstrip(conformer.split("/")[-1])+ src_copy.split("/")[-1]
        shutil.move(src_copy,src_new_location)

        ### retrieving scores and compute the mean 
        output_path=f'{output_prefix}_SF_high.dat'
        with open(output_path, 'r') as f:
            lines = f.readlines()
            slines = [l for l in lines if l.startswith('<energy>')]
            values = [l.split() for l in slines]
        # In each split string, item with index 2 should be the kcal/mol energy.
            score = [float(v[1]) for v in values]
            score = np.mean(score)
            print(f'score average for {conformer_name} is {score:.2f} kJ/mol')
    except:
        score = 0
    
    ### cleaning up docking directories for space 
    output_dir=conformer.rstrip(conformer.split("/")[-1])
    try:
        pass
        shutil.rmtree(output_dir)
        print(f"successfully removed dir: {output_dir}")
    except FileNotFoundError:
        pass
        print(f"couldn't found dir: {output_dir}")    
        
    return score 
   









