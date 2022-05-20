"""

Slurm Master

Params to be blended/changed :


"""
import os
import sys
import shutil 

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

print(f'working in directory:{os.getcwd()}')

from scripts.parsers import slurm_master_parser 
import subprocess


def write_sbatch_script_launcher(args):
    """ Writes slurm sh scripts for CEDAR that will be called to launch slurm processes """
    # Sampler 
    with open(os.path.join(script_dir, 'slurm_sampler.sh'), 'w') as file :
        file.write('#!/bin/sh\n')
        file.write('#SBATCH --account=def-jeromew\n')
        file.write('#SBATCH --time=00:45:00\n')
        file.write('#SBATCH --job-name=sampler\n') 
        file.write('#SBATCH --output=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/sampler_%A.out\n')
        file.write('#SBATCH --error=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/sampler_%A.err\n')
        file.write('#SBATCH --cpus-per-task=12')
        file.write('#SBATCH --gres=gpu:1\n') # gpu request
        file.write('#SBATCH --mem=8000M\n')  # memory (per node))
        file.write('python cbas/sampler.py  --model $1 --model_ckpt $2 --sample_mode $3 --name $4 --n_samples $5 --cores $6 --iteration $7 \n')
        
    # Docker
    with open(os.path.join(script_dir, 'slurm_docker.sh'), 'w') as file :
        file.write('#!/bin/sh\n')
        file.write('#SBATCH --account=def-jeromew\n')
        file.write('#SBATCH --job-name=docker\n') 
        file.write('#SBATCH --output=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/docker_%A.out\n')
        file.write('#SBATCH --error=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/docker_%A.err\n')
        
        if args.oracle=="RLDOCK":
            file.write('#SBATCH --mem=10000M\n')# should ask for  10 GB of memory
            file.write('#SBATCH --time=02:00:00\n')
            file.write('#SBATCH --cpus-per-task=8\n')
            file.write('#SBATCH --array=0-999\n')
            file.write('python cbas/docker.py $SLURM_ARRAY_TASK_ID 1000 --server $1 --exhaustiveness $2 --name $3 --cores $4 --oracle $5 --target $6')
        else:
            file.write('#SBATCH --mem=256M\n')# should ask for  256 MB of memory
            file.write('#SBATCH --cpus-per-task=2\n')
            file.write('#SBATCH --time=00:10:00\n')
            file.write('#SBATCH --array=0-2999\n')
            file.write('python cbas/docker.py $SLURM_ARRAY_TASK_ID 30000 --server $1 --exhaustiveness $2 --name $3 --cores $4 --oracle $5 --target $6')
        
        
    
    # Trainer
    with open(os.path.join(script_dir, 'slurm_trainer.sh'), 'w') as file :
        file.write('#!/bin/sh\n')
        file.write('#SBATCH --account=def-jeromew\n')
        file.write('#SBATCH --time=00:40:00\n')
        file.write('#SBATCH --job-name=trainer\n') 
        file.write('#SBATCH --output=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/trainer_%A.out\n')
        file.write('#SBATCH --error=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/trainer_%A.err\n')
        file.write('#SBATCH --cpus-per-task=12')
        file.write('#SBATCH --gres=gpu:1\n') # gpu request
        file.write('#SBATCH --mem=8000M\n')  # memory (per node))
        file.write('python cbas/trainer.py --iteration $1 --name $2  --quantile $3')

def launch_job(script_name="scripts",previous_job_ID=None,args_list=[],):

    ### set up begining of sbatch command
    script_path=os.path.join(script_dir,script_name)
    cmd='sbatch '
    if previous_job_ID is None:
        cmd+=f'{script_path} '
    else:
        cmd+=f'--depend=afterany:{previous_job_ID} {script_path} '
    
    ### add extra argument 
    extra_args=str()
    for arg in args_list:
        extra_args+=f'{arg} ' 
    extra_args.rstrip()
    cmd+=extra_args
    print(cmd)

    ### launch job and retrieve job_ib
    cmd_output=subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    job_ID=cmd_output.split()[3]

    return job_ID



if __name__ == '__main__':

    ### get script argument 
    parser=slurm_master_parser()
    args, _ = parser.parse_known_args()
    
    ###  write the script for ComputeCanada
    if args.server == 'cedar':
        write_sbatch_script_launcher(args)

    ### make a copy of original alphabet/vocab,weights, and base model  
    shutil.copy(args.alphabet_path,os.path.join(script_dir,f'../data/{args.name}_char_dict.pkl'))
    shutil.copy(args.weights_path,os.path.join(script_dir,f'../data/{args.name}_char_weights.npy'))
    shutil.copy(args.prior_path,os.path.join(script_dir,f'../checkpoints/000_{args.name}.ckpt'))
    
    ### variable to be initialised before loop
    job_ID = None

    for iteration in range(args.iters):

        ### SAMPLING
        sampler_args=[args.model,args.prior_path,args.sample_mode,args.name,args.n_samples,args.main_cores,iteration] 
        # the prior_path is always inputed but only used once at the beginning
        job_ID =launch_job(script_name='slurm_sampler.sh',previous_job_ID=job_ID,args_list=sampler_args)

        ### DOCKING
        docker_args=[args.server,args.exhaustiveness,args.name,args.docker_cores,args.oracle,args.target]
        job_ID =launch_job(script_name='slurm_docker.sh',previous_job_ID=job_ID,args_list=docker_args)
        
        ### TRAINER
        trainer_args=[iteration,args.name,args.quantile]
        job_ID =launch_job(script_name='slurm_trainer.sh',previous_job_ID=job_ID,args_list=trainer_args)

        print(f'launched iteration n°{iteration+1}')
