#!/bin/bash
#SBATCH -p cpu #pool2
#SBATCH --time=300:00:00
#SBATCH --ntasks=16 # was 16 w/ cpu, and 16GB and 12 tasks with pool2
#SBATCH --mem=16GB
#SBATCH --job-name=clock
#SBATCH --output=/project_data/volume0/tpirnia/data/EEGWS/logs/multisubject_train-all/permute/langEEG_perm_%j.log

args=$1
subj=$2


SECONDS=0
echo `date`
echo jobid ${SLURM_JOBID} cpupernode ${SLURM_JOB_CPUS_PER_NODE}
line="/project_data/volume0/tpirnia/data/EEGWS/maybe_repo/classify_across_all_subjects_interp_permute.py /project_data/volume0/tpirnia/data/EEGWS $args $subj"
echo $line

/home/tpirnia/miniconda3/envs/python36c2/bin/python ${line}

duration=$SECONDS
echo "Completed in $(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."