#!/bin/bash
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=5
#SBATCH --mem=48GB
#SBATCH --job-name=clock
#SBATCH --output=/project_data/volume0/tpirnia/data/EEGWS/logs/multisubject_train-all/langEEG_c_%j.log

args=$1
subj=$2
w=$3

SECONDS=0
echo `date`
echo jobid ${SLURM_JOBID} cpupernode ${SLURM_JOB_CPUS_PER_NODE}
line="/project_data/volume0/tpirnia/data/EEGWS/maybe_repo/classify_across_all_subjects_interp.py /project_data/volume0/tpirnia/data/EEGWS $args $subj $w"
echo $line

/home/tpirnia/miniconda3/envs/python36c2/bin/python ${line}

duration=$SECONDS
echo "Completed in $(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."