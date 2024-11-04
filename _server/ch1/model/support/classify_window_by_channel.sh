#!/bin/bash 
#SBATCH -p cpu 
#SBATCH --time=300:00:00
#SBATCH --ntasks=6
#SBATCH --mem=16GB
#SBATCH --job-name=clock
#SBATCH --output=/project_data/volume0/tpirnia/data/EEGWS/logs/cross_condition/p_chEEG_%j.log


args=$1
subj=$2
w=$3


SECONDS=0
echo `date`
echo jobid ${SLURM_JOBID}

line="/project_data/volume0/tpirnia/data/EEGWS/maybe_repo/classify_window_by_channel.py
/project_data/volume0/tpirnia/data/EEGWS $args $subj $w"
echo $line

/home/tpirnia/miniconda3/envs/python36c2/bin/python ${line}

duration=$SECONDS
echo "Completed in $(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."
# echo `date`
