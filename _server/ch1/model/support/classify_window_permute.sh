#!/bin/bash
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --job-name=clock
#SBATCH --output=/project_data/volume0/tpirnia/data/EEGWS/logs/phaseEEG_perm_%j.log
#SBATCH --nodes=1
#SBATCH --mem=120GB

args=$1
subj=$2
window=$3

echo `date`
line="/project_data/volume0/tpirnia/data/EEGWS/maybe_repo/classify_window_permute.py /project_data/volume0/tpirnia/data/EEGWS $args $subj $window"
echo $line

/home/tpirnia/miniconda3/envs/python36c2/bin/python ${line}
echo `date`
