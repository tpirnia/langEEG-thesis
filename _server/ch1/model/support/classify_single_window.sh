#!/bin/bash
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --mem=120GB
#SBATCH --job-name=clock
#SBATCH --output=/project_data/volume0/tpirnia/data/EEGWS/logs/phaseEEG_w_%j.log


args=$1
subj=$2
w=$3

echo `date`
line="/project_data/volume0/tpirnia/data/EEGWS/maybe_repo/classify_single_window.py /project_data/volume0/tpirnia/data/EEGWS $args $subj $w"
echo $line

/home/tpirnia/miniconda3/envs/python36c2/bin/python ${line}


echo `date`
