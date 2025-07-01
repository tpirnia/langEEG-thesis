#!/bin/bash
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=5
#SBATCH --mem=120GB
#SBATCH --job-name=clock
#SBATCH --output=/project_data/volume0/tpirnia/thesis_repository/repo/tmp/logs/contextEEG_%j.log

classifier=$1
subject=$2
window=$3
contexts="US UP"

PROJECT_PATH=/project_data/volume0/tpirnia/thesis_repository
PYTHON_PATH=/home/tpirnia/miniconda3/envs/pynew/bin/python

SECONDS=0
echo $subject

echo `date`
echo jobid ${SLURM_JOBID} cpupernode ${SLURM_JOB_CPUS_PER_NODE}
cd ${PROJECT_PATH}
for context_type in $contexts; do
    line="repo/context_classify_within_context.py -c $classifier -s $subject -w $window --context $context_type"
    echo $line
    /home/tpirnia/miniconda3/envs/pynew/bin/python ${line}
duration=$SECONDS
done
echo "Completed in $(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."



#to run: sbatch repo/context_classify_within_context.sh ridge EP03 1000