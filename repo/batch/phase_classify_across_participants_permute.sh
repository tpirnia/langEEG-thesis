#!/bin/bash
#SBATCH -p cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=5
#SBATCH --mem=120GB
#SBATCH --job-name=clock
#SBATCH --output=/project_data/volume0/tpirnia/thesis_repository/repo/tmp/logs/phaseEEG_loo_%j.log


classifier=$1
subject=$2
window=$3

PROJECT_PATH=/project_data/volume0/tpirnia/thesis_repository
PYTHON_PATH=/home/tpirnia/miniconda3/envs/pynew/bin/python

SECONDS=0
echo $subject

echo `date`
echo jobid ${SLURM_JOBID} cpupernode ${SLURM_JOB_CPUS_PER_NODE}
line="${PROJECT_PATH}/repo/phase_classify_across_participants.py -c $classifier -s $subject -w $window --run_perms"
echo $line

cd ${PROJECT_PATH}
/home/tpirnia/miniconda3/envs/pynew/bin/python ${line}

duration=$SECONDS
echo "Completed in $(($duration / 3600)) hours, $((($duration / 60) % 60)) minutes and $(($duration % 60)) seconds elapsed."

# to run: sbatch repo/batch/phase_classify_across_participants_pernute.sh ridge $subject 1000