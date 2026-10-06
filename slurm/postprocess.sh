#!/bin/bash
#SBATCH --job-name=postprocess-ace2
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=np
#SBATCH --mem=50gb
#SBATCH --output=logs/postprocess-ace2-%A.txt

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4




# # Dates required: 
# for year in 2035;
# do  
#     echo "Postprocessing ${year}12"
#     yearp1=$((year+1))
#     yearm4=$((year-4))
#     EXPID="n3.6_ace2_spinupCMIP6_laplacian_${yearm4}0101-${yearp1}0101_m0"
    
#     cp $PERM/repos/ace2_nemo_coupler/postprocess.py $SCRATCH/run_dir/${EXPID}/postprocess.py;
#     cd $SCRATCH/run_dir/${EXPID};
#     python postprocess.py --model-directory $SCRATCH/run_dir/${EXPID} --ocean-source "nemo" --atmosphere-source "ace2" --router-data-directory $SCRATCH/run_dir/${EXPID}/router --results-data-directory $HPCPERM/model_runs/${EXPID}/ --coupling-timestep-secs 21600 --overwrite --year $year --month 12 --first-poll-timeout 1 --polling-timeout 1;
# done
# for year in 2040 2045;
# do  
#     echo "Postprocessing ${year}12"
#     yearp1=$((year+1))
#     yearm4=$((year-4))
#     EXPID="n3.6_ace2_spinupCMIP6_laplacian_${yearm4}0101-${yearp1}0101_m0"
    
#     cp $PERM/repos/ace2_nemo_coupler/postprocess.py $SCRATCH/run_dir/${EXPID}/postprocess.py;
#     cd $SCRATCH/run_dir/${EXPID};
#     python postprocess.py --model-directory $SCRATCH/run_dir/${EXPID} --ocean-source "nemo" --atmosphere-source "ace2" --router-data-directory $SCRATCH/run_dir/${EXPID}/router --results-data-directory $HPCPERM/model_runs/${EXPID}/ --coupling-timestep-secs 21600 --overwrite --year $year --month 12;
# done

for year in 2083 2084 2085;
do
    echo "Postprocessing ${year}"

    EXPID="n3.6_ace2_spinupCMIP6_laplacian_20810101-20860101_m0"

    cp $PERM/repos/ace2_nemo_coupler/postprocess.py $SCRATCH/run_dir/${EXPID}/postprocess.py;
    cd $SCRATCH/run_dir/${EXPID};

    python postprocess.py --model-directory $SCRATCH/run_dir/${EXPID} --ocean-source "nemo" --atmosphere-source "ace2" --router-data-directory $SCRATCH/run_dir/${EXPID}/router --results-data-directory /home/ecme4254/hpcperm/model_runs/n3.6_ace2_spinupCMIP6_laplacian_19510101-21010101_m0 --coupling-timestep-secs 21600 --overwrite --year $year --month 12;
done

