#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-5:00:00
#SBATCH --qos=np
#SBATCH --output=logs/run-router-only-%A.txt

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4

cd /home/a/antonio/repos/ace2_nemo_coupler

# for y in $(seq 1951 1955);
# do
#     echo "YYYYY Running year ${y}"
#     for m in $(seq 1 12);
#     do
#         echo "MMM Running month ${m}"
#         for hour in 0 6 12 18;
#         do  
#             echo "HHHH Running hour ${hour}"
#             printf -v dt "%04d%02d%02d-%02d" ${y} ${m} 01 ${hour}

#             python -m router  --debug --model-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_1step/ --ocean-source era5 --router-data-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_1step/${dt} --atmosphere-source ace2 --atmosphere-gridfile /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/grid.nc --atmospheric-timestep-hrs 6 --coupling-timestep-secs 21600 --climatology-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5/climatology --era5-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --start-from-era5 --first-poll-timeout 1 --polling-timeout 1;
#         done
#     done
# done


for y in $(seq 1951 1955);
do
    echo "YYYYY Running year ${y}"
    for m in $(seq 1 12);
    do
        echo "MMM Running month ${m}"
        for hour in 0 6 12 18;
        do  
            echo "HHHH Running hour ${hour}"
            printf -v dt "%04d%02d%02d-%02d" ${y} ${m} 01 ${hour}

            mkdir -p /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_flux_comparison_1step/era5/${dt}

            python -m router  --debug --model-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_flux_comparison_1step/era5 --ocean-source era5 --router-data-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_flux_comparison_1step/era5/${dt} --atmosphere-source era5 --atmosphere-gridfile /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/grid.nc --atmospheric-timestep-hrs 6 --coupling-timestep-secs 21600 --climatology-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5/climatology --era5-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --start-from-era5 --first-poll-timeout 1 --polling-timeout 1;
        done
    done
done