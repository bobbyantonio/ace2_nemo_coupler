#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-5:00:00
#SBATCH --partition=priority-atmproc,shared
#SBATCH --output=logs/run-router-only-%A.txt

source ~/.bashrc
conda activate ece4

BASE_DIR="/network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_flux_comparison_1step/"
# cd /home/a/antonio/repos/ace2_nemo_coupler

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
#             mkdir -p ${BASE_DIR}/ace2/${dt}
#             mkdir -p ${BASE_DIR}/${dt}


#             python -m router  --debug --model-directory ${BASE_DIR}/ace2 --ocean-source era5 --router-data-directory ${BASE_DIR}/${dt} --atmosphere-source ace2 --atmosphere-gridfile /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/grid.nc --atmospheric-timestep-hrs 6 --coupling-timestep-secs 21600 --climatology-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5/climatology --era5-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --start-from-era5 --first-poll-timeout 1 --polling-timeout 1 --initialisation-datetime ${dt};
#         done
#     done
# done


ATM_TYPE="era5-calculated"

mkdir -p ${BASE_DIR}/${ATM_TYPE}
cp ${BASE_DIR}/namelist_cfg ${BASE_DIR}/${ATM_TYPE}/
cp ${BASE_DIR}/nemo-mlatmosphere-coupled-config.yml ${BASE_DIR}/${ATM_TYPE}/
cp ${BASE_DIR}/time.step ${BASE_DIR}/${ATM_TYPE}/
echo "***************** Running for atmosphere ${ATM_TYPE}"
for y in $(seq 1951 1955);
do
    echo "YYYYY Running year ${y}"
    for m in $(seq 1 12);
    do
        echo "MMM Running month ${m}"
        for hour in 0 6;
        do  
            echo "HHHH Running hour ${hour}"
            printf -v dt "%04d%02d%02d-%02d" ${y} ${m} 01 ${hour}

            mkdir -p ${BASE_DIR}/${ATM_TYPE}/${dt}

            # cp ${BASE_DIR}/ace2/${dt}/ace2_6h.nc ${BASE_DIR}/${ATM_TYPE}/${dt}

            python -m router  --debug --model-directory ${BASE_DIR}/${ATM_TYPE} --ocean-source era5 --router-data-directory ${BASE_DIR}/${ATM_TYPE}/${dt} --atmosphere-source ${ATM_TYPE} --atmosphere-gridfile /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/grid.nc --atmospheric-timestep-hrs 6 --coupling-timestep-secs 21600 --climatology-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5/climatology --era5-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --start-from-era5 --first-poll-timeout 1 --polling-timeout 1 --initialisation-datetime ${dt};
        done
    done
done


# y=1951
# hour=0
# m=1

# echo "HHHH Running hour ${hour}"
# printf -v dt "%04d%02d%02d-%02d" ${y} ${m} 01 ${hour}

# mkdir -p ${BASE_DIR}/${ATM_TYPE}/${dt}

# # cp ${BASE_DIR}/ace2/${dt}/ace2_6h.nc ${BASE_DIR}/${ATM_TYPE}/${dt}

# python -m router  --debug --model-directory ${BASE_DIR}/${ATM_TYPE} --ocean-source era5 --router-data-directory ${BASE_DIR}/${ATM_TYPE}/${dt} --atmosphere-source ${ATM_TYPE} --atmosphere-gridfile /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/grid.nc --atmospheric-timestep-hrs 6 --coupling-timestep-secs 21600 --climatology-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5/climatology --era5-directory /network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5 --start-from-era5 --first-poll-timeout 1 --polling-timeout 1 --initialisation-datetime ${dt};