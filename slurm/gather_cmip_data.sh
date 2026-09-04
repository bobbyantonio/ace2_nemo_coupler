#!/bin/bash 
#SBATCH --job-name=gather-cmip-data
#SBATCH --output=logs/gather-cmip-data-%A.txt 
#SBATCH --partition=transfer
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=50gb

for data_type in "EC-Earth3P_control-1950-3hr"; do
    for variable in "tos"; do
        echo "Gathering data for ${data_type} and variable ${variable}"
        output_dir=/network/group/aopp/predict/HMC005_ANTONIO_EERIE/CMIP6_data/${data_type}/${variable}

        mkdir -p $output_dir
        cd $output_dir

        cp /home/a/antonio/repos/ace2_nemo_coupler/data_download/${data_type}/${variable}_ceda_download.sh ceda_download_temp.sh

        chmod +x ceda_download_temp.sh
        ./ceda_download_temp.sh -s
    done
done
echo "Data gathering completed."