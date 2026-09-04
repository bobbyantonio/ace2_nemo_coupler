#/bin/sh

script_dir=$(pwd)

# # loop over all files in this directory
# # for data_type in "EC-Earth3P_hist-1950" "EC-Earth3P_control-1950" "EC-Earth3_historical"; do
# for data_type in "EC-Earth3_historical"; do
#     # for variable in "tos" "tas" "pr" "siconc" "sithick" "tauu" "tauv" "thetao" "zos" "hfls" "hfss" "rlds" "rlus" "rsus" "rsds" "mlotst" "uo" "vo" "uas" "vas"; do
        
#         output_dir=$SCRATCH/ece3_cmip6_data_download/${data_type}/${variable}

#         mkdir -p $output_dir
#         cd $output_dir

#         cp ${script_dir}/${data_type}/${variable}_ceda_download.sh ceda_download_temp.sh

#         chmod +x ceda_download_temp.sh
#         ./ceda_download_temp.sh -s
#     done
# done


# for data_type in "EC-Earth3P_hist-1950"; do
#     for variable in "uas" "vas"; do
        
#         output_dir=$SCRATCH/ece3_cmip6_data_download/${data_type}/${variable}

#         mkdir -p $output_dir
#         cd $output_dir

#         cp ${script_dir}/${data_type}/${variable}_ceda_download.sh ceda_download_temp.sh

#         chmod +x ceda_download_temp.sh
#         ./ceda_download_temp.sh -s
#     done
# done

# for data_type in "EC-Earth3P_hist-1950"; do
#     for variable in "psl"; do
        
#         output_dir=$SCRATCH/ece3_cmip6_data_download/${data_type}/${variable}

#         mkdir -p $output_dir
#         cd $output_dir

#         cp ${script_dir}/${data_type}/${variable}_ceda_download.sh ceda_download_temp.sh

#         chmod +x ceda_download_temp.sh
#         ./ceda_download_temp.sh -s
#     done
# done

for data_type in "EC-Earth3P_control-1950-3hr"; do
    for variable in "tos"; do
        
        output_dir=/network/group/aopp/predict/HMC005_ANTONIO_EERIE/CMIP6_data/${data_type}/${variable}

        mkdir -p $output_dir
        cd $output_dir

        cp ${script_dir}/${data_type}/${variable}_ceda_download.sh ceda_download_temp.sh

        chmod +x ceda_download_temp.sh
        ./ceda_download_temp.sh -s
    done
done