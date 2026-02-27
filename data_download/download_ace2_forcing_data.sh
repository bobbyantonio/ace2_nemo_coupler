#!/bin/bash

INIT_YEAR=1951
END_YEAR=2021
OUTPUT_DIR=${SCRATCH}/ace2_forcing_data/historical_${INIT_YEAR}-${END_YEAR}

mkdir -p ${OUTPUT_DIR}

cd ${OUTPUT_DIR}
for YEAR in $(seq ${INIT_YEAR} ${END_YEAR}); do
    if [ ! -f forcing_${YEAR}.nc ]; then
        wget https://huggingface.co/allenai/ACE2-ERA5/resolve/main/forcing_data/forcing_${YEAR}.nc
    else
        echo "File forcing_${YEAR}.nc already exists, skipping download"
    fi
done
