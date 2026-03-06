#!/bin/bash
#SBATCH --job-name=data-archive
#SBATCH --output=logs/data-archive-%A.txt 
#SBATCH --qos=np
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=10gb

cd $HPCPERM/model_runs
# ecfsdir -o $HPCPERM/model_runs/n3.6_ace2_1951-2021_hist_compressed_19510101-20210101_m2 ec:/ecme4254/hpcperm_model_runs/n3.6_ace2_1951-2021_hist_compressed_19510101-20210101_m2

for f in n3.6_ace2_1951_spinupCMIP6_19510101-20210101_m0; do
  cd $HPCPERM/model_runs
  echo `pwd`
  echo "Archving $f"
  ecfsdir -o $HPCPERM/model_runs/${f} ec:/ecme4254/hpcperm_model_runs
  break
done

for f in n3.6_ace2_1951_control_compressed_19510101-20510101_m0 n3.6_ace2_1951_control_compressed_19510101-20510101_m1 n3.6_ace2_1951_control_compressed_19510101-20510101_m2; do
  cd $PERM/old_model_runs
  echo `pwd`
  echo "Archving $f"
  ecfsdir -o $HPCPERM/model_runs/${f} ec:/ecme4254/hpcperm_model_runs
  break
done