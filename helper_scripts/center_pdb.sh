#!/bin/bash
#$ -S /bin/bash
#$ -o log_files/
#$ -e log_files/

conda activate SE3nv2 #SE3nv_new

FN1='../examples/input_pdbs/HH_SQR.pdb'
FN2='../examples/input_pdbs/HH_TET.pdb'
pdb1_resi='B1-11'
pdb2_resi='B1-11'
OUT_FOLDER='../examples/input_pdbs' # out filename will be FN1+ 'centered' and FN2 + 'aligned'

if [ ! -d $OUT_FOLDER ]; then
  mkdir -p $OUT_FOLDER
fi

python center_pdb.py \
    --pdb_fn1=$FN1 \
    --pdb_fn2=$FN2 \
    --pdb1_resi=$pdb1_resi \
    --pdb2_resi=$pdb2_resi \
    --out_folder=$OUT_FOLDER

