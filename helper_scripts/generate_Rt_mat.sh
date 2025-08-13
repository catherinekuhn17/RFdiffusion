#!/bin/bash
#$ -S /bin/bash
#$ -o log_files/
#$ -e log_files/

conda activate SE3nv2

FN1='../examples/input_pdbs/HH_SQR_centered.pdb'
FN2='../examples/input_pdbs/HH_TET_aligned.pdb'
pdb1_resi='A1-11'
pdb2_resi='A1-11'
OUT_FOLDER='../examples/input_rT'
OUT_FN='switch_rot2'


if [ ! -d $OUT_FOLDER ]; then
  mkdir -p $OUT_FOLDER
fi

python generate_Rt_mat.py \
    --pdb_fn1=$FN1 \
    --pdb_fn2=$FN2 \
    --pdb1_resi=$pdb1_resi \
    --pdb2_resi=$pdb2_resi \
    --out_folder=$OUT_FOLDER \
    --out_fn=$OUT_FN \

