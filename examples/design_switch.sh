# designing a switch (assuming we have already define the rotation/translation for the switching behavior)

#!/bin/bash
#$ -S /bin/bash
#$ -o log_files/
#$ -e log_files/
#$ -cwd
#$ -j y
#$ -r y
#$ -l mem_free=10G
#$ -l h_rt=96:00:00


LD_PRELOAD=/wynton/home/kortemme/ckuhn/Desktop/kortemme/glibc-2.27/opt/glibc-2.27/lib/libm.so.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wynton/home/kortemme/ckuhn/anaconda3/envs/SE3nv2/lib/
export CUDA_VISIBLE_DEVICES=$SGE_GPU
conda activate SE3nv2
export HYDRA_FULL_ERROR=1


INPUT_PDB1='input_pdbs/HH_SQR_centered2.pdb'
INPUT_PDB2='input_pdbs/HH_TET_aligned2.pdb'


CUTOFF=$1
SWITCH_PACE=$2
STARTING_PDB=$3
conserve_px0_part=$4 
conserve_px0_all=$5

CONTIGS='contigmap.contigs=[45-55/A1-11/5-8/B1-11/45-55]'

NUM_DES=5

OUT_FN="example_outputs/HH_45-55_contig1_5-8_contig2_45-55_cut${CUTOFF}_sw${SWITCH_PACE}_start${STARTING_PDB}_part${conserve_px0_part}_all${conserve_px0_all}/HH_45-55_contig1_5-8_contig2_45-55_cut${CUTOFF}_sw${SWITCH_PACE}_start${STARTING_PDB}_part${conserve_px0_part}_all${conserve_px0_all}"


SWITCH_RT='input_rT/switch_rot2.npz'

python ../scripts/run_inference.py \
    --config-name=switch \
	inference.write_trajectory=True \
    inference.recenter=False \
	inference.switch=True \
	inference.center_mot=False \
	inference.conserve_px0_part=$conserve_px0_part \
    inference.conserve_px0_all=$conserve_px0_all \
	inference.switch_pace=$SWITCH_PACE \
	inference.starting_pdb=$STARTING_PDB \
    inference.cutoff=$CUTOFF \
	inference.num_designs=$NUM_DES \
	inference.output_prefix=$OUT_FN \
    inference.input_pdb=$INPUT_PDB1 \
	inference.input_pdb2=$INPUT_PDB2 \
	inference.switch_Rt=$SWITCH_RT \
	'contigmap.length=80-150' \
	$CONTIGS 

