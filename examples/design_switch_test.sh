# designing a switch (assuming we have already define the rotation/translation for the switching behavior)

#!/bin/bash
#$ -S /bin/bash
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

INPUT_PDB1='input_pdbs/HH_SQR_centered.pdb'
INPUT_PDB2='input_pdbs/HH_TET_aligned.pdb'


CUTOFF=5
SWITCH_PACE=1
STARTING_PDB=1
conserve_px0_part=True
conserve_px0_all=False
CONTIGS='contigmap.contigs=[45-55/A1-11/5-8/B1-11/45-55]'
switch_px0=True

NUM_DES=5

OUT_FN="example_outputs/HH_45-55_contig1_5-8_contig2_45-55_cut${CUTOFF}_sw${SWITCH_PACE}_start${STARTING_PDB}_part${conserve_px0_part}_all${conserve_px0_all}/HH_45-55_contig1_5-8_contig2_45-55_cut${CUTOFF}_sw${SWITCH_PACE}_start${STARTING_PDB}_part${conserve_px0_part}_all${conserve_px0_all}_switch${switch_px0}_fix_idk"

SWITCH_RT='input_rT/switch_rot.npz'
python ../scripts/run_inference.py \
	--config-name=switch \
	inference.write_trajectory=True \
	inference.recenter=False \
	inference.switch=True \
	inference.center_mot=False \
	inference.conserve_px0_part=$conserve_px0_part \
	inference.conserve_px0_all=$conserve_px0_all \
	inference.switch_pace=1 \
	inference.starting_pdb=1 \
    inference.cutoff=5 \
	inference.num_designs=5 \
	inference.output_prefix=$OUT_FN \
	inference.input_pdb=$INPUT_PDB1 \
	inference.input_pdb2=$INPUT_PDB2 \
	inference.switch_Rt=$SWITCH_RT \
    inference.switch_px0=$switch_px0 \
	'contigmap.length=80-150' \
	$CONTIGS 

