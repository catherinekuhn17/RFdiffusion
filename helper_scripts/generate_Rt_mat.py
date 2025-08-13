import numpy as np
import os
import glob
from argparse import ArgumentParser
import inp_utils as iu

def get_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--pdb_fn1", type=str, help="fn of pdb 1")
    p.add_argument("--pdb_fn2", type=str, help="fn of pdb 2")
    p.add_argument("--pdb1_resi", type=str, help="residues you want to define Rt by of pdb1 (format example: A1-5/A7/B3-8")
    p.add_argument("--pdb2_resi", type=str, help="residues you want to define Rt by of pdb2")   
    p.add_argument("--out_folder", type=str, help="folder to write out to")
    p.add_argument("--out_fn", type=str, help="filename to write out to")
    args = p.parse_args()
    if argv is not None:
        args = p.parse_args(argv) # for use when testing
    else:
        args = p.parse_args()
    return args

def main():
    args = get_args()
    # load in pdbs
    pdb_fn1 = args.pdb_fn1
    pdb1 = iu.parse_pdb(pdb_fn1)
    
    pdb_fn2 = args.pdb_fn2
    pdb2 = iu.parse_pdb(pdb_fn2)    
    
    # align pdb1 and pdb2 by residues
    resi_idx0_1 = iu.get_resi_idx0(pdb1, args.pdb1_resi)
    resi_idx0_2 = iu.get_resi_idx0(pdb2, args.pdb2_resi)
                   
    # get Rt in each direction
    R21,t21 = iu.get_Rt(pdb2['xyz'][resi_idx0_2][:,1].reshape(-1, 3),
                     pdb1['xyz'][resi_idx0_1][:,1].reshape(-1, 3))

    R12,t12 = iu.get_Rt(pdb1['xyz'][resi_idx0_1][:,1].reshape(-1, 3),
                     pdb2['xyz'][resi_idx0_2][:,1].reshape(-1, 3))
    np.savez(f'{args.out_folder}/{args.out_fn}.npz', 
                 R12=R12,
                 R21=R21,
                 t12=t12,
                 t21=t21)
    print('generated Rt matrix')

if __name__ == "__main__":
    main()