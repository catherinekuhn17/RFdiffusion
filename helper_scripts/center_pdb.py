import numpy as np
import os
import glob
from argparse import ArgumentParser
import inp_utils as iu

def get_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--pdb_fn1", type=str, help="fn of pdb you want to center")
    p.add_argument("--pdb_fn2", type=str, help="fn of pdb you want to align")
    p.add_argument("--pdb1_resi", type=str, help="residues you want to align by of pdb1 (format example: A1-5/A7/B3-8")
    p.add_argument("--pdb2_resi", type=str, help="residues you want to align by of pdb2")    
    p.add_argument("--out_folder", type=str, help="folder to write output pdbs to")
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
    pre1 = pdb_fn1.split('/')[-1].split('.pdb')[0]
    pdb1 = iu.parse_pdb(pdb_fn1)
    
    pdb_fn2 = args.pdb_fn2
    pre2 = pdb_fn2.split('/')[-1].split('.pdb')[0]
    pdb2 = iu.parse_pdb(pdb_fn2)    
    
    # center pdb1 by Ca
    ca_com_pdb = pdb1['xyz'][:, 1, :].mean(axis=0, keepdims=True)
    centered_pdb1_xyz = pdb1['xyz'] - ca_com_pdb
    pdb1['xyz'] = centered_pdb1_xyz
    
    # align pdb1 and pdb2 by residues
    resi_idx0_1 = iu.get_resi_idx0(pdb1, args.pdb1_resi)
    resi_idx0_2 = iu.get_resi_idx0(pdb2, args.pdb2_resi)

    R,t = iu.get_Rt(pdb2['xyz'][resi_idx0_2][:,1].reshape(-1, 3),
                    pdb1['xyz'][resi_idx0_1][:,1].reshape(-1, 3))

    pdb2['xyz'] = ((R @ pdb2['xyz'].flatten().reshape(-1,3).T) + t).T.reshape(22,-1,3)
    
    iu.writepdb(f'{args.out_folder}/{pre1}_centered.pdb', 
                pdb1['xyz'], 
                pdb1['seq'], 
                binderlen=None, 
                idx_pdb=pdb1['idx'], 
                bfacts=pdb1['plddt_val'], 
                chain_idx=np.array(pdb1['pdb_idx'])[:,0])
    
    iu.writepdb(f'{args.out_folder}/{pre2}_aligned.pdb', 
                pdb2['xyz'], 
                pdb2['seq'], 
                binderlen=None, 
                idx_pdb=pdb2['idx'], 
                bfacts=pdb2['plddt_val'], 
                chain_idx=np.array(pdb2['pdb_idx'])[:,0])
    print('centered pdbs')

if __name__ == "__main__":
    main()