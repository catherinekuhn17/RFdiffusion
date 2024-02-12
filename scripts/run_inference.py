#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os, time, pickle
import torch
from omegaconf import OmegaConf
import hydra
import logging
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob


def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf) #added in pdb thing

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(".*_(\d+)\.pdb$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1
    print(design_startnum)
    print(sampler.inf_conf.num_designs)
    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix={}
        if sampler.inf_conf.switch: # set up design names for both outputs
            out_prefix[1] = f"{sampler.inf_conf.output_prefix}_{i_des}_1"
            out_prefix[2] = f"{sampler.inf_conf.output_prefix}_{i_des}_2"
        else:
            out_prefix[1] = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix[1]}")
        if sampler.inf_conf.cautious and os.path.exists(out_prefix[1] + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix[1]}.pdb already exists."
            )
            continue
        # initializing starting noise for state 1
        x_init, seq_init = sampler.sample_init() 
        denoised_xyz_stack = {}
        px0_xyz_stack = {}
        seq_stack = {}
        plddt_stack = {}

        # will always make atleast one output
        denoised_xyz_stack[1] = []
        px0_xyz_stack[1] = []
        seq_stack[1] = []
        plddt_stack[1] = []

        if sampler.inf_conf.switch: # seconds output if making a switch
            denoised_xyz_stack[2] = []
            px0_xyz_stack[2] = []
            seq_stack[2] = []
            plddt_stack[2] = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        
        
        if sampler.inf_conf.switch:
            switch_pace = sampler.inf_conf.switch_pace # pace at which denoising switching happens
            cutoff = sampler.inf_conf.cutoff
            starting_pdb = sampler.inf_conf.starting_pdb
            
            
        # Loop over number of reverse diffusion time steps.
        starting_T = sampler.t_step_input
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            stay=False

            if sampler.inf_conf.switch: # if making a switch
                if t > cutoff: # if in switching regime:
                    if t==starting_T: # if we are on the FIRST STEP
                        k = starting_pdb # start with whichever we want to denoise first
                    else: #otherwise, we want to alternate
                        val = (starting_T-t)%(2*switch_pace) - switch_pace
                        if val < 0:
                            k = starting_pdb
                            if val!=-switch_pace:
                                stay=True # determines if we maintain current state
                        else:
                            k = [1,2]
                            k.remove(starting_pdb)
                            k = k[0]
                            if val != 0: 
                                stay=True # determines if we maintain current state
                    # perform a denoising step:                   
                    
                    px0, x_t, seq_t, plddt = sampler.sample_step(cutoff=cutoff, stay=stay, k=k, t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step)

                    px0_xyz_stack_tmp = px0_xyz_stack[k]
                    px0_xyz_stack_tmp.append(px0)
                    px0_xyz_stack[k] = px0_xyz_stack_tmp
                    
                    denoised_xyz_stack_tmp = denoised_xyz_stack[k] 
                    denoised_xyz_stack_tmp.append(x_t)
                    denoised_xyz_stack[k] = denoised_xyz_stack_tmp
                    
                    seq_stack_tmp = seq_stack[k] 
                    seq_stack_tmp.append(seq_t)
                    seq_stack[k] = seq_stack_tmp
                    
                    plddt_stack_tmp = plddt_stack[k] 
                    plddt_stack_tmp.append(plddt[0])
                    plddt_stack[k] = plddt_stack_tmp
 
                        
                else: # if in "optimizing" regime
                    # do denoise step for design 1 and then denoise step for design 2 (these are now separate)
                    for k in [1,2]: 
                       #switch_resi_idx = range(sampler.contig_map.hal_idx0[-1], len(sampler.contig_map.ref_idx0))
                        px0, x_t, seq_t, plddt = sampler.sample_step(cutoff=cutoff,
                                                                     stay=True, k=k,
                            t=t, x_t=denoised_xyz_stack[k][-1], seq_init=seq_stack[k][-1], final_step=sampler.inf_conf.final_step
                         )
                        
                        px0_xyz_stack_tmp = px0_xyz_stack[k]
                        px0_xyz_stack_tmp.append(px0)
                        px0_xyz_stack[k] = px0_xyz_stack_tmp

                        denoised_xyz_stack_tmp1 = denoised_xyz_stack[k] 
                        denoised_xyz_stack_tmp1.append(x_t)
                        denoised_xyz_stack[k] = denoised_xyz_stack_tmp1

                        seq_stack_tmp = seq_stack[k] 
                        seq_stack_tmp.append(seq_t)
                        seq_stack[k] = seq_stack_tmp

                        plddt_stack_tmp = plddt_stack[k] 
                        plddt_stack_tmp.append(plddt[0])
                        plddt_stack[k] = plddt_stack_tmp       
                
            else: # if not switching, just do normal diffusion (on just one state)
                k=1 # we are always in state 1
                px0, x_t, seq_t, plddt = sampler.sample_step(cutoff=cutoff, first_step=first_step, stay=True, k=k, t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
                    )
                px0_xyz_stack_tmp = px0_xyz_stack[k]
                px0_xyz_stack_tmp.append(px0)
                px0_xyz_stack[k] = px0_xyz_stack_tmp

                denoised_xyz_stack_tmp1 = denoised_xyz_stack[k] 
                denoised_xyz_stack_tmp1.append(x_t)
                denoised_xyz_stack[k] = denoised_xyz_stack_tmp1

                seq_stack_tmp = seq_stack[k] 
                seq_stack_tmp.append(seq_t)
                seq_stack[k] = seq_stack_tmp

                plddt_stack_tmp = plddt_stack[k] 
                plddt_stack_tmp.append(plddt[0])
                plddt_stack[k] = plddt_stack_tmp    

        # Flip order for better visualization in pymol
        for k in px0_xyz_stack.keys():
            denoised_xyz_stack_tmp = torch.stack(denoised_xyz_stack[k])
            denoised_xyz_stack_tmp = torch.flip(
                denoised_xyz_stack_tmp,
                [
                    0,
                ],
            )
            px0_xyz_stack_tmp = torch.stack(px0_xyz_stack[k])
            px0_xyz_stack_tmp = torch.flip(
                px0_xyz_stack_tmp,
                [
                    0,
                ],
            )

            # For logging -- don't flip
            plddt_stack_tmp = torch.stack(plddt_stack[k])

            # Save outputs
            os.makedirs(os.path.dirname(out_prefix[k]), exist_ok=True)
            final_seq = seq_stack[k][-1]
            
            np.save('idk.npy', np.array(final_seq))
            # Output glycines, except for motif region
            final_seq = torch.where(
                torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
            )  # 7 is glycine

            bfacts = torch.ones_like(final_seq.squeeze())
            # make bfact=0 for diffused coordinates
            bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
            # pX0 last step
            out = f"{out_prefix[k]}.pdb"

            # Now don't output sidechains

            writepdb(
                out,
                denoised_xyz_stack_tmp[0, :, :4],
                final_seq,
                sampler.binderlen,
                chain_idx=sampler.chain_idx#,
             #   bfacts=bfacts,
            )

            # run metadata
            trb = dict(
                config=OmegaConf.to_container(sampler._conf, resolve=True),
                plddt=plddt_stack_tmp.cpu().numpy(),
                device=torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "CPU",
                time=time.time() - start_time,
            )
            if hasattr(sampler, "contig_map"):
                for key, value in sampler.contig_map.get_mappings().items():
                    trb[key] = value
            with open(f"{out_prefix[k]}.trb", "wb") as f_out:
                pickle.dump(trb, f_out)
    
            if sampler.inf_conf.write_trajectory:
                # trajectory pdbs
                traj_prefix = (
                    os.path.dirname(out_prefix[k]) + "/traj/" + os.path.basename(out_prefix[k])
                )
                os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
    
                out = f"{traj_prefix}_Xt-1_traj.pdb"
                writepdb_multi(
                    out,
                    denoised_xyz_stack_tmp,
                    bfacts,
                    final_seq.squeeze(),
                    use_hydrogens=False,
                    backbone_only=False,
                    chain_ids=sampler.chain_idx,
                )
    
                out = f"{traj_prefix}_pX0_traj.pdb"
                writepdb_multi(
                    out,
                    px0_xyz_stack_tmp,
                    bfacts,
                    final_seq.squeeze(),
                    use_hydrogens=False,
                    backbone_only=False,
                    chain_ids=sampler.chain_idx,
                )
        
        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
