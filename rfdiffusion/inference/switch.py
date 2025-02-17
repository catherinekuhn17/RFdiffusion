import numpy as np
from scipy.spatial.transform import Rotation
import functools as fn
import torch

format_rots = lambda r: torch.tensor(r).float()


class SwitchGen():
    def __init__(self, switch_rot):
        
        switch_Rt = np.load(switch_rot)
        self.R12 = torch.tensor(switch_Rt['R12'], dtype=torch.float32)
        self.R21 = torch.tensor(switch_Rt['R21'], dtype=torch.float32)
        self.t12 = torch.tensor(switch_Rt['t12'], dtype=torch.float32)
        self.t21 = torch.tensor(switch_Rt['t21'], dtype=torch.float32)
        
    def applyRt(self, coords, resi_list, R, t):
        xt = torch.clone(coords)
        xt_switch = xt[resi_list]
        flat_xt = torch.reshape(torch.flatten(xt_switch), (-1,3))
        new_xt = torch.transpose(torch.mm(R, torch.transpose(flat_xt, 0, 1))+t,0,1)
        new_xt_reshape = torch.reshape(new_xt,(-1,14,3))
        xt[resi_list] = new_xt_reshape
        return xt
