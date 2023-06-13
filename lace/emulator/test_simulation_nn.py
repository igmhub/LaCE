import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn

# LaCE modules
from lace.emulator import gp_emulator
from lace.emulator import pnd_archive
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import poly_p1d


import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler

cosmo_fid=camb_cosmo.get_cosmology()

def test_sim(emulator, test_data, emuparams, paramLims, device, yscaling, kmax_Mpc_test=3, Nz=11, Nk=32,ndeg=4, return_coeffs=False ):
    
    kMpc = test_data[0]["k_Mpc"]
    k_mask=(kMpc<kmax_Mpc_test) & (kMpc>0)
    Nk = len(kMpc[k_mask])
    
    print(Nk)

    fractional_error=np.zeros((Nz,Nk))
    emu_p1ds = np.zeros((Nz, Nk))
    emu_p1derrs = np.zeros((Nz, Nk))
    true_p1ds=np.zeros((Nz,Nk))
    emu_p1ds_transformed=np.zeros((Nz,Nk))
    
    coeffs=np.zeros((Nz,ndeg+1))
    coeffs_err=np.zeros((Nz,ndeg+1))
    
    with torch.no_grad():
        # for each entry (z) in truth/test simulation, compute residuals
        for aa,item in enumerate(test_data):
            # figure out redshift for this entry
            z=item["z"]
            # true p1d (some sims have an extra k bin, so we need to define mask again)
            true_k=item["k_Mpc"]
            k_mask=(true_k<kmax_Mpc_test) & (true_k>0)
            true_p1d=item["p1d_Mpc"][k_mask]#[:self.Nk_test]
            #assert len(true_p1d)==self.Nk_test
            true_k = true_k[k_mask]
            Nk = len(true_k)

            fit_p1d = poly_p1d.PolyP1D(true_k,true_p1d, kmin_Mpc=1.e-3,kmax_Mpc=kmax_Mpc_test,deg=ndeg)
            true_p1d = fit_p1d.P_Mpc(true_k)

            log_KMpc = torch.log10(torch.Tensor(true_k)).to(device)

            # for each entry, figure emulator parameter describing it (labels)

            emu_call={}
            for param in emuparams:
                emu_call[param]=item[param]

            emu_call = {k: emu_call[k] for k in emuparams}
            emu_call = list(emu_call.values())
            emu_call = np.array(emu_call)

            emu_call = (emu_call - paramLims[:,0]) / (paramLims[:,1] - paramLims[:,0]) - 0.5
            emu_call = torch.Tensor(emu_call).unsqueeze(0)


            # ask emulator to emulate P1D (and its uncertainty)
            coeffsPred,coeffs_logerr = emulator(emu_call.to(device))#.cuda()
            coeffs_logerr = torch.clamp(coeffs_logerr,-10,5)
            coeffserr = torch.exp(coeffs_logerr)**2

            
                
            powers = torch.arange(0,ndeg+1,1).cuda()
            emu_p1d = torch.sum(coeffsPred[:,powers,None] * (log_KMpc[None,:] ** powers[None,:,None]), axis=1)


            powers_err = torch.arange(0, ndeg*2+1, 2).cuda()
            emu_p1derr = torch.sqrt(torch.sum(coeffserr[:,powers,None] * (log_KMpc[None,:] ** powers_err[None,:,None]), axis=1))


            emu_p1d = emu_p1d.detach().cpu().numpy().flatten()
            emu_p1derr = emu_p1derr.detach().cpu().numpy().flatten()


            emu_p1ds_transformed[aa]=emu_p1d
            
            emu_p1derr = 10**(emu_p1d)*np.log(10)*emu_p1derr*yscaling
            emu_p1d = 10**(emu_p1d) * yscaling
            

            fractional_error[aa]=emu_p1d/true_p1d

            emu_p1ds[aa]=emu_p1d
            true_p1ds[aa]=true_p1d
            emu_p1derrs[aa]=emu_p1derr
            
            coeffs[aa]=coeffsPred.detach().cpu().numpy()
            coeffs_err[aa]=np.sqrt(coeffserr.detach().cpu().numpy())
            


        print('Mean fractional error:', fractional_error.mean())
        print('Std fractional error:', fractional_error.std())
        
    if return_coeffs==True:
        return fractional_error, emu_p1ds, true_p1ds, emu_p1derrs, emu_p1ds_transformed, coeffs, coeffs_err
    
    else:
        return fractional_error, emu_p1ds, true_p1ds, emu_p1derrs
