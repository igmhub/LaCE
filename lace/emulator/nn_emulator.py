import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time 
import sys
import sklearn

# LaCE modules
from lace.emulator import gp_emulator
from lace.emulator import pd_archive
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import poly_p1d
from lace.emulator import utils


import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler

from lace.emulator import nn_architecture
cosmo_fid=camb_cosmo.get_cosmology()
sys.path.append('emulator.py')

class NNEmulator:
    """A class for training an emulator.
    
    Args:
        emuparams (dict): A dictionary of emulator parameters.
        kmax_Mpc (float): The maximum k in Mpc^-1 to use for training. Default is 3.5.
        zmax (float): The maximum redshift to use for training. Default is 4.5.
        Nsim (int): The number of simulations to use for training. Default is 30.
        nepochs (int): The number of epochs to train for. Default is 200.
        postprocessing (str): The post-processing method to use. Default is '3A'.
        model_path (str): The path to a pretrained model. Default is None.
        list_archives (list): A list of archive names to use for training. Default is ['data'].
        drop_sim (float): The simulation to drop during training. Default is None.
        drop_z (float): Drop all snapshpts at redshift z from the training. Default is None.
        pick_z (float): Pick only snapshpts at redshift z. Default is None.
        drop_rescalings (bool): Wheather to drop the optical-depth rescalings or not. Default False.
        train (bool): Wheather to train the emulator or not. Default True. If False, a model path must is required.
    """
    
    def __init__(self, paramList, kmax_Mpc=4, zmax=4.5, ndeg=5, nepochs=100,step_size=75, postprocessing='768', Nsim=30, train=True, list_archives=['data'], initial_weights=True,drop_sim=None, drop_z=None, pick_z=None, save_path=None, drop_rescalings=False, model_path=None):
        
        self.emuparams = paramList
        self.zmax = zmax
        self.kmax_Mpc = kmax_Mpc
        self.nepochs=nepochs
        self.step_size=step_size
        self.postprocessing=postprocessing
        self.model_path=model_path
        self.drop_sim=drop_sim
        self.drop_z=drop_z
        self.pick_z=pick_z
        self.drop_rescalings=drop_rescalings

        self.ndeg=ndeg
       
        self.Nsim = Nsim
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path=save_path
        self.lace_path = utils.ls_level(os.getcwd(), 2)
        self.models_dir = os.path.join(self.lace_path,'/lace/emulator') 
        
        
        self.initial_weights=initial_weights
        
        if initial_weights==True:
             
            if self.kmax_Mpc == 4:
                self.initial_weights_path=os.path.join(self.lace_path,'lace/emulator/initial_params/initial_weights.pt')
            if self.kmax_Mpc == 8:
                self.initial_weights_path=os.path.join(self.lace_path,'lace/emulator/initial_params/initial_weights_extended.pt')
                
        self.key_list=list_archives
        
        if train==True:
            self._train(list_archives)
            
        if train==False:
            
            if self.model_path == None:
                raise Exception("If train==False, model path is required.")
                
            else:
                
                initial_weights = torch.load(os.path.join(self.models_dir,self.model_path), map_location='cpu') 
                self.emulator = nn_architecture.MDNemulator_polyfit(nhidden=5, ndeg=self.ndeg)
                self.emulator.load_state_dict(initial_weights)
                self.emulator.to(self.device)
                print('Model loaded. No training needed')
                
                self.k_Mpc, self.Nz, self.k_bin, kMpc_train = self._obtain_sim_params()

                self.log_KMpc = torch.log10(kMpc_train).to(self.device) 
                self._obtain_paramLims()
                        
                return
            

        if self.save_path != None:
            self._save_emulator()
        

                
    def _sort_dict(self, dct, keys):
        """
        Sort a list of dictionaries based on specified keys.

        Args:
            dct (list): List of dictionaries to be sorted.
            keys (list): List of keys to sort the dictionaries by.

        Returns:
            list: The sorted list of dictionaries.
        """
        for d in dct:
            sorted_d = {k: d[k] for k in keys}  # create a new dictionary with only the specified keys
            d.clear()  # remove all items from the original dictionary
            d.update(sorted_d)  # update the original dictionary with the sorted dictionary
        return dct

    def _obtain_paramLims(self):
        """
        Obtain parameter limits for the emulator training data.

        Returns:
            None
        """
        sim_all = pd_archive.archivePD(z_max=self.zmax)
        sim_all.average_over_samples(flag="all")
        sim_all.average_over_samples(flag="phases")
        sim_all.average_over_samples(flag="axes")

        data = [{key: value for key, value in sim_all.data_av_all[i].items() if key in self.emuparams} for i in range(len(sim_all.data_av_all))]
        data = self._sort_dict(data, self.emuparams)  # sort the data by emulator parameters
        data = [list(data[i].values()) for i in range(len(sim_all.data_av_all))]  
        data = np.array(data) 
 
        paramlims = np.concatenate((data.min(0).reshape(len(data.min(0)), 1), data.max(0).reshape(len(data.max(0)), 1)), 1)
        self.paramLims = paramlims  
        

        training_label = [{key: value for key, value in getattr(sim_all, 'data')[i].items() if key in ['p1d_Mpc']} for i in range(len(getattr(sim_all, 'data')))]
        training_label = [list(training_label[i].values())[0][1:(self.k_bin+1)].tolist() for i in range(len(getattr(sim_all, 'data')))]
        training_label = np.array(training_label)
        self.yscalings = np.median(training_label)     

    def _obtain_sim_params(self):
        """
        Obtain simulation parameters.

        Returns:
            k_Mpc (np.ndarray): Simulation k values.
            Nz (int): Number of redshift values.
            Nk (int): Number of k values.
            k_Mpc_train (tensor): k values in the k training range
        """
        sim = pd_archive.archivePD(z_max=self.zmax, pick_sim=0)
        sim.average_over_samples(flag="all")

        sim_zs = [data['z'] for data in sim.data_av_all]
        Nz = len(sim_zs)
        k_Mpc = sim.data[0]['k_Mpc']
        self.k_Mpc = k_Mpc

        k_mask = (k_Mpc < self.kmax_Mpc) & (k_Mpc > 0)
        k_Mpc_train = k_Mpc[k_mask]
        Nk = len(k_Mpc_train)
        k_Mpc_train = torch.Tensor(k_Mpc_train)
        
        self.Nz = Nz
        #self.k_bin = Nk
        

        return k_Mpc, Nz, Nk, k_Mpc_train

    
    def _get_training_data(self, archive, key_av='None'):
        """
        Given an archive and key_av, it obtains the training data based on self.emuparams
        Sorts the training data according to self.emuparams and scales the data based on self.paramLims
        Finally, it returns the training data as a torch.Tensor object.
        """
        training_data = [{key: value for key, value in getattr(archive, key_av)[i].items() if key in self.emuparams} for i in range(len(getattr(archive, key_av)))]
        training_data = self._sort_dict(training_data, self.emuparams)
        training_data = [list(training_data[i].values()) for i in range(len(getattr(archive, key_av)))]

        training_data = np.array(training_data)
        training_data = (training_data - self.paramLims[:, 0]) / (self.paramLims[:, 1] - self.paramLims[:, 0]) - 0.5
        training_data = torch.Tensor(training_data)

        return training_data

    def _get_training_pd1(self, archive, key_av='None'):
        """
        Given an archive and key_av, it obtains the p1d_Mpc values from the training data and scales it.
        Finally, it returns the scaled values as a torch.Tensor object along with the scaling factor.
        """
        training_label = [{key: value for key, value in getattr(archive, key_av)[i].items() if key in ['p1d_Mpc']} for i in range(len(getattr(archive, key_av)))]
        training_label = [list(training_label[i].values())[0][1:(self.k_bin+1)].tolist() for i in range(len(getattr(archive, key_av)))]

        training_label = np.array(training_label)
        yscalings = np.median(training_label)
        training_label = np.log10(training_label / yscalings)
        training_label = torch.Tensor(training_label)
        
        self.yscalings=yscalings

        return training_label, yscalings
    
    def _drop_redshfit(self, archive):

        for ii in self.key_list:
            instance_data = getattr(archive, ii)

            # Filter the data based on the 'ind_tau' key
            filtered_instance_data = [d for d in instance_data if d['z'] != self.drop_z]

            # Store the filtered data in a dictionary with the instance name as the key
            setattr(archive, ii, filtered_instance_data)

        return archive
    
    def _pick_redshfit(self, archive):

        for ii in self.key_list:
            instance_data = getattr(archive, ii)

            # Filter the data based on the 'ind_tau' key
            filtered_instance_data = [d for d in instance_data if d['z'] == self.pick_z]

            # Store the filtered data in a dictionary with the instance name as the key
            setattr(archive, ii, filtered_instance_data)

        return archive
        

    def _get_archive(self):
        """
        Returns an archive object that contains the P1D data and associated parameters
        based on the postprocessing method and the number of simulations to drop.

        Returns:
        archive (pd_archive object): Archive object containing P1D data and parameters
        """
        if self.postprocessing == '768':
            if self.drop_sim == None:
                archive = pd_archive.archivePD(z_max=self.zmax, nsamples=30)
            else: 
                archive = pd_archive.archivePD(z_max=self.zmax, drop_sim=self.drop_sim, nsamples=30)
                
            
            # Average over samples
            archive.average_over_samples(flag="all")
            archive.average_over_samples(flag="phases")
            archive.average_over_samples(flag="axes")

            # Input emulator
            archive.input_emulator(flag="all")
            archive.input_emulator(flag="phases")
            archive.input_emulator(flag="axes")

        if self.postprocessing == '500':
            if self.drop_sim == None:
                archive = pd_archive.archivePD(post_processing="500")
            else: 
                archive = pd_archive.archivePD(post_processing="500", drop_sim=self.drop_sim)
                
        if self.drop_z!=None:
            archive= self._drop_redshfit(archive)   
            
        if self.pick_z!=None:
            archive= self._pick_redshfit(archive)  
            
        if self.drop_rescalings==True:
            archive.data = [d for d in archive.data if d['scale_tau'] == 1] 
            archive.data_av_all = [d for d in archive.data_av_all if d['scale_tau'] == 1] 
            archive.data_av_phases = [d for d in archive.data_av_phases if d['scale_tau'] == 1] 
            archive.data_av_axes = [d for d in archive.data_av_axes if d['scale_tau'] == 1] 
            archive.data_input_axes = [d for d in archive.data_input_axes if d['scale_tau'] == 1] 
            archive.data_input_phases = [d for d in archive.data_input_phases if d['scale_tau'] == 1] 
        
            
        return archive


            

    def _train(self, key_list):

        """
        Trains the emulator with given key_list using the archive data.
        Args:
        key_list (list): List of keys to be used for training

        Returns:None
        """
        
        print('start the training of the emulator')
        self.k_Mpc, self.Nz, self.k_bin, kMpc_train = self._obtain_sim_params()
        self._obtain_paramLims()

        training = self._get_archive()
    
        log_KMpc_train = torch.log10(kMpc_train).to(self.device)  
        
        self.log_KMpc=log_KMpc_train

        self.emulator = nn_architecture.MDNemulator_polyfit(nhidden=5, ndeg=self.ndeg)
        if self.initial_weights==True:
            initial_weights = torch.load(self.initial_weights_path, map_location='cpu')
            self.emulator.load_state_dict(initial_weights)
            

        optimizer = optim.Adam(self.emulator.parameters(), lr=1e-3, weight_decay=1e-4)#
        scheduler = lr_scheduler.StepLR(optimizer, self.step_size, gamma=0.1)      

        
        training_data = torch.Tensor()
        training_label = torch.Tensor()
        
        for key in key_list:
                
            training_data_prime = self._get_training_data(training, key_av = key)
            training_label_prime, yscalings = self._get_training_pd1(training, key_av = key)
                
            training_data = torch.concat((training_data,training_data_prime),0)                   
            training_label = torch.concat((training_label,training_label_prime),0)
                
        print('Training network on %s'%len(training_data))
            
              
        trainig_dataset = TensorDataset(training_data,training_label)
        loader_train = DataLoader(trainig_dataset, batch_size=100, shuffle = True)
            

        if self.model_path != None:
            initial_weights = torch.load(self.model_path, map_location='cpu')    
            self.emulator.load_state_dict(initial_weights)
            
        self.emulator.to(self.device)

        t0 = time.time()
        for epoch in range(self.nepochs):
            for datain, p1D_true in loader_train:
                optimizer.zero_grad() 
            
       
                coeffsPred,coeffs_logerr = self.emulator(datain.to(self.device))#
                coeffs_logerr = torch.clamp(coeffs_logerr,-10,5)
                coeffserr = torch.exp(coeffs_logerr)**2
                
                
                powers = torch.arange(0,self.ndeg+1,1).cuda()
                P1Dpred = torch.sum(coeffsPred[:,powers,None] * (log_KMpc_train[None,:] ** powers[None,:,None]), axis=1)
                
                
                powers_err = torch.arange(0, self.ndeg*2+1, 2).cuda()
                P1Derr = torch.sqrt(torch.sum(coeffserr[:,powers,None] * (log_KMpc_train[None,:] ** powers_err[None,:,None]), axis=1))

                P1Dlogerr = torch.log(P1Derr)
            
                      
                log_prob = ((P1Dpred - p1D_true.to(self.device)) / P1Derr).pow(2) + 2*P1Dlogerr#
            
                loss = torch.nansum(log_prob,1)
                loss = torch.nanmean(loss,0)
            
                loss.backward()
                optimizer.step()

            scheduler.step()
        print(f'Emualtor trained in {time.time() - t0} seconds')
            
    def _save_emulator(self):
        if self.drop_sim!=None:
            torch.save(self.emulator.state_dict(), os.path.join(self.save_path,f'emulator_{self.drop_sim}.pt'))
        else:
            torch.save(self.emulator.state_dict(), self.save_path)
