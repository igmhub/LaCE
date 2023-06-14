import sys
 from lace.emulator import gp_emulator
 sys.path.append('emulator.py')
 from lace.emulator import nn_emulator


 class P1Demulator:
     def __init__(self, emu_algorithm='NN', paramList=['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc'], zmax=4.5, kmax_Mpc=4, ndeg=5, Nsim=30, train=True, emu_type='polyfit', list_archives=['data_input_axes', 'data_input_phases'], postprocessing='768', drop_rescalings=False, drop_sim=None, drop_z=None, passarchive=None, model_path=None, save_path=None):
         """
         Initializes the P1Demulator class.

         Args:
             emu_algorithm (str): The emulator algorithm to use. Possibilities are Gaussian Process 'GP' and Neural Network (NN). Default is 'NN'.
             paramList (list): List of parameters. Default is ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc'].
             zmax (float): Maximum redshift. Default is 4.5.
             kmax_Mpc (float): Maximum wavenumber in Mpc. Default is 4.
             ndeg (int): Degree of polynomial for polyfit emulator. Default is 5.
             Nsim (int): Number of simulations. Default is 30.
             train (bool): Flag indicating whether to train the emulator. Default is True.
             emu_type (str): Type of emulator. Possibilities are 'k_bin' and 'polyfit'. Default is 'polyfit'.
             list_archives (list): List of archives. Default is ['data_input_axes', 'data_input_phases'].
             postprocessing (str): Post-processing option. Old post-processing along the line of sight: "500". \n
             New post processing alogn the three axes: "768"Default is '768'.
             drop_rescalings (bool): Flag indicating whether to drop rescalings. Default is False.\n
             Setting it to True is recommended only for GP.
             drop_sim (int): Simulations number from the hypercube to drop from the training sample.
             drop_z (float): Drop all snapshots at redshift z.
             passarchive: (class) Archive to train the emulator. Only for the GP.
             model_path: (str) Path to a pretrained model. Default is None. Only for NN
             save_path: (str) Path to store the model. Default is None. Only for NN
         """

         self.emu_algorithm = emu_algorithm
         self.paramList = paramList


         if postprocessing == '500':
             # If postprocessing is '500', print a warning message and modify list_archives
             print('Warning: With the one-axis version of the post-processing, the only archive allowed is "data".\nModified to continue running')
             list_archives = ['data']

         if (postprocessing != '500')&(postprocessing != '768'):
             raise ValueError('Available post-processings are "500" and "768"')


         if self.emu_algorithm == 'NN':
             # If emu_algorithm is 'NN', create an instance of NNEmulator and assign it to the emulator attribute
             self.emulator = nn_emulator.NNEmulator(
                 paramList=paramList,
                 kmax_Mpc=4,
                 zmax=4.5,
                 ndeg=5,
                 nepochs=100,
                 step_size=75,
                 train=True,
                 postprocessing='768',
                 model_path=model_path,
                 Nsim=30,
                 list_archives=list_archives,
                 drop_sim=None,
                 drop_z=None,
                 pick_z=None,
                 save_path=save_path,
                 drop_rescalings=drop_rescalings
             )


         if self.emu_algorithm == 'GP':
             # If emu_algorithm is 'GP', create an instance of GPEmulator and assign it to the emulator attribute
             self.emulator = gp_emulator.GPEmulator(
                 z_max=zmax,
                 kmax_Mpc=kmax_Mpc,
                 ndeg=ndeg,
                 train=True,
                 asymmetric_kernel=True,
                 rbf_only=True,
                 passarchive=None,
                 emu_type=emu_type,
                 check_hull=False,
                 key_data=list_archives[0]
             ) 

