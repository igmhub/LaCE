import os
import numpy as np
from lace.cosmo import camb_cosmo, cosmology


class RescaledCosmology(object):
    """
    Given a fiducial cosmology, make predictions for other cosmologies
    that do not modify the background expansion.
    """

    def __init__(self, fid_cosmo, new_params_dict=None):

        # make sure that you are not modifying the background
        assert fid_cosmo.same_background(new_params_dict), "background not fixed"

        self.fid_cosmo = fid_cosmo
        self.new_params = new_params_dict

        return 


    def dkms_dMpc(self, z):
        return self.fid_cosmo.dkms_dMpc(z)

    def dkms_dhMpc(self, z):
        return self.fid_cosmo.dkms_dhMpc(z)

    def dAA_dMpc(self, z, lambda_AA):
        return self.fid_cosmo.dAA_dMpc(z)

    def get_linP_hMpc(self, zs, kmax_Mpc=None):
        fid_linP_hMpc = self.fid_cosmo.get_linP_hMpc(zs, kmax_Mpc)
        raise ValueError('implement rescaling')

    def get_linP_Mpc(self, zs, kmax_Mpc=None):
        fid_linP_Mpc = self.fid_cosmo.get_linP_Mpc(zs, kmax_Mpc)
        raise ValueError('implement rescaling')

    def get_linP_kms(self, zs, kmax_Mpc=None):
        fid_linP_kms = self.fid_cosmo.get_linP_kms(zs, kmax_Mpc)
        raise ValueError('implement rescaling')


