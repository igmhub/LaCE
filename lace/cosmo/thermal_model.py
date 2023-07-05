import numpy as np
import copy
from scipy.interpolate import interp1d
from lace_manager.likelihood import likelihood_parameter
import os

class ThermalModel(object):
    """ Model the redshift evolution of the gas temperature parameters gamma
        and sigT_kms.
        We use a power law rescaling around a fiducial simulation at the centre
        of the initial Latin hypercube in simulation space."""

    def __init__(self,z_T=3.0,ln_sigT_kms_coeff=None,ln_gamma_coeff=None,
                            basedir="/lace/emulator/sim_suites/Australia20/"):
        """ Model the redshift evolution of the thermal broadening scale and gamma.
        We use a power law rescaling around a fiducial simulation at the centre
        of the initial Latin hypercube in simulation space."""

        assert ('LACE_REPO' in os.environ),'export LACE_REPO'
        repo=os.environ['LACE_REPO']

        ## Load fiducial model
        fiducial=np.loadtxt(repo+basedir+"fiducial_igm_evolution.txt")
        self.fid_z=fiducial[0]
        self.fid_gamma=fiducial[2] ## gamma(z)
        self.fid_sigT_kms=fiducial[3] ## sigT_kms(z)
        self.fid_sigT_kms_interp=interp1d(self.fid_z,self.fid_sigT_kms,kind="cubic")
        self.fid_gamma_interp=interp1d(self.fid_z,self.fid_gamma,kind="cubic")

        self.z_T=z_T
        if not ln_sigT_kms_coeff:
            ln_sigT_kms_coeff=[0.0,0.0]
        if not ln_gamma_coeff:
            ln_gamma_coeff=[0.0,0.0]
        self.ln_sigT_kms_coeff=ln_sigT_kms_coeff
        self.ln_gamma_coeff=ln_gamma_coeff
        # store list of likelihood parameters (might be fixed or free)
        self.set_sigT_kms_parameters()
        self.set_gamma_parameters()

    def get_sigT_kms_Nparam(self):
        """Number of parameters in the model of T_0"""
        assert len(self.ln_sigT_kms_coeff)==len(self.sigT_kms_params),"size mismatch"
        return len(self.ln_sigT_kms_coeff)


    def get_gamma_Nparam(self):
        """Number of parameters in the model of gamma"""
        assert len(self.ln_gamma_coeff)==len(self.gamma_params),"size mismatch"
        return len(self.ln_gamma_coeff)


    def power_law_scaling_gamma(self,z):
        """ Power law rescaling around z_T """
        xz=np.log((1+z)/(1+self.z_T))
        ln_poly=np.poly1d(self.ln_gamma_coeff)
        ln_out=ln_poly(xz)
        return np.exp(ln_out)


    def power_law_scaling_sigT_kms(self,z):
        """ Power law rescaling around z_T """
        xz=np.log((1+z)/(1+self.z_T))
        ln_poly=np.poly1d(self.ln_sigT_kms_coeff)
        ln_out=ln_poly(xz)
        return np.exp(ln_out)


    def get_sigT_kms(self,z):
        """sigT_kms at the input redshift"""
        sigT_kms=self.power_law_scaling_sigT_kms(z)*self.fid_sigT_kms_interp(z)
        return sigT_kms


    def get_gamma(self,z):
        """gamma at the input redshift"""
        gamma=self.power_law_scaling_gamma(z)*self.fid_gamma_interp(z)
        return gamma


    def set_sigT_kms_parameters(self):
        """Setup sigT_kms likelihood parameters for thermal model"""

        self.sigT_kms_params=[]
        Npar=len(self.ln_sigT_kms_coeff)
        for i in range(Npar):
            name='ln_sigT_kms_'+str(i)
            if i==0:
                xmin=-0.4
                xmax=0.4
            else:
                xmin=-0.4
                xmax=0.4
            value=self.ln_sigT_kms_coeff[i]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            self.sigT_kms_params.append(par)
        return


    def set_gamma_parameters(self):
        """Setup gamma likelihood parameters for thermal model"""

        self.gamma_params=[]
        Npar=len(self.ln_gamma_coeff)
        for i in range(Npar):
            name='ln_gamma_'+str(i)
            if i==0:
                xmin=-0.2
                xmax=0.2
            else:
                xmin=-0.4
                xmax=0.4
            # note non-trivial order in coefficients
            value=self.ln_gamma_coeff[Npar-i-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            self.gamma_params.append(par)

        return


    def get_sigT_kms_parameters(self):
        """Return sigT_kms likelihood parameters from the thermal model"""
        return self.sigT_kms_params


    def get_gamma_parameters(self):
        """Return gamma likelihood parameters from the thermal model"""
        return self.gamma_params


    def update_parameters(self,like_params):
        """Look for thermal parameters in list of parameters"""

        Npar_sigT_kms=self.get_sigT_kms_Nparam()
        Npar_gamma=self.get_gamma_Nparam()

        # loop over likelihood parameters
        for like_par in like_params:
            if 'sigT_kms' in like_par.name:
                # make sure you find the parameter
                found=False
                # loop over T0 parameters in thermal model
                for ip in range(len(self.sigT_kms_params)):
                    if self.sigT_kms_params[ip].name == like_par.name:
                        assert found==False,'can not update parameter twice'
                        # note different order than with gamma parameters
                        #self.ln_sigT_kms_coeff[Npar_T0-ip-1]=like_par.value
                        self.ln_sigT_kms_coeff[ip]=like_par.value
                        found=True
                assert found==True,'could not update parameter '+like_par.name
            elif 'ln_gamma' in like_par.name:
                # make sure you find the parameter
                found=False
                # loop over gamma parameters in thermal model
                for ip in range(len(self.gamma_params)):
                    if self.gamma_params[ip].name == like_par.name:
                        assert found==False,'can not update parameter twice'
                        self.ln_gamma_coeff[Npar_gamma-ip-1]=like_par.value
                        found=True
                assert found==True,'could not update parameter '+like_par.name

        return


    def get_new_model(self,parameters=[]):
        """Return copy of model, updating values from list of parameters"""

        T = ThermalModel(z_T=self.z_T,
                            ln_sigT_kms_coeff=copy.deepcopy(self.ln_sigT_kms_coeff),
                            ln_gamma_coeff=copy.deepcopy(self.ln_gamma_coeff))
        T.update_parameters(parameters)
        return T


def thermal_broadening_kms(T_0):
    """Thermal broadening RMS in velocity units, given T_0"""

    sigma_T_kms=9.1 * np.sqrt(T_0/1.e4)
    return sigma_T_kms

