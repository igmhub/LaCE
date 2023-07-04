import numpy as np
from abc import ABC, abstractmethod

class BaseEmulator(ABC):
    """Minimum amount of documentation"""
 
    @abstractmethod
    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False,z=None):
        """Minimum amount of documentation (input, output) """
        pass

