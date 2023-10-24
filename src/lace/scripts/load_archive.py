import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

from lace.emulator import p1d_archive
basedir='lace/emulator/sim_suites/Australia20/'
drop_tau_rescalings=True
archive=p1d_archive.archiveP1D(basedir=basedir,
                    drop_tau_rescalings=drop_tau_rescalings,verbose=True)



