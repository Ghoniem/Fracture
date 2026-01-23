"""
Fracture Mechanics Utilities Package
====================================
"""

# Import from Ugenerator
from .Ugenerator.network_gen import CrackNetworkGenerator

# # Import Usolver submodules directly (not through Usolver.__init__)
# from .Usolver.build import *
# from .Usolver.constraints import *
# from .Usolver.KKT import *
# # ... etc for each file that EXISTS

# Or import the modules themselves:
from .Usolver import build, KKT, constraints  # These are module objects

# Import from Uplotter
from .Uplotter import core, opts, plot_PK, reconstruct, smooth

# Import from Uprocessor  
from .Uprocessor import kernels, PK, results, SIF

__version__ = '0.1.0'