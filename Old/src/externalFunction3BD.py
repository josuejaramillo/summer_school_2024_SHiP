import pandas as pd
import numpy as np
import sympy as sp
from src import PDG
from src import HNLmerging

def ParamProductMass(pdg_id):
    return PDG.get_mass(pdg_id)

def ParamProductCharge(pdg_id):
    return PDG.get_charge(pdg_id)

def ParamProductStability(pdg_id):
    return PDG.get_stability(pdg_id)

def PDGcodes(path, SpecificDecay):
    HNL_decay = pd.read_json(path)
    decayChannel = HNL_decay[HNL_decay[0] == SpecificDecay]
    PDGs = decayChannel.iloc[:, 1].to_numpy()
    return PDGs[0][:3]
    
def Msquared3BodyLLP(analysis, SpecificDecay, E1, E3, MASSM):
    return analysis.compute_M2Merged(MASSM, E1, E3, SpecificDecay)



