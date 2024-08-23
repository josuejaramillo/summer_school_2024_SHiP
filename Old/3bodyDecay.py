import numpy as np
from numba import jit, njit, prange
from numpy.random import random, uniform, choice
from src.externalFunction3BD import *
from src import rotateVectors
from src import  HNLmerging
import time

@njit
def E2valEnergies(m, E1, E3):
    return m - E1 - E3

@njit
def block_random_energies_old(m, m1, m2, m3):
    while True:
        E3r = np.random.uniform(m3, (m**2 + m3**2 - (m1 + m2)**2) / (2 * m))
        E1r = np.random.uniform(m1, (m**2 + m1**2 - (m2 + m3)**2) / (2 * m))
        E2v = m - E1r - E3r
        if E2v > m2 and (E2v**2 - m2**2 - (E1r**2 - m1**2) - (E3r**2 - m3**2))**2 < 4 * (E1r**2 - m1**2) * (E3r**2 - m3**2):
            break
    return np.array([E1r, E3r])

@njit
def block_random_energies_old1(m, m1, m2, m3, Nevents):
    result = np.empty((Nevents, 2))  # Preallocate array with the right shape
    for i in prange(Nevents):
        result[i] = block_random_energies_old(m, m1, m2, m3)
    return result

def weights_non_uniform_comp(tabe1e3, MASSM, MASS1, MASS2, MASS3, distr):
    e1 = np.array(tabe1e3[:,0])
    e3 = np.array(tabe1e3[:,1])
    
    # Assuming distr can be applied in a vectorized manner
    ME = distr(e1, e3)
    return ME

def block_random_energies(m, m1, m2, m3, Nevents, distr):
    # Table of energies E_1, E_3 assuming unit matrix element.
    tabE1E3unweighted = block_random_energies_old1(m, m1, m2, m3, Nevents)
    
    # Calculating the squared matrix element for these energies.
    weights1 = np.abs(weights_non_uniform_comp(tabE1E3unweighted, m, m1, m2, m3, distr))
    weights1 = np.where(weights1 < 0, 0, weights1)  # Ensure no negative weights
    tabsel_indeces = choice(len(tabE1E3unweighted), size=Nevents, p=weights1/weights1.sum())
    return tabE1E3unweighted[tabsel_indeces]

@njit
def tabPS3bodyCompiled(tabPSenergies, MASSM, MASS1, MASS2, MASS3, pdg1, pdg2, pdg3, charge1, charge2, charge3, stability1, stability2, stability3):
    # Random energies of particles 1, 3 in 3-body decay.
    eprod1 = tabPSenergies[0]
    eprod3 = tabPSenergies[1]
    # Random angles defining the orientations of the 3-momenta.

    thetaRand = np.arccos(uniform(-1, 1))
    phiRand = uniform(-np.pi, np.pi)
    kappaRand = uniform(-np.pi, np.pi)

    eprod2 = MASSM - eprod1 - eprod3


    pxprod1 = rotateVectors.p1rotatedX_jit(eprod1, MASS1, thetaRand, phiRand)
    pyprod1 = rotateVectors.p1rotatedY_jit(eprod1, MASS1, thetaRand, phiRand)
    pzprod1 = rotateVectors.p1rotatedZ_jit(eprod1, MASS1, thetaRand, phiRand)

    pxprod2 = rotateVectors.p2rotatedX_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pyprod2 = rotateVectors.p2rotatedY_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pzprod2 = rotateVectors.p2rotatedZ_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)

    pxprod3 = rotateVectors.p3rotatedX_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pyprod3 = rotateVectors.p3rotatedY_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pzprod3 = rotateVectors.p3rotatedZ_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)

    return np.array([pxprod1, pyprod1, pzprod1, eprod1, MASS1, pdg1, charge1, stability1,
                     pxprod2, pyprod2, pzprod2, eprod2, MASS2, pdg2, charge2, stability2,
                     pxprod3, pyprod3, pzprod3, eprod3, MASS3, pdg3, charge3, stability3])

def three_body_decays_events_at_rest(particle, SpecificDecay, mparticle, Nevents, external_funcs):
    ParamProductMass, PDGcodes, ParamProductCharge, ParamProductStability, Msquared3BodyLLP, analysis = external_funcs

    MASSM = mparticle

    pdg1, pdg2, pdg3 = analysis.PDGcodes


    MASS1, MASS2, MASS3 = [ParamProductMass(p) for p in [pdg1, pdg2, pdg3]]
    charge1, charge2, charge3 = [ParamProductCharge(p) for p in [pdg1, pdg2, pdg3]]
    stability1, stability2, stability3 = [ParamProductStability(p) for p in [pdg1, pdg2, pdg3]]

    def distr(E1, E3):
        return Msquared3BodyLLP(analysis, SpecificDecay, E1, E3, MASSM)
    
    
    tabE1E3true = block_random_energies(MASSM, MASS1, MASS2, MASS3, Nevents, distr)

    # t = time.time()
    result  = np.array([tabPS3bodyCompiled(e, MASSM, MASS1, MASS2, MASS3, pdg1, pdg2, pdg3, charge1, charge2, charge3, stability1, stability2, stability3) for e in tabE1E3true])
    # print(time.time()-t)
    return result

analysis_2ev = HNLmerging.HNLMerging("./Distributions/HNL", "2ev")

external_funcs_2ev = (ParamProductMass, PDGcodes, ParamProductCharge,
                   ParamProductStability, Msquared3BodyLLP, analysis_2ev)


analysis_2muv = HNLmerging.HNLMerging("./Distributions/HNL", "2muv")

external_funcs_2muv = (ParamProductMass, PDGcodes, ParamProductCharge,
                   ParamProductStability, Msquared3BodyLLP, analysis_2muv)

# t = time.time()
# result = three_body_decays_events_at_rest("HNL", "2muv", 0.5, 10**3, external_funcs)
# print("full code (1)", time.time()-t)

# print("\n")

t = time.time()
result = three_body_decays_events_at_rest("HNL", "2ev", 0.5, 1, external_funcs_2ev)
print("First run (compile)", time.time()-t)

t = time.time()
result = three_body_decays_events_at_rest("HNL", "2ev", 0.5, 10**5, external_funcs_2ev)
print("full code (1)", time.time()-t)

t = time.time()
result = three_body_decays_events_at_rest("HNL", "2muv", 0.5, 10**5, external_funcs_2muv)
print("full code (2)", time.time()-t)

# print(result)
