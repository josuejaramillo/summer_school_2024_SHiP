import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from numba import jit
import time

# Step 1: Define Symbols
# These symbols represent the variables used in the momentum calculations.
E1, E3, m, m1, m2, m3, thetaV, phiV, kappa = sp.symbols('E1 E3 m m1 m2 m3 thetaV phiV kappa')

# Step 2: Define Helper Functions
# Function to calculate the magnitude of momentum based on energy and mass.
def pPar(En, mn):
    return sp.sqrt(En**2 - mn**2)

# Function to calculate the energy of the second particle.
def E2valEnergies(m, E1, E3):
    return m - E1 - E3

# Function to calculate the angle θ12 based on the energies and masses of the particles.
def Theta12(E1, E3, m, m1, m2, m3): # alpha
    E2 = E2valEnergies(m, E1, E3)
    cosTheta12 = (pPar(E3, m3)**2 - pPar(E1, m1)**2 - pPar(E2, m2)**2) / (2 * pPar(E1, m1) * pPar(E2, m2))
    return sp.acos(cosTheta12)

# Function to calculate the angle θ13 based on the energies and masses of the particles.
def Theta13(E1, E3, m, m1, m2, m3):
    E2 = E2valEnergies(m, E1, E3)
    cosTheta13 = (pPar(E2, m2)**2 - pPar(E1, m1)**2 - pPar(E3, m3)**2) / (2 * pPar(E1, m1) * pPar(E3, m3))
    return sp.acos(cosTheta13)

# Define rotation matrices around the z-axis (φ) and x-axis (θ).
def PhiRotMatrix(phi):
    return sp.Matrix([[sp.cos(phi), sp.sin(phi), 0], 
                      [-sp.sin(phi), sp.cos(phi), 0], 
                      [0, 0, 1]])

def ThetaRotMatrix(theta):
    return sp.Matrix([[1, 0, 0], 
                      [0, sp.cos(theta), sp.sin(theta)], 
                      [0, -sp.sin(theta), sp.cos(theta)]])

# Function to rotate a vector using the rotation matrices.
def pvecRotated(px, py, pz, thetaV, phiV):
    return PhiRotMatrix(phiV) * ThetaRotMatrix(thetaV) * sp.Matrix([px, py, pz])

# Step 3: Define the Rotated Momentum Components
# Rotated components for particle 1
p1_unrot = sp.Matrix([0, 0, pPar(E1, m1)])
p1_rot = pvecRotated(p1_unrot[0], p1_unrot[1], p1_unrot[2], thetaV, phiV)
p1rotatedX = p1_rot[0]
p1rotatedY = p1_rot[1]
p1rotatedZ = p1_rot[2]

# Rotated components for particle 2
theta12_val = Theta12(E1, E3, m, m1, m2, m3)
E2 = E2valEnergies(m, E1, E3)
p2_unrot = sp.Matrix([pPar(E2, m2) * sp.sin(theta12_val) * sp.sin(kappa), 
                      pPar(E2, m2) * sp.sin(theta12_val) * sp.cos(kappa), 
                      pPar(E2, m2) * sp.cos(theta12_val)])
p2_rot = pvecRotated(p2_unrot[0], p2_unrot[1], p2_unrot[2], thetaV, phiV)
p2rotatedX = p2_rot[0]
p2rotatedY = p2_rot[1]
p2rotatedZ = p2_rot[2]

# Rotated components for particle 3
theta13_val = Theta13(E1, E3, m, m1, m2, m3)
p3_unrot = sp.Matrix([-pPar(E3, m3) * sp.sin(theta13_val) * sp.sin(kappa), 
                      -pPar(E3, m3) * sp.sin(theta13_val) * sp.cos(kappa), 
                      pPar(E3, m3) * sp.cos(theta13_val)])
p3_rot = pvecRotated(p3_unrot[0], p3_unrot[1], p3_unrot[2], thetaV, phiV)
p3rotatedX = p3_rot[0]
p3rotatedY = p3_rot[1]
p3rotatedZ = p3_rot[2]

# Step 4: Convert Symbolic Expressions to Numerical Functions Using Lambdify
# Convert the symbolic expressions to numerical functions
p1rotatedX_num = lambdify((E1, m1, thetaV, phiV), p1rotatedX, 'numpy')
p1rotatedY_num = lambdify((E1, m1, thetaV, phiV), p1rotatedY, 'numpy')
p1rotatedZ_num = lambdify((E1, m1, thetaV, phiV), p1rotatedZ, 'numpy')

p2rotatedX_num = lambdify((E1, E3, m, m1, m2, m3, thetaV, phiV, kappa), p2rotatedX, 'numpy')
p2rotatedY_num = lambdify((E1, E3, m, m1, m2, m3, thetaV, phiV, kappa), p2rotatedY, 'numpy')
p2rotatedZ_num = lambdify((E1, E3, m, m1, m2, m3, thetaV, phiV, kappa), p2rotatedZ, 'numpy')

p3rotatedX_num = lambdify((E1, E3, m, m1, m2, m3, thetaV, phiV, kappa), p3rotatedX, 'numpy')
p3rotatedY_num = lambdify((E1, E3, m, m1, m2, m3, thetaV, phiV, kappa), p3rotatedY, 'numpy')
p3rotatedZ_num = lambdify((E1, E3, m, m1, m2, m3, thetaV, phiV, kappa), p3rotatedZ, 'numpy')

# Step 5: JIT Compile the Functions Using Numba
# JIT compile the functions to optimize performance
p1rotatedX_jit = jit(nopython=True)(p1rotatedX_num)
p1rotatedY_jit = jit(nopython=True)(p1rotatedY_num)
p1rotatedZ_jit = jit(nopython=True)(p1rotatedZ_num)

p2rotatedX_jit = jit(nopython=True)(p2rotatedX_num)
p2rotatedY_jit = jit(nopython=True)(p2rotatedY_num)
p2rotatedZ_jit = jit(nopython=True)(p2rotatedZ_num)

p3rotatedX_jit = jit(nopython=True)(p3rotatedX_num)
p3rotatedY_jit = jit(nopython=True)(p3rotatedY_num)
p3rotatedZ_jit = jit(nopython=True)(p3rotatedZ_num)

# # Calculate the rotated X component of p1
# t = time.time()
# for i in range(10):
#     p1x = p1rotatedX_num(np.random.uniform(1,100), 1, np.random.uniform(0.0001, 0.03), np.random.uniform(0.0001, 0.04))
# print(time.time()-t)

# t = time.time()
# for i in range(10):
#     p1x = p1rotatedX_jit(np.random.uniform(1,100), 1, np.random.uniform(0.0001, 0.03), np.random.uniform(0.0001, 0.04))
# print(time.time()-t)