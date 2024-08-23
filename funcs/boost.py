import numpy as np
import numba as nb

# Constants for table indices
indexpx1 = 0
indexpy1 = 1
indexpz1 = 2
indexE1 = 3
indexm1 = 4
indexpdg1 = 5
indexcharge1 = 6
indexstability1 = 7
LengthDataProduct = indexstability1 + 1

@nb.njit
def vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3):
    return np.array([pMotherLab1, pMotherLab2, pMotherLab3]) / EmotherLab

@nb.njit
def gamma_factor(Energy, m):
    return Energy / m

@nb.njit
def gamma_factor_mother_lab(EmotherLab, mMother):
    gamma = gamma_factor(EmotherLab, mMother)
    v = np.sqrt(1 - 1 / gamma**2)
    return (gamma - 1) / (v**2)

@nb.njit
def pvec_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3, EProdRest, pProdRest1, pProdRest2, pProdRest3):
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    gamma_factor_lab = gamma_factor_mother_lab(EmotherLab, mMother)

    pVecProdRest = np.array([pProdRest1, pProdRest2, pProdRest3])
    vdotp = np.dot(vvec, pVecProdRest)

    return pVecProdRest + gamma * vvec * EProdRest + gamma_factor_lab * vvec * vdotp

@nb.njit
def E_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3, EProdRest, pProdRest1, pProdRest2, pProdRest3):
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    pVecProdRest = np.array([pProdRest1, pProdRest2, pProdRest3])
    return gamma * (EProdRest + np.dot(vvec, pVecProdRest))

@nb.njit
def boosted_nbody_from_decays_comp(tablemother, tabledaughters):
    mmother = tablemother[indexm1]
    motherE = tablemother[indexE1]
    motherpx = tablemother[indexpx1]
    motherpy = tablemother[indexpy1]
    motherpz = tablemother[indexpz1]

    num_daughters = len(tabledaughters) // LengthDataProduct
    boosted_daughters = np.zeros((num_daughters, LengthDataProduct))

    for i in range(num_daughters):
        start_index = i * LengthDataProduct
        end_index = start_index + LengthDataProduct
        daughter = tabledaughters[start_index:end_index]

        daughterErest = daughter[indexE1]
        daughterpxrest = daughter[indexpx1]
        daughterpyrest = daughter[indexpy1]
        daughterpzrest = daughter[indexpz1]

        daughterElab = E_prod_lab(motherE, mmother, motherpx, motherpy, motherpz, daughterErest, daughterpxrest, daughterpyrest, daughterpzrest)
        daughterpxlab, daughterpylab, daughterpzlab = pvec_prod_lab(motherE, mmother, motherpx, motherpy, motherpz, daughterErest, daughterpxrest, daughterpyrest, daughterpzrest)

        boosted_daughters[i] = np.array([daughterpxlab, daughterpylab, daughterpzlab, daughterElab, mmother, daughter[indexpdg1], daughter[indexcharge1], daughter[indexstability1]])

    return boosted_daughters

@nb.njit
def tab_boosted_decay_products(tablemother, tabledaughters):
    return boosted_nbody_from_decays_comp(tablemother, tabledaughters)


# # Example input data
# # Table of mother particle
# # Format: [px, py, pz, E, mass, pdg, charge, stability]
# tablemother = np.array([
#     0.0, 0.0, 1.0, 2.0,  # px, py, pz, E
#     1.0,  # mass
#     11,   # pdg code (e.g., electron)
#     -1,   # charge
#     1     # stability
# ])

# # Table of daughter particles
# # Format: [px, py, pz, E, mass, pdg, charge, stability]
# # Flat array with 8 entries per daughter particle
# # Example with 2 daughter particles
# tabledaughters = np.array([
#     # First daughter
#     0.1, 0.2, 0.3, 0.5,  # px, py, pz, E
#     0.5,  # mass
#     22,   # pdg code (e.g., photon)
#     0,    # charge
#     1,    # stability
#     # Second daughter
#     0.4, 0.1, 0.2, 0.7,  # px, py, pz, E
#     0.5,  # mass
#     13,   # pdg code (e.g., muon)
#     -1,   # charge
#     1,    # stability
#     # Third daughter
#     -0.3, -0.4, 0.5, 0.8,  # px, py, pz, E
#     0.5,  # mass
#     211,  # pdg code (e.g., pion)
#     1,    # charge
#     1     # stability
# ])


# # Import the functions (assuming they are in a module named `decay_module`)
# # from decay_module import tab_boosted_decay_products

# # Perform the boost operation
# boosted_results = tab_boosted_decay_products(tablemother, tabledaughters)
# print(boosted_results.flatten())
# # Print results
# # print("Boosted 4-momenta of daughter particles in the lab frame:")
# # for i, result in enumerate(boosted_results):
# #     print(f"Daughter Particle {i+1}:")
# #     print(f"  px = {result[indexpx1]}")
# #     print(f"  py = {result[indexpy1]}")
# #     print(f"  pz = {result[indexpz1]}")
# #     print(f"  E  = {result[indexE1]}")
# #     print(f"  Mass = {result[indexm1]}")
# #     print(f"  PDG = {result[indexpdg1]}")
# #     print(f"  Charge = {result[indexcharge1]}")
# #     print(f"  Stability = {result[indexstability1]}")
# #     print()
