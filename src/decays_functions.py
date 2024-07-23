import numpy as np
import numba as nb

"""
Indices referring to the meaning of the final table columns.
"""
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
def pvec_mother_lab(pMotherLab1, pMotherLab2, pMotherLab3):
    """
    Return the 3-momentum vector of the mother particle in the lab frame.

    Parameters:
    pMotherLab1 (float): x-component of the mother particle's momentum in the lab frame.
    pMotherLab2 (float): y-component of the mother particle's momentum in the lab frame.
    pMotherLab3 (float): z-component of the mother particle's momentum in the lab frame.

    Returns:
    numpy.ndarray: 3-momentum vector of the mother particle.
    """
    return np.array([pMotherLab1, pMotherLab2, pMotherLab3])

@nb.njit
def vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3):
    """
    Return the velocity vector of the mother particle in the lab frame.

    Parameters:
    EmotherLab (float): Energy of the mother particle in the lab frame.
    pMotherLab1 (float): x-component of the mother particle's momentum in the lab frame.
    pMotherLab2 (float): y-component of the mother particle's momentum in the lab frame.
    pMotherLab3 (float): z-component of the mother particle's momentum in the lab frame.

    Returns:
    numpy.ndarray: Velocity vector of the mother particle.
    """
    return pvec_mother_lab(pMotherLab1, pMotherLab2, pMotherLab3) / EmotherLab

@nb.njit
def gamma_factor(Energy, m):
    """
    Calculate the Lorentz gamma factor.

    Parameters:
    Energy (float): Energy of the particle.
    m (float): Mass of the particle.

    Returns:
    float: Lorentz gamma factor.
    """
    return Energy / m

@nb.njit
def gamma_factor_mother_lab(EmotherLab, mMother):
    """
    Calculate the gamma factor for the mother particle in the lab frame.

    Parameters:
    EmotherLab (float): Energy of the mother particle in the lab frame.
    mMother (float): Mass of the mother particle.

    Returns:
    float: Adjusted gamma factor for the mother particle.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    v = np.sqrt(1 - 1 / gamma**2)
    return (gamma - 1) / (v**2)

@nb.njit
def pvec_prod_rest(pProdRest1, pProdRest2, pProdRest3):
    """
    Return the 3-momentum vector of the decay product in the rest frame.

    Parameters:
    pProdRest1 (float): x-component of the decay product's momentum in the rest frame.
    pProdRest2 (float): y-component of the decay product's momentum in the rest frame.
    pProdRest3 (float): z-component of the decay product's momentum in the rest frame.

    Returns:
    numpy.ndarray: 3-momentum vector of the decay product.
    """
    return np.array([pProdRest1, pProdRest2, pProdRest3])

@nb.njit
def pvec_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3, EProdRest, pProdRest1, pProdRest2, pProdRest3):
    """
    Calculate the 3-momentum of the decay product in the lab frame.

    Parameters:
    EmotherLab (float): Energy of the mother particle in the lab frame.
    mMother (float): Mass of the mother particle.
    pMotherLab1 (float): x-component of the mother particle's momentum in the lab frame.
    pMotherLab2 (float): y-component of the mother particle's momentum in the lab frame.
    pMotherLab3 (float): z-component of the mother particle's momentum in the lab frame.
    EProdRest (float): Energy of the decay product in the rest frame.
    pProdRest1 (float): x-component of the decay product's momentum in the rest frame.
    pProdRest2 (float): y-component of the decay product's momentum in the rest frame.
    pProdRest3 (float): z-component of the decay product's momentum in the rest frame.

    Returns:
    numpy.ndarray: 3-momentum vector of the decay product in the lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    gamma_factor_lab = gamma_factor_mother_lab(EmotherLab, mMother)

    pVecProdRest = pvec_prod_rest(pProdRest1, pProdRest2, pProdRest3)
    vdotp = np.dot(vvec, pVecProdRest)

    return pVecProdRest + gamma * vvec * EProdRest + gamma_factor_lab * vvec * vdotp

@nb.njit
def E_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3, EProdRest, pProdRest1, pProdRest2, pProdRest3):
    """
    Calculate the energy of the decay product in the lab frame.

    Parameters:
    EmotherLab (float): Energy of the mother particle in the lab frame.
    mMother (float): Mass of the mother particle.
    pMotherLab1 (float): x-component of the mother particle's momentum in the lab frame.
    pMotherLab2 (float): y-component of the mother particle's momentum in the lab frame.
    pMotherLab3 (float): z-component of the mother particle's momentum in the lab frame.
    EProdRest (float): Energy of the decay product in the rest frame.
    pProdRest1 (float): x-component of the decay product's momentum in the rest frame.
    pProdRest2 (float): y-component of the decay product's momentum in the rest frame.
    pProdRest3 (float): z-component of the decay product's momentum in the rest frame.

    Returns:
    float: Energy of the decay product in the lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    pVecProdRest = pvec_prod_rest(pProdRest1, pProdRest2, pProdRest3)
    return gamma * (EProdRest + np.dot(vvec, pVecProdRest))

@nb.njit
def n_vector_particles(thetaVals, phiVals, E1, E2, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2):
    """
    Generate the phase space of the decay at the rest frame.

    Parameters:
    thetaVals (float): Polar angles defining the directions of momenta of decay products.
    phiVals (float): Azimuthal angles defining the directions of momenta of decay products.
    E1 (float): Energy of the first decay product.
    E2 (float): Energy of the second decay product.
    m1 (float): Mass of the first decay product.
    m2 (float): Mass of the second decay product.
    pdg1 (int): PDG code of the first decay product.
    pdg2 (int): PDG code of the second decay product.
    charge1 (int): Charge of the first decay product.
    charge2 (int): Charge of the second decay product.
    stability1 (int): Stability of the first decay product.
    stability2 (int): Stability of the second decay product.

    Returns:
    numpy.ndarray: Phase space of the decay products at rest frame.
    """
    pmod = np.sqrt(E1**2 - m1**2)
    px1 = pmod * np.sin(thetaVals) * np.cos(phiVals)
    px2 = -px1
    py1 = pmod * np.sin(thetaVals) * np.sin(phiVals)
    py2 = -py1
    pz1 = pmod * np.cos(thetaVals)
    pz2 = -pz1
    return np.array([[px1, py1, pz1, E1, m1, pdg1, charge1, stability1], [px2, py2, pz2, E2, m2, pdg2, charge2, stability2]])

@nb.njit
def boosted_llps_from_decays_comp(tablemother, tabledaughter):
    """
    Boost the rest-frame 4-momenta of decay products to the lab frame.

    Parameters:
    tablemother (numpy.ndarray): Array containing the kinematics of the mother particle.
    tabledaughter (numpy.ndarray): Array containing the kinematics of the daughter particle.

    Returns:
    numpy.ndarray: Boosted 4-momenta of the daughter particle in the lab frame.
    """
    # Extracting the mother kinematics from mother table
    mmother = tablemother[indexm1]
    motherE = tablemother[indexE1]
    motherpx = tablemother[indexpx1]
    motherpy = tablemother[indexpy1]
    motherpz = tablemother[indexpz1]

    # Extracting the daughter's kinematics at mother's rest and parameters from daughter table
    daughterErest = tabledaughter[indexE1]
    daughterpxrest = tabledaughter[indexpx1]
    daughterpyrest = tabledaughter[indexpy1]
    daughterpzrest = tabledaughter[indexpz1]
    daughtercharge = tabledaughter[indexcharge1]
    daughterpdg = tabledaughter[indexpdg1]
    daughterstability = tabledaughter[indexstability1]

    # Boosting daughter's rest 4-momenta to mother's lab frame
    daughterElab = E_prod_lab(motherE, mmother, motherpx, motherpy, motherpz, daughterErest, daughterpxrest, daughterpyrest, daughterpzrest)
    daughterpxlab, daughterpylab, daughterpzlab = pvec_prod_lab(motherE, mmother, motherpx, motherpy, motherpz, daughterErest, daughterpxrest, daughterpyrest, daughterpzrest)

    return np.array([daughterpxlab, daughterpylab, daughterpzlab, daughterElab, mmother, daughterpdg, daughtercharge, daughterstability])

@nb.njit
def tab_boosted_decay_products(tablemother, tabledaughters):
    """
    Generate boosted 4-momenta for a list of decay products.

    Parameters:
    tablemother (numpy.ndarray): Array containing the kinematics of the mother particle.
    tabledaughters (numpy.ndarray): Array containing the kinematics of the daughter particles.

    Returns:
    numpy.ndarray: Array of boosted 4-momenta of the daughter particles in the lab frame.
    """
    numberdaughters = tabledaughters.shape[0]
    result = np.zeros((numberdaughters, LengthDataProduct))
    for i in range(numberdaughters):
        daughter = tabledaughters[i]
        result[i] = boosted_llps_from_decays_comp(tablemother, daughter)
    return result

@nb.njit
def simulate_and_boost(m, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2, tablemother):
    """
    Simulate the decay of a particle and boost the decay products to the lab frame.

    Parameters:
    m (float): Mass of the mother particle.
    m1 (float): Mass of the first decay product.
    m2 (float): Mass of the second decay product.
    pdg1 (int): PDG code of the first decay product.
    pdg2 (int): PDG code of the second decay product.
    charge1 (int): Charge of the first decay product.
    charge2 (int): Charge of the second decay product.
    stability1 (int): Stability of the first decay product.
    stability2 (int): Stability of the second decay product.
    tablemother (numpy.ndarray): Array containing the kinematics of the mother particle.

    Returns:
    numpy.ndarray: Array of boosted 4-momenta of the decay products in the lab frame.
    """
    # Random polar and azimuthal angles defining the directions of momenta of decay products at rest frame
    # thetaVals = np.random.uniform(0, np.pi)
    # phiVals = np.random.uniform(0, 2 * np.pi)
    thetaVals = np.random.rand() * np.pi
    phiVals = np.random.rand() * 2 * np.pi
    E1 = (m**2 + m1**2 - m2**2) / (2 * m)
    E2 = (m**2 + m2**2 - m1**2) / (2 * m)
    tabledaughters = n_vector_particles(thetaVals, phiVals, E1, E2, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2)

    # Boost to the lab frame
    boosted_data = tab_boosted_decay_products(tablemother, tabledaughters)

    return boosted_data

#('(float64, float64[:,::1], float64, float64, float64, float64, float64, float64, float64, float64)', parallel=True)
@nb.njit
def decay_products(m, momentum, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2):
        products = np.empty((len(momentum), 16), dtype=np.float64)
        for i in nb.prange(len(momentum)):
            px, py, pz, E = momentum[i]

            # Define the kinematics of the mother particle in the lab frame
            tablemother = np.array([px, py, pz, E, m])
            r = simulate_and_boost(m, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2, tablemother)

            px1, py1, pz1, E1, mmother, pdg2, charge1, stability1 = r[0]
            px2, py2, pz2, E2, mmother, pdg2, charge2, stability2 = r[1]
            products[i] = np.asarray([px1, py1, pz1, E1, m1, pdg1, charge1, stability1, px2, py2, pz2, E2, m2, pdg2, charge2, stability2])

        return products