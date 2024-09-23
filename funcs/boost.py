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
    """
    Calculate the velocity vector of the mother particle in the lab frame.

    Args:
        EmotherLab (float): Energy of the mother particle in the lab frame.
        pMotherLab1 (float): x-component of the momentum of the mother particle in the lab frame.
        pMotherLab2 (float): y-component of the momentum of the mother particle in the lab frame.
        pMotherLab3 (float): z-component of the momentum of the mother particle in the lab frame.

    Returns:
        np.ndarray: Velocity vector of the mother particle in the lab frame.
    """
    return np.array([pMotherLab1, pMotherLab2, pMotherLab3]) / EmotherLab

@nb.njit
def gamma_factor(Energy, m):
    """
    Compute the Lorentz factor (gamma) given the energy and mass.

    Args:
        Energy (float): Energy of the particle.
        m (float): Mass of the particle.

    Returns:
        float: Lorentz factor (gamma).
    """
    return Energy / m

@nb.njit
def gamma_factor_mother_lab(EmotherLab, mMother):
    """
    Compute the Lorentz factor in the mother particle's lab frame and the corresponding gamma factor.

    Args:
        EmotherLab (float): Energy of the mother particle in the lab frame.
        mMother (float): Mass of the mother particle.

    Returns:
        float: Gamma factor in the mother particle's lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    v = np.sqrt(1 - 1 / gamma**2)
    return (gamma - 1) / (v**2)

@nb.njit
def pvec_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3, EProdRest, pProdRest1, pProdRest2, pProdRest3):
    """
    Transform the momentum vector of a decay product from the rest frame of the mother particle to the lab frame.

    Args:
        EmotherLab (float): Energy of the mother particle in the lab frame.
        mMother (float): Mass of the mother particle.
        pMotherLab1 (float): x-component of the momentum of the mother particle in the lab frame.
        pMotherLab2 (float): y-component of the momentum of the mother particle in the lab frame.
        pMotherLab3 (float): z-component of the momentum of the mother particle in the lab frame.
        EProdRest (float): Energy of the decay product in the rest frame of the mother particle.
        pProdRest1 (float): x-component of the momentum of the decay product in the rest frame.
        pProdRest2 (float): y-component of the momentum of the decay product in the rest frame.
        pProdRest3 (float): z-component of the momentum of the decay product in the rest frame.

    Returns:
        np.ndarray: Momentum vector of the decay product in the lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    gamma_factor_lab = gamma_factor_mother_lab(EmotherLab, mMother)

    pVecProdRest = np.array([pProdRest1, pProdRest2, pProdRest3])
    vdotp = np.dot(vvec, pVecProdRest)

    return pVecProdRest + gamma * vvec * EProdRest + gamma_factor_lab * vvec * vdotp

@nb.njit
def E_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3, EProdRest, pProdRest1, pProdRest2, pProdRest3):
    """
    Compute the energy of a decay product in the lab frame.

    Args:
        EmotherLab (float): Energy of the mother particle in the lab frame.
        mMother (float): Mass of the mother particle.
        pMotherLab1 (float): x-component of the momentum of the mother particle in the lab frame.
        pMotherLab2 (float): y-component of the momentum of the mother particle in the lab frame.
        pMotherLab3 (float): z-component of the momentum of the mother particle in the lab frame.
        EProdRest (float): Energy of the decay product in the rest frame of the mother particle.
        pProdRest1 (float): x-component of the momentum of the decay product in the rest frame.
        pProdRest2 (float): y-component of the momentum of the decay product in the rest frame.
        pProdRest3 (float): z-component of the momentum of the decay product in the rest frame.

    Returns:
        float: Energy of the decay product in the lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    pVecProdRest = np.array([pProdRest1, pProdRest2, pProdRest3])
    return gamma * (EProdRest + np.dot(vvec, pVecProdRest))

@nb.njit
def boosted_nbody_from_decays_comp(tablemother, tabledaughters):
    """
    Boost the decay products from the rest frame of the mother particle to the lab frame.

    Args:
        tablemother (np.ndarray): Array containing properties of the mother particle [px, py, pz, E, m].
        tabledaughters (np.ndarray): Array containing properties of the decay products.

    Returns:
        np.ndarray: Array of boosted decay products in the lab frame.
    """
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

        boosted_daughters[i] = np.array([daughterpxlab, daughterpylab, daughterpzlab, daughterElab, daughter[indexm1], daughter[indexpdg1], daughter[indexcharge1], daughter[indexstability1]])

    return boosted_daughters

@nb.njit('(float64, float64[:,::1], float64[:,::1],)')
def tab_boosted_decay_products(m, momentum, tabledaughters):
    """
    Compute the boosted decay products for multiple events.

    Args:
        m (float): Mass of the mother particle.
        momentum (np.ndarray): Array of momentum and energy of the mother particle [px, py, pz, E].
        tabledaughters (np.ndarray): Array of decay products for each event.

    Returns:
        np.ndarray: Array of boosted decay products for all events.
    """
    products = np.empty((len(momentum), 24), dtype=np.float64)

    for i in nb.prange(len(momentum)):
        px, py, pz, E = momentum[i]

        # Define the kinematics of the mother particle in the lab frame
        tablemother = np.array([px, py, pz, E, m])
        r = boosted_nbody_from_decays_comp(tablemother, tabledaughters[i])

        # Extract kinematic properties for the two decay products
        products[i] = r.flatten()
    return products

def saveProducts(boostedProducts, LLP_name, mass, MixingPatternArray, c_tau, decayChannels, size_per_channel):
    
    if any(MixingPatternArray) == None:
        outputfileName = f'./outputs/{LLP_name}_{str(mass )}_{c_tau}_decayProducts.dat'
    else:
        outputfileName = f'./outputs/{LLP_name}_{str(mass )}_{round(MixingPatternArray[0], 2)}_{round(MixingPatternArray[1],2)}_{round(MixingPatternArray[2],2)}_{c_tau}_decayProducts.dat'

    with open(outputfileName, 'w') as f:
        start_row = 0  # Initialize the starting row index
        for i, channel in enumerate(decayChannels):
            channel_size = size_per_channel[i]

            # Skip channels with size 0
            if channel_size == 0:
                continue

            end_row = start_row + channel_size
            data = boostedProducts[start_row:end_row, :]  # Extract the relevant rows
            
            # Write the header
            f.write(f"#<process={channel}; sample_points={channel_size}>\n\n")
            
            # Write the data
            for row in data:
                row_str = ' '.join(map(str, row))
                f.write(f"{row_str}\n")
            
            # Add a blank line between channels
            f.write("\n")
            
            # Update the starting row index for the next channel
            start_row = end_row

    # np.savetxt(f'./outputs/{LLP_name}_{str(mass )}_{MixingPatternArray[0]}_{MixingPatternArray[1]}_{MixingPatternArray[2]}_decayProducts.dat', boostedProducts)

