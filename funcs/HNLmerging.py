import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sympy as sp
import time 

def load_data(paths):
    """
    Loads various datasets related to HNL (Heavy Neutral Lepton) decay, branching ratios, 
    decay widths, total yields, and distributions.

    Parameters:
    -----------
    paths : tuple
        Contains the file paths to the decay data, decay width data, yield data, and distribution data.

    Returns:
    --------
    tuple
        A tuple containing the loaded data:
        (decayChannels, PDGs, BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames)
    """
    # Unpack the paths for different data files
    decay_json_path, decay_width_path, yield_e_path, yield_mu_path, yield_tau_path, distrHNL_e_path, distrHNL_mu_path, distrHNL_tau_path = paths

    # Load decay data from JSON file
    HNL_decay = pd.read_json(decay_json_path)

    # Extract decay channels and particle PDGs (Particle Data Group IDs)
    decayChannels = HNL_decay.iloc[:, 0].to_numpy()
    PDGs = HNL_decay.iloc[:, 1].apply(np.array).to_numpy()

    # Extract branching ratios for different mixing patterns (e, mu, tau)
    Br_e = np.array(HNL_decay.iloc[:, 2])
    Br_mu = np.array(HNL_decay.iloc[:, 3])
    Br_tau = np.array(HNL_decay.iloc[:, 4])

    # Combine branching ratios into a single array
    BrRatios = np.array([Br_e, Br_mu, Br_tau])
    
    # Extract matrix elements from the decay data
    Matrix_elements = np.column_stack((
            HNL_decay.iloc[:, 5], 
            HNL_decay.iloc[:, 6], 
            HNL_decay.iloc[:, 7]
        ))
    
    # Load decay width data from a CSV file
    HNL_decay_width = pd.read_csv(decay_width_path, header=None, sep="\t")
    decay_mass = np.array(HNL_decay_width.iloc[:, 0])
    DW_e = np.array(HNL_decay_width.iloc[:, 1])
    DW_mu = np.array(HNL_decay_width.iloc[:, 2])
    DW_tau = np.array(HNL_decay_width.iloc[:, 3])
    
    # Combine decay width data into a single array
    decayWidthData = np.array([decay_mass, DW_e, DW_mu, DW_tau])

    # Load total yield data from CSV files
    HNL_yield_e = pd.read_csv(yield_e_path, header=None, sep="\t")
    HNL_yield_mu = pd.read_csv(yield_mu_path, header=None, sep="\t")
    HNL_yield_tau = pd.read_csv(yield_tau_path, header=None, sep="\t")

    yield_mass = np.array(HNL_yield_e.iloc[:, 0])
    yield_e = np.array(HNL_yield_e.iloc[:, 1])
    yield_mu = np.array(HNL_yield_mu.iloc[:, 1])
    yield_tau = np.array(HNL_yield_tau.iloc[:, 1])

    # Combine yield data into a single array
    yieldData = np.array([yield_mass, yield_e, yield_mu, yield_tau])

    # Load distribution data from CSV files
    DistrHNL_e = pd.read_csv(distrHNL_e_path, header=None, sep="\t")
    DistrHNL_mu = pd.read_csv(distrHNL_mu_path, header=None, sep="\t")
    DistrHNL_tau = pd.read_csv(distrHNL_tau_path, header=None, sep="\t")

    # Combine distribution data frames
    DistrDataFrames = (DistrHNL_e, DistrHNL_mu, DistrHNL_tau)

    # Extract unique mass values for distributions
    Distr_mass = np.unique(DistrHNL_e.iloc[:, 0])
    Distr_mass_mu = np.unique(DistrHNL_mu.iloc[:, 0])
    Distr_mass_tau = np.unique(DistrHNL_tau.iloc[:, 0])

    # Combine mass distribution data into a single array
    massDistrData = np.array([Distr_mass, Distr_mass_mu, Distr_mass_tau])

    # Package all loaded data into a tuple
    loadedData = (decayChannels, PDGs, BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames)

    return loadedData

def regular_interpolator(point, axis, distr):
    """
    Interpolates values for a given point using a regular grid interpolator.

    Parameters:
    -----------
    point : float or list
        The point(s) at which to interpolate the distribution values.
    axis : array-like
        The axis of the grid on which the distribution is defined.
    distr : array-like
        The distribution values corresponding to the grid points.

    Returns:
    --------
    float or np.ndarray
        The interpolated value(s) at the specified point(s).
    """
    interpolator = RegularGridInterpolator((axis,), distr)
    if type(point) == int or type(point) == float:
        return interpolator([point])[0]
    else:
        return interpolator(point)

def compute_decay_widths(mass, decayWidthData):
    """
    Computes the decay widths for different mixing patterns (e, mu, tau) at a specified mass.

    Parameters:
    -----------
    mass : float
        The mass at which to compute the decay widths.
    decayWidthData : np.ndarray
        The decay width data containing mass and width information for e, mu, and tau.

    Returns:
    --------
    np.ndarray
        An array containing the interpolated decay widths for e, mu, and tau at the specified mass.
    """
    decay_mass, DW_e, DW_mu, DW_tau = decayWidthData
    return np.array([
        regular_interpolator(mass, decay_mass, DW_e),
        regular_interpolator(mass, decay_mass, DW_mu),
        regular_interpolator(mass, decay_mass, DW_tau)
    ])

def BrRatios_interpolator(mass, BrRatios_all):
    """
    Interpolates branching ratios at a specified mass for all mixing patterns (e, mu, tau).

    Parameters:
    -----------
    mass : float
        The mass at which to interpolate the branching ratios.
    BrRatios_all : list of np.ndarray
        A list containing branching ratio arrays for e, mu, and tau.

    Returns:
    --------
    np.ndarray
        An array containing the interpolated branching ratios at the specified mass for e, mu, and tau.
    """
    BrRatios_all = [np.array(BrRatio) for BrRatio in BrRatios_all]
    interpolators = [
        RegularGridInterpolator((BrRatio[:, 0],), BrRatio[:, 1])
        for BrRatio in BrRatios_all
    ]
    return np.array([interpolator([mass])[0] for interpolator in interpolators])

def compute_BrMerged(mass, BrRatios, MixingPatternArray, decay_widths):
    """
    Computes the merged branching ratios based on mixing patterns, decay widths, and interpolated branching ratios.

    Parameters:
    -----------
    mass : float
        The mass at which to compute the merged branching ratios.
    BrRatios : np.ndarray
        The branching ratio data for e, mu, and tau.
    MixingPatternArray : np.ndarray
        An array containing the mixing pattern values (Ue2, Umu2, Utau2).
    decay_widths : np.ndarray
        The interpolated decay widths for e, mu, and tau.

    Returns:
    --------
    np.ndarray
        The computed merged branching ratios at the specified mass.
    """
    Ue2, Umu2, Utau2 = MixingPatternArray
    Br_e, Br_mu, Br_tau = BrRatios

    # Compute branching ratios for HNL
    BrRatios_LLP_HNL = np.column_stack((
        BrRatios_interpolator(mass, Br_e),
        BrRatios_interpolator(mass, Br_mu),
        BrRatios_interpolator(mass, Br_tau)
    ))

    # Calculate merged branching ratios using decay widths and mixing patterns
    numerator = (
        Ue2 * decay_widths[0] * BrRatios_LLP_HNL[:, 0] +
        Umu2 * decay_widths[1] * BrRatios_LLP_HNL[:, 1] +
        Utau2 * decay_widths[2] * BrRatios_LLP_HNL[:, 2]
    )
    denominator = np.sum(MixingPatternArray * decay_widths)
    BrMerged = numerator / denominator

    return BrMerged

def MatrixElements(expression_strs, decay_widths, MixingPatternArray):
    """
    Compiles matrix element expressions and applies mixing patterns and decay widths.

    Parameters:
    -----------
    expression_strs : list of str
        List of expressions representing the matrix elements for e, mu, and tau.
    decay_widths : np.ndarray
        The decay widths for e, mu, and tau.
    MixingPatternArray : np.ndarray
        An array containing the mixing pattern values (Ue2, Umu2, Utau2).

    Returns:
    --------
    list
        A list of compiled functions that compute matrix elements given specific values of mLLP, E1, and E3.
    """
    # Define symbols for sympy
    mLLP, E1, E3 = sp.symbols('mLLP E_1 E_3')
    
    # Unpack input arrays
    Ue2, Umu2, Utau2 = MixingPatternArray
    DWe, DWmu, DWtau = decay_widths

    # Compile the expressions into callable functions
    compiled_expressions = []
    for expr in expression_strs:
        # Parse and lambdify each expression for E1 and E3 components
        func_e = sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[0].replace("***", "e")), 'numpy')
        func_mu = sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[1].replace("***", "e")), 'numpy')
        func_tau = sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[2].replace("***", "e")), 'numpy')
        
        # Create a function that applies the mixing and decay width factors
        def compiled_func(mLLP_val, E1_val, E3_val):
            # Convert E1 and E3 to numpy arrays if they aren't already
            E1_val = np.array(E1_val)
            E3_val = np.array(E3_val)
            
            # Compute the results for each component, ensuring element-wise operations
            result_e = Ue2 * DWe * func_e(mLLP_val, E1_val, E3_val)
            result_mu = Umu2 * DWmu * func_mu(mLLP_val, E1_val, E3_val)
            result_tau = Utau2 * DWtau * func_tau(mLLP_val, E1_val, E3_val)

            # Return the total result across all components
            if type(result_e + result_mu + result_tau) == np.float64:
                return np.ones_like(E1_val) * (result_e + result_mu + result_tau)
            else:
                return result_e + result_mu + result_tau

        compiled_expressions.append(compiled_func)

    return compiled_expressions

def interpolate_total_yield(mass, yield_data):
    """
    Interpolates the total yield for a given mass for different mixing patterns (e, mu, tau).

    Parameters:
    -----------
    mass : float
        The mass at which to interpolate the total yield.
    yield_data : np.ndarray
        The yield data containing mass and yield information for e, mu, and tau.

    Returns:
    --------
    np.ndarray
        An array containing the interpolated yields for e, mu, and tau at the specified mass.
    """
    yield_mass, yield_e, yield_mu, yield_tau = yield_data
    return np.array([
        regular_interpolator(mass, yield_mass, yield_e),
        regular_interpolator(mass, yield_mass, yield_mu),
        regular_interpolator(mass, yield_mass, yield_tau)
    ])

def total_production_yield(mass, MixingPatternArray, yield_data):
    """
    Computes the total production yield for a given mass, applying mixing pattern factors.

    Parameters:
    -----------
    mass : float
        The mass at which to compute the total production yield.
    MixingPatternArray : np.ndarray
        An array containing the mixing pattern values (Ue2, Umu2, Utau2).
    yield_data : np.ndarray
        The yield data containing mass and yield information for e, mu, and tau.

    Returns:
    --------
    np.ndarray
        An array containing the total production yields for e, mu, and tau.
    """
    return np.asarray([interpolate_total_yield(mass, yield_data)[0] * MixingPatternArray[0],
                       interpolate_total_yield(mass, yield_data)[1] * MixingPatternArray[1],
                       interpolate_total_yield(mass, yield_data)[2] * MixingPatternArray[2]])

def merge_distributions(massDistrData, MixingPatternArray, yieldData, DistrDataFrames):
    """
    Merges distribution data for different mixing patterns (e, mu, tau) into a single scaled distribution.

    Parameters:
    -----------
    massDistrData : np.ndarray
        Array containing the unique mass values for the distributions (e, mu, tau).
    MixingPatternArray : np.ndarray
        An array containing the mixing pattern values (Ue2, Umu2, Utau2).
    yieldData : np.ndarray
        The yield data containing mass and yield information for e, mu, and tau.
    DistrDataFrames : tuple of pd.DataFrame
        A tuple containing the distribution data frames for e, mu, and tau.

    Returns:
    --------
    pd.DataFrame
        A data frame containing the merged and scaled distribution for all mixing patterns.
    """
    Distr_mass, Distr_mass_mu, Distr_mass_tau = massDistrData  # Unique values for each distribution type
    DistrHNL_e, DistrHNL_mu, DistrHNL_tau = DistrDataFrames  # Dataframes for each distribution type

    # Calculate total production yield for each distribution mass
    total_prod_yield = total_production_yield(Distr_mass, MixingPatternArray, yieldData)

    # Unpack the total production yield for different mixing patterns
    yield_e = total_prod_yield[0]
    yield_mu = total_prod_yield[1]
    yield_tau = total_prod_yield[2]

    # Compute the sum of all yields for each mass
    sum_total_yield = [e + mu + tau for e, mu, tau in zip(yield_e, yield_mu, yield_tau)]

    # Calculate scaled yields for each mixing pattern
    scaled_yield_e = [e / total for e, total in zip(yield_e, sum_total_yield)]
    scaled_yield_mu = [mu / total for mu, total in zip(yield_mu, sum_total_yield)]
    scaled_yield_tau = [tau / total for tau, total in zip(yield_tau, sum_total_yield)]

    # Create mappings from mass to scaled production yield
    mass_to_yield_e = dict(zip(Distr_mass, scaled_yield_e))
    mass_to_yield_mu = dict(zip(Distr_mass_mu, scaled_yield_mu))
    mass_to_yield_tau = dict(zip(Distr_mass_tau, scaled_yield_tau))

    # Combine all mass-to-yield mappings into a single dictionary
    mass_to_yield = {**mass_to_yield_e, **mass_to_yield_mu, **mass_to_yield_tau}

    # Function to scale distribution data based on the production yield
    def scale_by_yield(df, mass_col, yield_map):
        return df[mass_col].map(yield_map) * df[3]

    # Apply scaling for each distribution data frame
    f_e_scaled = scale_by_yield(DistrHNL_e, 0, mass_to_yield)
    f_mu_scaled = scale_by_yield(DistrHNL_mu, 0, mass_to_yield)
    f_tau_scaled = scale_by_yield(DistrHNL_tau, 0, mass_to_yield)

    # Sum the scaled results, handling missing values with fill_value=0
    merged_distr = f_e_scaled.add(f_mu_scaled, fill_value=0).add(f_tau_scaled, fill_value=0)

    # Create a new merged data frame with the scaled distribution
    Merged_dataframe = DistrHNL_e.copy()
    Merged_dataframe[3] = merged_distr

    return Merged_dataframe
