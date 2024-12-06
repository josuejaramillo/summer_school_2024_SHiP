import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sympy as sp

def load_data(paths):
    """
    Loads various datasets related to HNL (Heavy Neutral Lepton) decay, branching ratios, 
    decay widths, total yields, and distributions.

    Returns:
    --------
    tuple
        A tuple containing the loaded data:
        (decayChannels, PDGs, BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames)
    """
    # Unpack the paths for different data files
    (decay_json_path, decay_width_path, yield_e_path, yield_mu_path, yield_tau_path,
     distrHNL_e_path, distrHNL_mu_path, distrHNL_tau_path) = paths

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
    loadedData = (decayChannels, PDGs, BrRatios, Matrix_elements,
                  decayWidthData, yieldData, massDistrData, DistrDataFrames)

    return loadedData

def compute_decay_widths(mass, decayWidthData):
    """
    Computes the decay widths for different mixing patterns (e, mu, tau) at a specified mass.
    """
    decay_mass, DW_e, DW_mu, DW_tau = decayWidthData
    return np.array([
        regular_interpolator(mass, decay_mass, DW_e),
        regular_interpolator(mass, decay_mass, DW_mu),
        regular_interpolator(mass, decay_mass, DW_tau)
    ])

def compute_ctau(mass, decayWidthData, MixingPatternArray):
    """
    Computes the intrinsic lifetime c*tau for HNLs at a specified mass.
    """
    decay_widths = compute_decay_widths(mass, decayWidthData)
    Ue2, Umu2, Utau2 = MixingPatternArray
    total_decay_width = Ue2 * decay_widths[0] + Umu2 * decay_widths[1] + Utau2 * decay_widths[2]
    # c_tau in meters
    c_tau = 1.973269788e-16 / total_decay_width  # Convert from GeV^-1 to meters
    return c_tau

def compute_total_yield(mass, yieldData, MixingPatternArray):
    """
    Computes the total production yield for HNLs at a specified mass.
    """
    interpolated_yields = interpolate_total_yield(mass, yieldData)  # returns [yield_e, yield_mu, yield_tau]
    Ue2, Umu2, Utau2 = MixingPatternArray
    total_yield = Ue2 * interpolated_yields[0] + Umu2 * interpolated_yields[1] + Utau2 * interpolated_yields[2]
    return total_yield

def MatrixElements(expression_strs, decay_widths, MixingPatternArray):
    """
    Compiles matrix element expressions and applies mixing patterns and decay widths.

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
        func_e = None
        func_mu = None
        func_tau = None

        if expr[0] not in [None, "", "-"]:
            expr_e = sp.sympify(expr[0].replace("***", "e"))
            func_e = sp.lambdify((mLLP, E1, E3), expr_e, 'numpy')
        if expr[1] not in [None, "", "-"]:
            expr_mu = sp.sympify(expr[1].replace("***", "e"))
            func_mu = sp.lambdify((mLLP, E1, E3), expr_mu, 'numpy')
        if expr[2] not in [None, "", "-"]:
            expr_tau = sp.sympify(expr[2].replace("***", "e"))
            func_tau = sp.lambdify((mLLP, E1, E3), expr_tau, 'numpy')

        # Create a function that applies the mixing and decay width factors
        def compiled_func(mLLP_val, E1_val, E3_val, func_e=func_e, func_mu=func_mu, func_tau=func_tau):
            # Initialize result
            total_result = np.zeros_like(E1_val)

            # Compute the results for each component if the function exists
            if func_e:
                result_e = Ue2 * DWe * func_e(mLLP_val, E1_val, E3_val)
                total_result += result_e
            if func_mu:
                result_mu = Umu2 * DWmu * func_mu(mLLP_val, E1_val, E3_val)
                total_result += result_mu
            if func_tau:
                result_tau = Utau2 * DWtau * func_tau(mLLP_val, E1_val, E3_val)
                total_result += result_tau

            return total_result

        compiled_expressions.append(compiled_func)

    return compiled_expressions

def regular_interpolator(point, axis, distr):
    """
    Interpolates values for a given point using a regular grid interpolator.
    """
    interpolator = RegularGridInterpolator((axis,), distr, bounds_error=False, fill_value=None)
    if isinstance(point, (int, float)):
        return interpolator([point])[0]
    else:
        return interpolator(point)

def interpolate_total_yield(mass, yield_data):
    """
    Interpolates the total yield for a given mass for different mixing patterns (e, mu, tau).
    """
    yield_mass, yield_e, yield_mu, yield_tau = yield_data
    return np.array([
        regular_interpolator(mass, yield_mass, yield_e),
        regular_interpolator(mass, yield_mass, yield_mu),
        regular_interpolator(mass, yield_mass, yield_tau)
    ])

def BrRatios_interpolator(mass, BrRatios_all):
    """
    Interpolates branching ratios at a specified mass for all mixing patterns (e, mu, tau).
    """
    BrRatios_all = [np.array(BrRatio) for BrRatio in BrRatios_all]
    interpolators = [
        RegularGridInterpolator((BrRatio[:, 0],), BrRatio[:, 1], bounds_error=False, fill_value=0.0)
        for BrRatio in BrRatios_all
    ]
    return np.array([interpolator([mass])[0] for interpolator in interpolators])

def compute_BrMerged(mass, BrRatios, MixingPatternArray, decay_widths):
    """
    Computes the merged branching ratios based on mixing patterns, decay widths, and interpolated branching ratios.
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

    # To avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        BrMerged = np.where(denominator > 0, numerator / denominator, 0.0)

    return BrMerged

def merge_distributions(massDistrData, MixingPatternArray, yieldData, DistrDataFrames):
    """
    Merges distribution data for different mixing patterns (e, mu, tau) into a single scaled distribution.
    """
    Distr_mass, Distr_mass_mu, Distr_mass_tau = massDistrData  # Unique values for each distribution type
    DistrHNL_e, DistrHNL_mu, DistrHNL_tau = DistrDataFrames  # Dataframes for each distribution type

    # Interpolate total production yield for each mass
    interpolated_yields = interpolate_total_yield(Distr_mass, yieldData)
    Ue2, Umu2, Utau2 = MixingPatternArray

    # Apply mixing patterns
    yield_e = Ue2 * interpolated_yields[0]
    yield_mu = Umu2 * interpolated_yields[1]
    yield_tau = Utau2 * interpolated_yields[2]

    # Compute the sum of all yields for each mass
    sum_total_yield = yield_e + yield_mu + yield_tau

    # To avoid division by zero, set sum_total_yield to a small positive value where it's zero
    sum_total_yield_safe = np.where(sum_total_yield > 0, sum_total_yield, 1e-100)

    # Calculate scaled yields for each mixing pattern
    scaled_yield_e = np.divide(yield_e, sum_total_yield_safe, out=np.zeros_like(yield_e), where=sum_total_yield_safe > 0)
    scaled_yield_mu = np.divide(yield_mu, sum_total_yield_safe, out=np.zeros_like(yield_mu), where=sum_total_yield_safe > 0)
    scaled_yield_tau = np.divide(yield_tau, sum_total_yield_safe, out=np.zeros_like(yield_tau), where=sum_total_yield_safe > 0)

    # Create mappings from mass to scaled production yield
    mass_to_yield_e = dict(zip(Distr_mass, scaled_yield_e))
    mass_to_yield_mu = dict(zip(Distr_mass_mu, scaled_yield_mu))
    mass_to_yield_tau = dict(zip(Distr_mass_tau, scaled_yield_tau))

    # Function to scale distribution data based on the production yield
    def scale_by_yield(df, mass_col, yield_map):
        """
        Scales the distribution data based on the production yield.
        """
        # Map mass values to their scaled yields and multiply by the distribution column
        return df[mass_col].map(yield_map) * df[3]

    # Apply scaling for each distribution data frame
    f_e_scaled = scale_by_yield(DistrHNL_e, 0, mass_to_yield_e)
    f_mu_scaled = scale_by_yield(DistrHNL_mu, 0, mass_to_yield_mu)
    f_tau_scaled = scale_by_yield(DistrHNL_tau, 0, mass_to_yield_tau)

    # Sum the scaled results, handling missing values with fill_value=0
    merged_distr = f_e_scaled.add(f_mu_scaled, fill_value=0).add(f_tau_scaled, fill_value=0)

    # Create a new merged DataFrame with the scaled distribution
    Merged_dataframe = DistrHNL_e.copy()
    Merged_dataframe[3] = merged_distr

    return Merged_dataframe

