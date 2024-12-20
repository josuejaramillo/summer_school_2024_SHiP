import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import sympy as sp
import time

class HNLMerging:
    def __init__(self, filesPath, SpecificDecay = None):
        self.prompt_mixing_pattern()
        self.MixingPatternArray = np.array([self.Ue2, self.Umu2, self.Utau2])
        

        # Load data
        self.load_data(filesPath, SpecificDecay)
    
    def load_data(self, filesPath, SpecificDecay):
        #Paths
        decay_json_path =filesPath+"/HNL-decay.json"
        decay_width_path = filesPath+"/HNLdecayWidth.dat"

        yield_e_path = filesPath+"/Total-yield-HNL-e.txt"
        yield_mu_path = filesPath+"/Total-yield-HNL-mu.txt"
        yield_tau_path = filesPath+"/Total-yield-HNL-tau.txt"

        distrHNL_e_path = filesPath+"/DoubleDistrHNL-Mixing-e.txt"
        distrHNL_mu_path = filesPath+"/DoubleDistrHNL-Mixing-mu.txt"
        distrHNL_tau_path = filesPath+"/DoubleDistrHNL-Mixing-tau.txt"

        # Load decay data
        self.HNL_decay = pd.read_json(decay_json_path)

        decayChannel = self.HNL_decay[self.HNL_decay[0] == SpecificDecay]
        PDGs = decayChannel.iloc[:, 1].to_numpy()
        self.PDGcodes = PDGs[0][:3]
        
        if SpecificDecay == None:
            self.channels = np.array(self.HNL_decay.iloc[:, 0])
        else:
            self.channels = [SpecificDecay]
        self.Br_e = np.array(self.HNL_decay.iloc[:, 2])
        self.Br_mu = np.array(self.HNL_decay.iloc[:, 3])
        self.Br_tau = np.array(self.HNL_decay.iloc[:, 4])
        
        # Initialize Matrix_elements
        if SpecificDecay == None:
            self.Matrix_elements = np.column_stack((
                self.HNL_decay.iloc[:, 5], 
                self.HNL_decay.iloc[:, 6], 
                self.HNL_decay.iloc[:, 7]
            ))
        else:
            decayChannel = self.HNL_decay[self.HNL_decay[0] == SpecificDecay]
            self.Matrix_elements = np.column_stack((
                decayChannel.iloc[:, 5], 
                decayChannel.iloc[:, 6], 
                decayChannel.iloc[:, 7]
            ))
        
        # Load decay width data
        self.HNL_decay_width = pd.read_csv(decay_width_path, header=None, sep="\t")
        self.decay_mass = np.array(self.HNL_decay_width.iloc[:, 0])
        self.DW_e = np.array(self.HNL_decay_width.iloc[:, 1])
        self.DW_mu = np.array(self.HNL_decay_width.iloc[:, 2])
        self.DW_tau = np.array(self.HNL_decay_width.iloc[:, 3])

        # Load total yield
        self.HNL_yield_e = pd.read_csv(yield_e_path, header=None, sep="\t")
        self.HNL_yield_mu = pd.read_csv(yield_mu_path, header=None, sep="\t")
        self.HNL_yield_tau = pd.read_csv(yield_tau_path, header=None, sep="\t")

        self.yield_mass = np.array(self.HNL_yield_e.iloc[:, 0])
        self.yield_e = np.array(self.HNL_yield_e.iloc[:, 1])
        self.yield_mu = np.array(self.HNL_yield_mu.iloc[:, 1])
        self.yield_tau = np.array(self.HNL_yield_tau.iloc[:, 1])

        # Load distributions

        self.DistrHNL_e = pd.read_csv(distrHNL_e_path, header=None, sep="\t")
        self.DistrHNL_mu = pd.read_csv(distrHNL_mu_path, header=None, sep="\t")
        self.DistrHNL_tau = pd.read_csv(distrHNL_tau_path, header=None, sep="\t")

        self.Distr_mass = np.unique(self.DistrHNL_e.iloc[:, 0])
        self.Distr_mass_mu = np.unique(self.DistrHNL_mu.iloc[:, 0])
        self.Distr_mass_tau = np.unique(self.DistrHNL_tau.iloc[:, 0])

    def prompt_mixing_pattern(self):
        try:
            self.Ue2 = float(input("\nUe2: "))
            self.Umu2 = float(input("\nUmu2: "))
            self.Utau2 = float(input("\nUtau2: "))

            sumMixingPattern = self.Ue2 + self.Umu2 + self.Utau2
            if sumMixingPattern != 1:
                self.Ue2 = self.Ue2/sumMixingPattern
                self.Umu2 = self.Umu2/sumMixingPattern
                self.Utau2 = self.Utau2/sumMixingPattern

        except ValueError:
            raise ValueError("Invalid input. Please enter numerical values.")
    
    def BrRatios_interpolator(self, mass, BrRatios_all):
        BrRatios_all = [np.array(BrRatio) for BrRatio in BrRatios_all]

        interpolators = [
            RegularGridInterpolator((BrRatio[:, 0],), BrRatio[:, 1])
            for BrRatio in BrRatios_all
        ]
        return np.array([interpolator([mass])[0] for interpolator in interpolators])
    
    def regular_interpolator(self, point, axis, distr):
        interpolator = RegularGridInterpolator((axis,), distr)
        if type(point) == int or type(point) == float:
            return interpolator([point])[0]
        else:
            return interpolator(point)

    def compute_decay_widths(self, mass):
        return np.array([
            self.regular_interpolator(mass, self.decay_mass, self.DW_e),
            self.regular_interpolator(mass, self.decay_mass, self.DW_mu),
            self.regular_interpolator(mass, self.decay_mass, self.DW_tau)
        ])
    
    
    def compute_BrMerged(self, mass):
        # Compute branching ratios
        BrRatios_LLP_HNL = np.column_stack((
            self.BrRatios_interpolator(mass, self.Br_e),
            self.BrRatios_interpolator(mass, self.Br_mu),
            self.BrRatios_interpolator(mass, self.Br_tau)
        ))

        # Compute decay widths
        decay_widths = self.compute_decay_widths(mass)

        # Calculate merged branching ratios
        numerator = (
            self.Ue2 * decay_widths[0] * BrRatios_LLP_HNL[:, 0] +
            self.Umu2 * decay_widths[1] * BrRatios_LLP_HNL[:, 1] +
            self.Utau2 * decay_widths[2] * BrRatios_LLP_HNL[:, 2]
        )
        denominator = np.sum(self.MixingPatternArray * decay_widths)
        BrMerged = numerator / denominator

        return BrMerged

    # def evaluate_matrix_elements(self, expression_str, m, E1_vals, E3_vals):
    #     num_values = len(E1_vals)
    #     evaluated_matrix_elements = np.empty((num_values, 3), dtype=np.float64)

    #     # Define symbols
    #     mLLP, E1, E3 = sp.symbols('mLLP E_1 E_3')

    #     # Precompile expressions
    #     compiled_expressions = [
    #         (
    #             sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[0].replace("***", "e"), evaluate=False), 'numpy'),
    #             sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[1].replace("***", "e"), evaluate=False), 'numpy'),
    #             sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[2].replace("***", "e"), evaluate=False), 'numpy')
    #         )
    #         for expr in expression_str
    #     ]

    #     # Vectorized evaluation
    #     for idx in range(num_values):
    #         E1_val = E1_vals[idx]
    #         E3_val = E3_vals[idx]

    #         for i, (func_e, func_mu, func_tau) in enumerate(compiled_expressions):
    #             evaluated_matrix_elements[idx, 0] = func_e(m, E1_val, E3_val)
    #             evaluated_matrix_elements[idx, 1] = func_mu(m, E1_val, E3_val)
    #             evaluated_matrix_elements[idx, 2] = func_tau(m, E1_val, E3_val)

    #     return evaluated_matrix_elements
    
    # def compute_M2Merged(self, mass, E1_vals, E3_vals, SpecificDecay=None):
    #     # Compute decay widths once
    #     decay_widths = self.compute_decay_widths(mass)
    #     # Evaluate matrix elements
    #     t = time.time()

    #     evaluated_matrix_elements = self.evaluate_matrix_elements(
    #         self.Matrix_elements, mass, E1_vals, E3_vals
    #     )
    #     print(time.time()-t)

    #     # Calculate the merged matrix elements
    #     M2_merged = np.empty((len(E1_vals), len(self.channels)), dtype=np.float64)

    #     # Perform calculations in a vectorized manner
    #     for idx in range(len(E1_vals)):
    #         matrix_elements = evaluated_matrix_elements[idx]
    #         numerator = (
    #             self.Ue2 * decay_widths[0] * matrix_elements[0] +
    #             self.Umu2 * decay_widths[1] * matrix_elements[1] +
    #             self.Utau2 * decay_widths[2] * matrix_elements[2]
    #         )
    #         denominator = np.sum(self.MixingPatternArray * decay_widths) #Not necessary
    #         M2_merged[idx] = numerator / denominator

    #     return M2_merged[:,0]

    def precompute_functions(self, expression_str):
        # Define symbols
        mLLP, E1, E3 = sp.symbols('mLLP E_1 E_3')

        # Precompile expressions
        compiled_expressions = [
            (
                sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[0].replace("***", "e"), evaluate=False), 'numpy'),
                sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[1].replace("***", "e"), evaluate=False), 'numpy'),
                sp.lambdify((mLLP, E1, E3), sp.parse_expr(expr[2].replace("***", "e"), evaluate=False), 'numpy')
            )
            for expr in expression_str
        ]

        return compiled_expressions
    
    def evaluate_matrix_elements(self, m, E1_vals, E3_vals, compiled_expressions):
        num_values = len(E1_vals)
        evaluated_matrix_elements = np.empty((num_values, 3), dtype=np.float64)

        E1_vals = np.asarray(E1_vals)
        E3_vals = np.asarray(E3_vals)
        m = np.asarray(m)

        for i, (func_e, func_mu, func_tau) in enumerate(compiled_expressions):
            evaluated_matrix_elements[:, 0] = func_e(m, E1_vals, E3_vals)
            evaluated_matrix_elements[:, 1] = func_mu(m, E1_vals, E3_vals)
            evaluated_matrix_elements[:, 2] = func_tau(m, E1_vals, E3_vals)

        return evaluated_matrix_elements

    def compute_M2Merged(self, mass, E1_vals, E3_vals, SpecificDecay=None):
        # Compute decay widths once
        decay_widths = self.compute_decay_widths(mass)
        # Evaluate matrix elements

        compiled_expressions = self.precompute_functions(self.Matrix_elements)
        evaluated_matrix_elements = self.evaluate_matrix_elements(
            mass, E1_vals, E3_vals, compiled_expressions
        )

        # Calculate the merged matrix elements
        decay_widths = np.asarray(decay_widths)
        Ue2 = self.Ue2
        Umu2 = self.Umu2
        Utau2 = self.Utau2
        MixingPatternArray = np.asarray(self.MixingPatternArray)

        # Calculate numerator and denominator
        numerator = (
            Ue2 * decay_widths[0] * evaluated_matrix_elements[:, 0] +
            Umu2 * decay_widths[1] * evaluated_matrix_elements[:, 1] +
            Utau2 * decay_widths[2] * evaluated_matrix_elements[:, 2]
        )
        
        denominator = np.sum(MixingPatternArray * decay_widths, axis=0)

        # Avoid division by zero
        # denominator = np.where(denominator == 0, 1, denominator)

        M2_merged = numerator
        return M2_merged

    def interpolate_total_yield(self, mass):
        return np.array([
            self.regular_interpolator(mass, self.yield_mass, self.yield_e),
            self.regular_interpolator(mass, self.yield_mass, self.yield_mu),
            self.regular_interpolator(mass, self.yield_mass, self.yield_tau)
        ])
    
    def total_production_yield(self, mass):
        return np.asarray([self.interpolate_total_yield(mass)[0]*self.MixingPatternArray[0],
                self.interpolate_total_yield(mass)[1]*self.MixingPatternArray[1],
                self.interpolate_total_yield(mass)[2]*self.MixingPatternArray[2]])

    def merge_distributions(self):

        # Calculate total production yield
        total_prod_yield = self.total_production_yield(self.Distr_mass)

        # Unpack the total production yield for different types
        yield_e = total_prod_yield[0]
        yield_mu = total_prod_yield[1]
        yield_tau = total_prod_yield[2]

        # Compute the sum of all yields
        sum_total_yield = [e + mu + tau for e, mu, tau in zip(yield_e, yield_mu, yield_tau)]

        # Calculate scaled yields
        scaled_yield_e = [e / total for e, total in zip(yield_e, sum_total_yield)]
        scaled_yield_mu = [mu / total for mu, total in zip(yield_mu, sum_total_yield)]
        scaled_yield_tau = [tau / total for tau, total in zip(yield_tau, sum_total_yield)]

        # Create mappings from mass to production yield
        mass_to_yield_e = dict(zip(self.Distr_mass, scaled_yield_e))
        mass_to_yield_mu = dict(zip(self.Distr_mass_mu, scaled_yield_mu))
        mass_to_yield_tau = dict(zip(self.Distr_mass_tau, scaled_yield_tau))

        # Combine all mass-to-yield dictionaries
        mass_to_yield = {**mass_to_yield_e, **mass_to_yield_mu, **mass_to_yield_tau}

        # Function to scale values based on the mass-to-yield mapping
        def scale_by_yield(df, mass_col, yield_map):
            return df[mass_col].map(yield_map) * df[3]

        # Apply scaling for each DataFrame
        f_e_scaled = scale_by_yield(self.DistrHNL_e, 0, mass_to_yield)
        f_mu_scaled = scale_by_yield(self.DistrHNL_mu, 0, mass_to_yield)
        f_tau_scaled = scale_by_yield(self.DistrHNL_tau, 0, mass_to_yield)

        # Sum the scaled results, handling missing values with fill_value=0
        merged_distr = f_e_scaled.add(f_mu_scaled, fill_value=0).add(f_tau_scaled, fill_value=0)

        Merged_dataframe = self.DistrHNL_e.copy()
        Merged_dataframe[3] = merged_distr
        Merged_dataframe.to_csv("MergedDistrHNL.txt", index=False, sep="\t", header=None)


#{mass, theta, energy, 
# Sum f_alpha Total_yield_alpha(mass) U^2_alpha / (Sum f_alpha Total_yield_alpha(mass) U^2_alpha)}


if __name__ == "__main__":
#     # Instantiate the class
    analysis = HNLMerging("../Distributions/HNL", "2ev")

#     print(analysis.compute_M2Merged(1, [10], [20], "2ev"))

    # Define arrays of values for mLLP, E1, and E3
    mass = 1
    E1_vals = [50, 400]
    E3_vals = [100, 200]

    # Compute BrMerged using the class method
    # BrMerged = analysis.compute_BrMerged(mass)
    # Compute M2Merged using the class method
    M2Merged = analysis.merge_distributions()
    
    # print(analysis.total_yield(analysis.Distr_mass))
    # analysis.merge_distributions()
    # print(analysis.Distr_mass)