# ==============================================================================
# Title:        AlGaInP LED Charge Transport & I-V Characteristic Analysis
# Author:       Dang-Khoa N. Nguyen
# Institution:  HCMC University of Technology and Engineering
# Description:  Extracts non-ideal diode parameters (Is, n, Rs) from experimental 
#               I-V data of an AlGaInP (Red) LED. Leverages derivative-based 
#               initial parameter estimation and a highly parallelized, weighted 
#               Monte Carlo simulation to ascertain robust parameter uncertainties.
# ==============================================================================

import os
import time
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.constants import k, e
from joblib import Parallel, delayed
import multiprocessing
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Numba JIT fallback mechanism
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    logging.info("Numba is available. JIT compilation enabled.")
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not installed. Running in standard Python execution mode.")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ==============================================================================
# PHYSICAL MODELS & CORE COMPUTATION
# ==============================================================================

@njit(cache=True)
def compute_diode_voltage(I: np.ndarray, I_s: float, n: float, R_s: float, V_T: float) -> np.ndarray:
    """
    Computes the theoretical diode voltage given a current array.

    Equation:
        V(I) = n * V_T * ln(I / I_s + 1) + I * R_s

    Args:
        I (np.ndarray): Injected current array (A).
        I_s (float): Reverse saturation current (A).
        n (float): Ideality factor.
        R_s (float): Series resistance (Ohms).
        V_T (float): Thermal voltage (V).

    Returns:
        np.ndarray: Computed theoretical voltage array (V).
    """
    return (n * V_T * np.log((I / I_s) + 1.0)) + (I * R_s)


def make_fit_func(V_T: float):
    """Factory to generate a fit function with a fixed Thermal Voltage."""
    def fit_func(I: np.ndarray, I_s: float, n: float, R_s: float) -> np.ndarray:
        return compute_diode_voltage(I, I_s, n, R_s, V_T)
    return fit_func


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class DiodeExperiment:
    """Encapsulates the experimental dataset, environmental constants, and errors."""
    V: np.ndarray
    I: np.ndarray
    temp_k: float = 300.0
    lsd_V: float = 0.01
    lsd_I: float = 0.01

    def __post_init__(self):
        """Calculates derived constants and measurement uncertainties."""
        self.V_T = (k * self.temp_k) / e
        
        # Uncertainty model: 0.5% of reading + 3 least significant digits
        self.delta_V = (0.005 * self.V) + (3 * self.lsd_V)
        self.delta_I = (0.005 * self.I) + (3 * self.lsd_I)


# ==============================================================================
# ANALYSIS & FITTING PIPELINE
# ==============================================================================

def _mc_worker(seed: int, V: np.ndarray, I: np.ndarray, delta_V: np.ndarray, delta_I: np.ndarray,
               lsd_V: float, initial_guesses: np.ndarray, bounds: Tuple[List[float], List[float]],
               V_T: float) -> np.ndarray:
    """
    Independent worker function for Monte Carlo simulation execution.
    Defined at the top level to ensure compatibility with joblib serialization.
    """
    np.random.seed(seed)
    
    # Generate perturbed dataset
    V_sim = V + delta_V * np.random.randn(len(V))
    I_sim = I + delta_I * np.random.randn(len(I))
    
    # Enforce physical constraints
    I_sim[I_sim <= 1e-9] = 1e-9
    delta_V_sim = (0.005 * V_sim) + (3 * lsd_V)
    
    fit_func = make_fit_func(V_T)
    
    try:
        popt_sim, _ = curve_fit(
            f=fit_func,
            xdata=I_sim,
            ydata=V_sim,
            p0=initial_guesses,
            sigma=delta_V_sim,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=10000
        )
        return popt_sim
    except RuntimeError:
        return np.array([np.nan, np.nan, np.nan])


class MonteCarloFitter:
    """Handles the robust estimation of initial parameters and Monte Carlo error analysis."""
    
    def __init__(self, exp: DiodeExperiment, cache_filename: str = 'cache_mc_algainp_weighted.npz'):
        self.exp = exp
        self.cache_filename = cache_filename
        self.bounds = ([1e-20, 1.0, 1e-4], [1e-8, 4.0, 1.0])
        self.initial_guesses: Optional[np.ndarray] = None
        self.params: Optional[np.ndarray] = None
        self.param_errors: Optional[np.ndarray] = None

    def estimate_initial_guesses(self) -> np.ndarray:
        """
        Derives initial guesses using the derivative method.
        Equation: dV/dI = n * V_T * (1/I) + R_s
        """
        logging.info("Deriving initial parameter guesses via linear derivative fitting.")
        dV = np.diff(self.exp.V)
        dI = np.diff(self.exp.I)
        
        # Prevent division by zero
        dI[dI == 0] = 1e-9
        dVdI = dV / dI
        
        I_mid = (self.exp.I[:-1] + self.exp.I[1:]) / 2.0
        one_over_I = 1.0 / I_mid
        
        # Linear regression on high-current regime to extract n and Rs
        high_current_mask = I_mid > 0.5
        slope, intercept = np.polyfit(one_over_I[high_current_mask], dVdI[high_current_mask], 1)
        
        R_s_guess = intercept
        n_guess = slope / self.exp.V_T
        
        # Reverse calculate I_s using guessed n and R_s
        I_s_values = self.exp.I / (np.exp((self.exp.V - self.exp.I * R_s_guess) / (n_guess * self.exp.V_T)) - 1.0)
        I_s_guess = np.median(I_s_values[I_s_values > 0])
        
        raw_guesses = [I_s_guess, n_guess, R_s_guess]
        self.initial_guesses = np.clip(raw_guesses, self.bounds[0], self.bounds[1])
        
        logging.info(f"Guesses derived: I_s = {self.initial_guesses[0]:.2e} A, "
                     f"n = {self.initial_guesses[1]:.2f}, R_s = {self.initial_guesses[2]:.3f} Ohm")
        return self.initial_guesses

    def run(self, num_simulations: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Executes or loads the Monte Carlo fit routine."""
        if self.initial_guesses is None:
            self.estimate_initial_guesses()

        if os.path.exists(self.cache_filename):
            logging.info(f"Loading cached Monte Carlo results from '{self.cache_filename}'.")
            with np.load(self.cache_filename) as data:
                self.params = data['coeffs']
                self.param_errors = data['se_coeffs']
        else:
            logging.info(f"Executing parallel Monte Carlo simulation ({num_simulations} iterations).")
            num_cores = multiprocessing.cpu_count()
            start_time = time.time()
            
            # Utilizing a distinct seed per thread to ensure stochastic variance
            base_seed = int(time.time()) % 10000
            results = Parallel(n_jobs=num_cores)(
                delayed(_mc_worker)(
                    base_seed + i, self.exp.V, self.exp.I, self.exp.delta_V, self.exp.delta_I,
                    self.exp.lsd_V, self.initial_guesses, self.bounds, self.exp.V_T
                ) for i in range(num_simulations)
            )
            
            logging.info(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
            
            results_array = np.array(results)
            valid_results = results_array[~np.isnan(results_array).any(axis=1)]
            
            if len(valid_results) < num_simulations * 0.8:
                logging.warning(f"High failure rate: Only {len(valid_results)}/{num_simulations} "
                                "fits converged successfully.")
            
            self.params = np.mean(valid_results, axis=0)
            self.param_errors = np.std(valid_results, axis=0)
            
            np.savez(self.cache_filename, coeffs=self.params, se_coeffs=self.param_errors)
            logging.info(f"Results cached to '{self.cache_filename}'.")
            
        return self.params, self.param_errors


# ==============================================================================
# PUBLICATION-READY VISUALIZATION
# ==============================================================================

class PublicationPlotter:
    """Manages aesthetic rendering and high-resolution output of analytical plots."""
    
    @staticmethod
    def configure_rcparams():
        """Applies stringent physical sciences journal formatting standards."""
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.0,
            "figure.dpi": 300
        })

    @staticmethod
    def calculate_r_squared(V_data: np.ndarray, V_fit: np.ndarray) -> float:
        """Calculates the Coefficient of Determination (R^2)."""
        ss_res = np.sum((V_data - V_fit) ** 2)
        ss_tot = np.sum((V_data - np.mean(V_data)) ** 2)
        return 1.0 - (ss_res / ss_tot)

    @classmethod
    def plot_iv_curve(cls, exp: DiodeExperiment, params: np.ndarray, param_errors: np.ndarray, output_file: str):
        """Generates the primary I-V characteristic fit plot with an inset."""
        cls.configure_rcparams()
        
        I_s_fit, n_fit, R_s_fit = params
        delta_Is, delta_n, delta_Rs = param_errors

        # High-resolution theoretical curve
        I_fit_line = np.linspace(1e-3, np.max(exp.I), 500)
        V_fit_line = compute_diode_voltage(I_fit_line, I_s_fit, n_fit, R_s_fit, exp.V_T)
        
        # Calculate standard R^2 metric
        V_fit_data = compute_diode_voltage(exp.I, I_s_fit, n_fit, R_s_fit, exp.V_T)
        R_squared = cls.calculate_r_squared(exp.V, V_fit_data)

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='w')

        # Experimental Data
        ax.errorbar(exp.V, exp.I, yerr=exp.delta_I, xerr=exp.delta_V, fmt='o', markersize=6,
                    markerfacecolor='#0077BE', markeredgecolor='k', ecolor='k',
                    elinewidth=1.2, capsize=3, label=r'Experimental Data')
        
        # Theoretical Fit
        ax.plot(V_fit_line, I_fit_line, 'r-', linewidth=2.5, 
                label=r'Realistic Diode Fit ($R^2 = {:.4f}$)'.format(R_squared))

        ax.set_title(r'\textbf{I-V Characteristic of a Red LED}', pad=15)
        ax.set_xlabel(r'Voltage (V)')
        ax.set_ylabel(r'Current (A)')

        # Parameter Overlay Box
        # param_str = (
        #     r'$V(I) = n V_T \ln\left(\frac{I}{I_s} + 1\right) + I R_s$' + '\n\n'
        #     r'$I_s = ({:.2e} \pm {:.2e})$ A' + '\n'
        #     r'$n = {:.2f} \pm {:.2f}$' + '\n'
        #     r'$R_s = {:.3f} \pm {:.3f}\ \Omega$'
        # ).format(I_s_fit, delta_Is, n_fit, delta_n, R_s_fit, delta_Rs)
        # Parameter Overlay Box (Notice the double curly braces in the LaTeX fraction)
        param_str = (
            r'$V(I) = n V_T \ln\left(\frac{{I}}{{I_s}} + 1\right) + I R_s$' + '\n\n'
            r'$I_s = ({:.2e} \pm {:.2e})$ A' + '\n'
            r'$n = {:.2f} \pm {:.2f}$' + '\n'
            r'$R_s = {:.3f} \pm {:.3f}\ \Omega$'
        ).format(I_s_fit, delta_Is, n_fit, delta_n, R_s_fit, delta_Rs)
        
        ax.text(0.05, 0.95, param_str, transform=ax.transAxes,
                ha='left', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='black', alpha=0.9))

        ax.legend(loc='lower right')
        ax.set_xlim(np.min(exp.V) - 0.05, np.max(exp.V) + 0.05)
        ax.set_ylim(-1, np.max(exp.I) + 1)
        ax.grid(True, which='major', color='#CCCCCC', linestyle='-', alpha=0.7)
        ax.grid(True, which='minor', color='#EEEEEE', linestyle='--', alpha=0.5)
        ax.minorticks_on()

        # Inset Plotting Setup
        axins = ax.inset_axes([0.1, 0.40, 0.35, 0.3])
        target_idx = np.where(exp.I == 9)[0][0]
        x_c, y_c = exp.V[target_idx], exp.I[target_idx]
        dx, dy = exp.delta_V[target_idx], exp.delta_I[target_idx]

        axins.errorbar(x_c, y_c, yerr=dy, xerr=dx, fmt='o', markersize=6,
                       markerfacecolor='#0077BE', markeredgecolor='k', ecolor='k',
                       elinewidth=1.5, capsize=3)
        axins.plot(V_fit_line, I_fit_line, 'r-', linewidth=2)
        axins.set_xlim(x_c - 2*dx, x_c + 2*dx)
        axins.set_ylim(y_c - 5*dy, y_c + 5*dy)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.grid(True)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        logging.info(f"Saved optimal I-V curve plot to {output_file}")
        plt.close(fig)

    @classmethod
    def plot_residuals(cls, exp: DiodeExperiment, params: np.ndarray, output_file: str):
        """Generates the residual diagnostics plot mapped with a nonparametric LOWESS trend."""
        cls.configure_rcparams()
        
        I_s_fit, n_fit, R_s_fit = params
        V_fit_data = compute_diode_voltage(exp.I, I_s_fit, n_fit, R_s_fit, exp.V_T)
        V_residuals = exp.V - V_fit_data

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='w')

        # Generate LOWESS systematic trendline
        lowess_trend = sm.nonparametric.lowess(V_residuals, exp.I, frac=0.5)

        ax.axhline(0, color='r', linestyle='--', linewidth=2, label=r'Zero Ideal Fit')
        ax.plot(lowess_trend[:, 0], lowess_trend[:, 1], '--', color='#9400D3',
                linewidth=2.5, label=r'Systematic Trend (LOWESS)')
        ax.errorbar(exp.I, V_residuals, yerr=exp.delta_V, xerr=exp.delta_I, fmt='o',
                    markersize=6, markerfacecolor='#32CD32', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=3, label=r'Voltage Residuals')

        ax.set_title(r'\textbf{Residual Diagnostic for Red LED Model}', pad=15)
        ax.set_xlabel(r'Current (A)')
        ax.set_ylabel(r'Residual Voltage (V)')

        ax.legend(loc='best')
        ax.set_xlim(-1, np.max(exp.I) + 1)
        y_abs_max = np.max(np.abs(V_residuals) + np.abs(exp.delta_V)) * 1.3
        ax.set_ylim(-y_abs_max, y_abs_max)
        
        ax.grid(True, which='major', color='#CCCCCC', linestyle='-', alpha=0.7)
        ax.grid(True, which='minor', color='#EEEEEE', linestyle='--', alpha=0.5)
        ax.minorticks_on()

        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        logging.info(f"Saved residual analytical plot to {output_file}")
        plt.close(fig)


# ==============================================================================
# MAIN EXECUTION THREAD
# ==============================================================================

def main():
    # 1. Experimental Dataset 
    # (AlGaInP LED Transport Data - Hardcoded for self-contained reproduction)
    voltage_data = np.array([
        1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.77, 1.78, 1.79, 1.79, 1.80, 1.80, 1.80, 
        1.81, 1.81, 1.81, 1.82, 1.82, 1.82, 1.82, 1.85, 1.86, 1.88, 1.89, 1.90, 1.91, 
        1.92, 1.93, 1.94, 1.95, 1.96, 1.97
    ])
    current_data = np.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 
        1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0
    ])

    # 2. Instantiate Analytical Pipeline
    experiment = DiodeExperiment(V=voltage_data, I=current_data, temp_k=300.0)
    fitter = MonteCarloFitter(experiment)
    
    # 3. Execute Fitting
    params, param_errors = fitter.run(num_simulations=10000)
    
    # Print Final Physical Outputs
    logging.info("--- FINAL PARAMETER ESTIMATES ---")
    logging.info(f"Is  = {params[0]:.4e} ± {param_errors[0]:.4e} A")
    logging.info(f"n   = {params[1]:.4f} ± {param_errors[1]:.4f}")
    logging.info(f"Rs  = {params[2]:.4f} ± {param_errors[2]:.4f} Ω")

    # 4. Generate Publication Plots
    PublicationPlotter.plot_iv_curve(
        exp=experiment, 
        params=params, 
        param_errors=param_errors, 
        output_file='fig_algainp_iv_characteristic.pdf'
    )
    PublicationPlotter.plot_residuals(
        exp=experiment, 
        params=params, 
        output_file='fig_algainp_residuals_lowess.pdf'
    )
    
    logging.info("Analysis suite executed successfully. Exiting.")

if __name__ == '__main__':
    # Required for safe cross-platform multiprocessing inside joblib
    multiprocessing.freeze_support()
    main()