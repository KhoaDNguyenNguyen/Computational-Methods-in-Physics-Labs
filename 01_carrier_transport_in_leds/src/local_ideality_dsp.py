# ==============================================================================
# Title:        Local Ideality Factor Analysis via Digital Signal Processing
# Author:       Dang-Khoa N. Nguyen
# Institution:  HCMC University of Technology and Engineering
# Description:  Computes the local ideality factor n(I) of an AlGaInP (Red) LED. 
#               Applies a Savitzky-Golay filter to smooth experimental voltage 
#               data prior to numerical differentiation, thereby mitigating noise 
#               amplification in the derivative. 
# ==============================================================================

import logging
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, e
from scipy.signal import savgol_filter

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class DiodeExperiment:
    """Encapsulates the experimental dataset and environmental constants."""
    V: np.ndarray
    I: np.ndarray
    temp_k: float = 300.0

    def __post_init__(self):
        """Calculates derived environmental constants."""
        self.V_T = (k * self.temp_k) / e


# ==============================================================================
# ANALYSIS PIPELINE
# ==============================================================================

class LocalIdealityAnalyzer:
    """Handles the smoothing and numerical differentiation of the I-V data."""
    
    def __init__(self, exp: DiodeExperiment):
        self.exp = exp

    def calculate_ideality(self, window_length: int = 7, polyorder: int = 2, 
                           rs_est: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the local ideality factor from smoothed V(I) data.
        
        Equation:
            n(I) = (I / V_T) * (dV/dI - R_s)

        Args:
            window_length (int): Savitzky-Golay filter window size.
            polyorder (int): Savitzky-Golay filter polynomial order.
            rs_est (float): Estimated series resistance (Ohms).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays for midpoint current (I_mid) and n(I).
        """
        logging.info(f"Applying Savitzky-Golay filter (window={window_length}, poly={polyorder}).")
        V_smooth = savgol_filter(self.exp.V, window_length=window_length, polyorder=polyorder)
        
        logging.info(f"Calculating local ideality factor n(I) with estimated Rs = {rs_est} Ohms.")
        dV_smooth = np.diff(V_smooth)
        dI = np.diff(self.exp.I)
        
        # Prevent division by zero mathematically
        dI[dI == 0] = 1e-12
        dVdI_smooth = dV_smooth / dI
        
        # The derivative mathematically corresponds to the midpoint of the intervals
        I_mid = (self.exp.I[:-1] + self.exp.I[1:]) / 2.0
        
        n_local_smooth = (I_mid / self.exp.V_T) * (dVdI_smooth - rs_est)
        
        return I_mid, n_local_smooth


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

    @classmethod
    def plot_local_ideality(cls, I_mid: np.ndarray, n_local: np.ndarray, output_file: str):
        """Generates the local ideality factor analytical plot."""
        cls.configure_rcparams()
        
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='w')

        # Data Plot
        ax.plot(np.log(I_mid), n_local, 'o-', markersize=7, markerfacecolor='#0077BE', 
                markeredgecolor='k', linewidth=2.0, label=r'Local Ideality Factor (Smoothed)')

        # Theoretical Thresholds
        ax.axhline(1.0, color='r', linestyle='--', linewidth=2.0, label=r'$n=1$ (Ideal Diffusion)')
        ax.axhline(2.0, color='g', linestyle='--', linewidth=2.0, label=r'$n=2$ (Recombination)')

        ax.set_title(r'\textbf{Analysis of Local Ideality Factor for Red LED}', pad=15)
        ax.set_xlabel(r'Logarithm of Current, $\ln(I)$')
        ax.set_ylabel(r'Ideality Factor, $n(I)$')
        
        ax.legend(loc='best')
        ax.grid(True, which='major', color='#CCCCCC', linestyle='-', alpha=0.7)
        ax.grid(True, which='minor', color='#EEEEEE', linestyle='--', alpha=0.5)
        ax.minorticks_on()
        
        # Optimize Y-axis limit for physical readability
        ax.set_ylim(bottom=0, top=max(2.5, np.max(n_local) * 1.1))

        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        logging.info(f"Saved local ideality analytical plot to '{output_file}'")
        plt.close(fig)


# ==============================================================================
# MAIN EXECUTION THREAD
# ==============================================================================

def main():
    # 1. Experimental Dataset (AlGaInP LED)
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
    analyzer = LocalIdealityAnalyzer(exp=experiment)
    
    # 3. Execute Analysis (using estimated Rs = 0.01 Ohms from the original script)
    I_mid, n_local = analyzer.calculate_ideality(window_length=7, polyorder=2, rs_est=0.01)

    # 4. Generate Publication Plot
    output_filename = 'fig_algainp_local_ideality.pdf'
    PublicationPlotter.plot_local_ideality(
        I_mid=I_mid, 
        n_local=n_local, 
        output_file=output_filename
    )
    
    logging.info("Analysis executed successfully. Exiting.")

if __name__ == '__main__':
    main()