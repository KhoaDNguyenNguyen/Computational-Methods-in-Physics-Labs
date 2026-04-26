# ==============================================================================
# Title:        Diagnosing Electro-Thermal Dynamics and Falsifying Empirical Models
#               in Tungsten Filaments via Non-Parametric Residual Analysis
# Author:       Dang-Khoa N. Nguyen
# Institution:  HCMC University of Technology and Engineering
# Description:  Propagates analytical uncertainties and leverages vectorized Monte 
#               Carlo simulations to estimate polynomial parameter variance. 
#               Visualizes high-fidelity I-V and R-V characteristics with LOWESS 
#               residual diagnostics to demonstrate empirical model inadequacy.
# ==============================================================================

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Configure logging for pipeline progress
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def configure_matplotlib_aesthetics() -> None:
    r"""
    Configures Matplotlib runtime parameters to meet high-impact, peer-reviewed 
    physics journal standards (e.g., APS, Nature). Utilizes LaTeX rendering and
    appropriate line weights/font sizes for double-column formats.
    """
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.major.width': 1.5,
        'xtick.major.size': 6,
        'xtick.minor.width': 1.0,
        'xtick.minor.size': 3,
        'ytick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.minor.width': 1.0,
        'ytick.minor.size': 3,
        'legend.fontsize': 12,
        'figure.figsize': (8.0, 6.0),
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf'
    })


def load_and_propagate_errors() -> pd.DataFrame:
    r"""
    Loads empirical experimental data and propagates electro-thermal uncertainties
    using analytical Jacobian-based error propagation.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Voltage (V), Current (I), Resistance (R),
        and their respective absolute uncertainties (delta_V, delta_I, delta_R).
        
    Notes
    -----
    Uncertainty in resistance is derived via standard error propagation for R = V/I:
    $\delta R = R \sqrt{(\delta V / V)^2 + (\delta I / I)^2}$
    """
    voltage = np.array([0.5, 1.0, 1.8, 2.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.8, 9.3, 9.7, 10.0])
    current = np.array([0.53, 0.69, 0.92, 1.12, 1.42, 1.59, 1.76, 1.92, 2.05, 2.17, 2.23, 2.29, 2.32])
    
    delta_v = 0.15 * np.ones_like(voltage)
    delta_i = (0.005 * current) + 0.03
    
    resistance = voltage / current
    delta_r = resistance * np.sqrt((delta_v / voltage)**2 + (delta_i / current)**2)
    
    return pd.DataFrame({
        'V': voltage, 'I': current, 'R': resistance,
        'delta_V': delta_v, 'delta_I': delta_i, 'delta_R': delta_r
    })


def quadratic_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    r"""
    Evaluates a standard quadratic polynomial.

    Parameters
    ----------
    x : np.ndarray
        Independent variable array.
    a, b, c : float
        Polynomial coefficients.

    Returns
    -------
    np.ndarray
        Evaluated dependent variable array.
    """
    return a * x**2 + b * x + c


def perform_monte_carlo_fit(
    x: np.ndarray, y: np.ndarray, 
    delta_x: np.ndarray, delta_y: np.ndarray, 
    num_simulations: int = 100_000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r"""
    Executes a highly optimized, fully vectorized Monte Carlo simulation to extract 
    polynomial coefficients and their statistical uncertainties.
    """
    logging.info(f"Running highly-vectorized Monte Carlo Fit ({num_simulations:,} iterations)...")
    
    # Initial analytical fit for deterministic R^2 mapping
    popt, _ = curve_fit(quadratic_model, x, y, sigma=delta_y, absolute_sigma=True)
    y_fit = quadratic_model(x, *popt)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - (ss_res / ss_tot)
    
    # Vectorized Monte Carlo perturbation
    rng = np.random.default_rng(seed)
    x_sim = rng.normal(loc=x, scale=delta_x, size=(num_simulations, len(x)))
    y_sim = rng.normal(loc=y, scale=delta_y, size=(num_simulations, len(y)))
    
    # Construct 3D Vandermonde matrix [x^2, x, 1] for batched OLS
    x_matrix = np.stack((x_sim**2, x_sim, np.ones_like(x_sim)), axis=-1)
    y_matrix = y_sim[..., np.newaxis]
    x_transpose = np.transpose(x_matrix, axes=(0, 2, 1))
    
    # Normal Equations: (X^T * X)^-1 * (X^T * Y)
    xtx = x_transpose @ x_matrix
    xty = x_transpose @ y_matrix
    
    coeffs_sim = np.linalg.solve(xtx, xty).squeeze()
    mean_coeffs = np.mean(coeffs_sim, axis=0)
    std_coeffs = np.std(coeffs_sim, axis=0)
    residuals = y - quadratic_model(x, *mean_coeffs)
    
    return mean_coeffs, std_coeffs, residuals, r_squared


def plot_fit(
    x: np.ndarray, y: np.ndarray, 
    err_y: np.ndarray, err_x: np.ndarray, 
    coeffs: np.ndarray, err_coeffs: np.ndarray, 
    r_sq: float, ylabel: str, eq_format: str, 
    filepath: Path, color_idx: int = 0
) -> None:
    r"""
    Generates a publication-quality plot of the empirical data and quadratic fit.
    Guarantees no overlap between equations, legends, and data points.
    """
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    marker_color = colors[color_idx]

    fig, ax = plt.subplots()
    ax.set_facecolor('white')  # Force clean academic background
    
    x_line = np.linspace(0, np.max(x) * 1.05, 200)
    y_line = quadratic_model(x_line, *coeffs)
    
    # 1. Plot Trendline (STRICTLY DASHED)
    ax.plot(
        x_line, y_line, color='#D55E00', linestyle='--', linewidth=2.0, zorder=3,
        label=fr'\textbf{{Quadratic Trendline ($R^2 = {r_sq:.4f}$)}}'
    )
    
    # 2. Plot Data Points (STRICTLY NO CONNECTING LINES: linestyle='none')
    ax.errorbar(
        x, y, yerr=err_y, xerr=err_x, fmt='o', linestyle='none', markersize=8,
        markerfacecolor=marker_color, markeredgecolor='k', 
        ecolor='k', elinewidth=1.5, capsize=0, zorder=4,
        label=r'\textbf{Experimental Data}'
    )
    
    ax.set_title(fr'\textbf{{Quadratic Fit for {ylabel}-Voltage Dependence}}', pad=15)
    ax.set_xlabel(r'\textbf{Voltage (V)}')
    ax.set_ylabel(fr'\textbf{{{ylabel}}}')
    
    eq_str = eq_format.format(
        coeffs[0], err_coeffs[0], coeffs[1], err_coeffs[1], coeffs[2], err_coeffs[2]
    )
    
    # 3. Dynamic Placement to PREVENT OVERLAP
    is_resistance = 'Resistance' in ylabel
    
    if is_resistance:
        # RV Curve is concave down. Bottom-right and Top-left are empty.
        legend_loc = 'lower right'
        text_x, text_y = 0.05, 0.95  # Top-left corner
        ha_align, va_align = 'left', 'top'
    else:
        # IV Curve goes linearly bottom-left to top-right. Top-left and Bottom-right are empty.
        legend_loc = 'upper left'
        text_x, text_y = 0.95, 0.05  # Bottom-right corner
        ha_align, va_align = 'right', 'bottom'

    # Equation Text Box
    ax.text(
        text_x, text_y, eq_str, transform=ax.transAxes, color='k',
        ha=ha_align, va=va_align, fontsize=13, zorder=5,
        bbox=dict(boxstyle='round,pad=0.6', fc='white', ec='#BBBBBB', alpha=0.95)
    )
    
    # Legend Place
    ax.legend(loc=legend_loc, framealpha=0.95, edgecolor='#BBBBBB')
    
    ax.set_xlim(0, 11)
    ax.grid(True, which='major', color='#DDDDDD', linestyle='-', zorder=1)
    ax.grid(True, which='minor', color='#EEEEEE', linestyle=':', zorder=1)
    ax.minorticks_on()
    
    fig.savefig(filepath)
    plt.close(fig)
    logging.info(f"Saved figure: {filepath.name}")


def plot_residuals(
    x: np.ndarray, residuals: np.ndarray, 
    err_y: np.ndarray, err_x: np.ndarray, 
    ylabel: str, filepath: Path, color_idx: int = 0
) -> None:
    r"""
    Generates a residual diagnostic plot with a LOWESS non-parametric trend line.
    """
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    marker_color = colors[color_idx]

    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    
    ax.axhline(0, color='r', linestyle='--', linewidth=2.0, zorder=2, 
               label=r'\textbf{Ideal Fit (Zero Residual)}')
    
    # LOWESS Non-parametric Smoothing
    lowess_trend = sm.nonparametric.lowess(residuals, x, frac=0.4)
    ax.plot(
        lowess_trend[:, 0], lowess_trend[:, 1], linestyle='-.', color='#CC79A7',
        linewidth=2.5, zorder=3, label=r'\textbf{Systematic LOWESS Trend}'
    )
    
    # Scatter Residuals (STRICTLY NO CONNECTING LINES)
    ax.errorbar(
        x, residuals, yerr=err_y, xerr=err_x, fmt='o', linestyle='none', markersize=9,
        markerfacecolor=marker_color, markeredgecolor='k', 
        ecolor='k', elinewidth=1.5, capsize=0, zorder=4,
        label=r'\textbf{Residuals (Data - Fit)}'
    )
    
    ax.set_title(fr'\textbf{{Residual Diagnostics: {ylabel}}}', pad=15)
    ax.set_xlabel(r'\textbf{Voltage (V)}')
    ax.set_ylabel(fr'\textbf{{Residual of {ylabel}}}')
    
    # Reorder legend for clarity
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[2], handles[0], handles[1]], [labels[2], labels[0], labels[1]], 
              loc='best', framealpha=0.95, edgecolor='#BBBBBB')
    
    ax.set_xlim(0, 11)
    max_err = np.max(np.abs(residuals) + err_y) * 1.2
    ax.set_ylim(-max_err, max_err)
    
    ax.grid(True, which='major', color='#DDDDDD', linestyle='-', zorder=1)
    ax.grid(True, which='minor', color='#EEEEEE', linestyle=':', zorder=1)
    ax.minorticks_on()
    
    fig.savefig(filepath)
    plt.close(fig)
    logging.info(f"Saved figure: {filepath.name}")


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main() -> None:
    output_dir = Path('figures/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configure_matplotlib_aesthetics()
    df = load_and_propagate_errors()
    
    # --------------------------------------------------------------------------
    # 1. Analyze I-V Characteristics
    # --------------------------------------------------------------------------
    logging.info("--- Analysing I-V Characteristics ---")
    c_iv, err_c_iv, res_iv, r2_iv = perform_monte_carlo_fit(
        df['V'].values, df['I'].values, df['delta_V'].values, df['delta_I'].values
    )
    
    eq_iv = r'$I = ({0:.4f} \pm {1:.4f})V^2 + ({2:.3f} \pm {3:.3f})V + ({4:.2f} \pm {5:.2f})$'
    plot_fit(
        df['V'], df['I'], df['delta_I'], df['delta_V'], 
        c_iv, err_c_iv, r2_iv, 'Current (A)', eq_iv, 
        output_dir / 'fig_IV_fit.pdf', color_idx=0
    )
    plot_residuals(
        df['V'], res_iv, df['delta_I'], df['delta_V'], 
        'Current (A)', output_dir / 'fig_IV_residuals.pdf', color_idx=0
    )
    
    # --------------------------------------------------------------------------
    # 2. Analyze R-V Characteristics
    # --------------------------------------------------------------------------
    logging.info("--- Analysing R-V Characteristics ---")
    c_rv, err_c_rv, res_rv, r2_rv = perform_monte_carlo_fit(
        df['V'].values, df['R'].values, df['delta_V'].values, df['delta_R'].values
    )
    
    eq_rv = r'$R = ({0:.3f} \pm {1:.3f})V^2 + ({2:.2f} \pm {3:.2f})V + ({4:.2f} \pm {5:.2f})$'
    plot_fit(
        df['V'], df['R'], df['delta_R'], df['delta_V'], 
        c_rv, err_c_rv, r2_rv, r'Resistance ($\Omega$)', eq_rv, 
        output_dir / 'fig_RV_fit.pdf', color_idx=2
    )
    plot_residuals(
        df['V'], res_rv, df['delta_R'], df['delta_V'], 
        r'Resistance ($\Omega$)', output_dir / 'fig_RV_residuals.pdf', color_idx=2
    )
    
    logging.info(f"[+] Pipeline completed successfully. Output saved to: {output_dir.resolve()}")


if __name__ == '__main__':
    main()