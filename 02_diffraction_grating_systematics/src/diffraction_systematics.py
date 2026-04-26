# ==============================================================================
# Title:        Diagnosing Latent Systematic Errors in Diffraction Grating Experiments
# Author:       Dang-Khoa N. Nguyen
# Institution:  HCMC University of Technology and Engineering
# Description:  Rigorous statistical pipeline utilizing vectorized Monte Carlo 
#               uncertainty propagation, Weighted Least Squares (WLS) extraction,
#               and non-parametric LOWESS residual analysis to statistically falsify 
#               geometric zero-point offset hypotheses in diffraction systems.
# ==============================================================================

import os
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# CONFIGURATION & DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Encapsulates physical constants and hyperparameters for the experiment.
    """
    L_meters: float = 1.000
    delta_L_meters: float = 0.005
    delta_x_meters: float = 0.005
    grating_spec_lpm: float = 600.0
    num_simulations: int = 10001000
    cache_filename: str = 'cache_mc_diffraction.npz'

@dataclass
class FitResults:
    """
    Encapsulates the statistical parameters and residuals extracted from WLS regression.
    """
    popt_lin: np.ndarray
    pcov_lin: np.ndarray
    res_lin: np.ndarray
    chi2_lin: float
    popt_nl: np.ndarray
    pcov_nl: np.ndarray
    res_nl: np.ndarray
    chi2_nl: float

# =============================================================================
# AESTHETICS & PLOT CONFIGURATION
# =============================================================================

def configure_journal_style() -> None:
    """
    Configures matplotlib parameters to meet rigorous academic publishing standards.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif=['Computer Modern Roman'], size=11)
    plt.rc('axes', linewidth=1.0, labelsize=12, titlesize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick.major', width=1.0, size=5)
    plt.rc('ytick.major', width=1.0, size=5)
    plt.rc('legend', fontsize=10, frameon=False)
    plt.rc('figure', figsize=(7.0, 5.0))

# =============================================================================
# DATA ACQUISITION & MONTE CARLO PROPAGATION
# =============================================================================

def load_empirical_data() -> pd.DataFrame:
    """Loads experimental spatial and wavelength data into a structured DataFrame."""
    data = {
        'Colour': ['Indigo', 'Blue', 'Green', 'Yellow', 'Red'],
        'x1_m': [0.230, 0.255, 0.295, 0.315, 0.335],
        'x2_m': [0.225, 0.270, 0.300, 0.320, 0.335],
        'lambda_nm': [445, 488, 532, 589, 650]
    }
    df = pd.DataFrame(data)
    df['lambda_m'] = df['lambda_nm'] * 1e-9
    return df

def propagate_uncertainties(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Propagates spatial uncertainty into angular uncertainty via vectorized MC simulation."""
    if os.path.exists(config.cache_filename):
        print(f"[*] Loading cached MC uncertainty propagations from '{config.cache_filename}'...")
        with np.load(config.cache_filename) as data:
            df['sin_theta'] = data['sin_theta_mean']
            df['delta_sin_theta'] = data['sin_theta_std']
        return df

    print(f"[*] Initiating vectorized Monte Carlo propagation (N={config.num_simulations} trials)...")
    start_time = time.time()
    
    x1 = df['x1_m'].to_numpy()
    x2 = df['x2_m'].to_numpy()
    n_points = len(x1)
    
    L_sim = np.random.normal(config.L_meters, config.delta_L_meters, (config.num_simulations, 1))
    x1_sim = np.random.normal(x1, config.delta_x_meters, (config.num_simulations, n_points))
    x2_sim = np.random.normal(x2, config.delta_x_meters, (config.num_simulations, n_points))
    
    theta_sim = np.arctan(((x1_sim + x2_sim) / 2.0) / L_sim)
    sin_theta_sim = np.sin(theta_sim)
    
    df['sin_theta'] = np.mean(sin_theta_sim, axis=0)
    df['delta_sin_theta'] = np.std(sin_theta_sim, axis=0)
    
    np.savez(config.cache_filename, 
             sin_theta_mean=df['sin_theta'].values, 
             sin_theta_std=df['delta_sin_theta'].values)
    
    print(f"[+] MC Simulation completed analytically in {time.time() - start_time:.4f}s.")
    return df

# =============================================================================
# THEORETICAL MODELS & STATISTICAL METRICS
# =============================================================================

def linear_model(wavelength_m: np.ndarray, m: float) -> np.ndarray:
    """Ideal grating equation model (Zeroth-order approximation)."""
    return m * wavelength_m

def make_nonlinear_offset_model(L_meters: float) -> Callable:
    """Closure to generate a non-linear geometric misalignment model."""
    def non_linear_offset_model(wavelength_m: np.ndarray, d: float, delta_x: float) -> np.ndarray:
        theta_ideal = np.arcsin(wavelength_m / d)
        x_true = L_meters * np.tan(theta_ideal)
        theta_meas = np.arctan((x_true + delta_x) / L_meters)
        return np.sin(theta_meas)
    return non_linear_offset_model

def calculate_reduced_chi_squared(y_data: np.ndarray, y_model: np.ndarray, 
                                  y_err: np.ndarray, num_params: int) -> float:
    """Calculates the reduced chi-squared statistic (χ²/ν)."""
    chi2 = np.sum(((y_data - y_model) / y_err) ** 2)
    dof = len(y_data) - num_params
    return chi2 / dof if dof > 0 else np.inf

# =============================================================================
# STATISTICAL EXTRACTION & HYPOTHESIS TESTING
# =============================================================================

def execute_statistical_analysis(df: pd.DataFrame, config: ExperimentConfig) -> FitResults:
    """Performs WLS curve fitting and evaluates competing physical hypotheses."""
    print("[*] Executing Weighted Least Squares (WLS) analytical regression...")
    wavelengths = df['lambda_m'].values
    y_empirical = df['sin_theta'].values
    y_err = df['delta_sin_theta'].values

    # Hypothesis 1: Ideal Linear Model
    popt_lin, pcov_lin = curve_fit(linear_model, wavelengths, y_empirical,
                                   sigma=y_err, absolute_sigma=True)
    y_pred_lin = linear_model(wavelengths, *popt_lin)
    res_lin = y_empirical - y_pred_lin
    chi2_lin = calculate_reduced_chi_squared(y_empirical, y_pred_lin, y_err, num_params=1)
    
    # Hypothesis 2: Geometric Misalignment (Non-Linear Offset)
    bounds = ([1e-6, -0.01], [3e-6, 0.01])  # Physical Constraint: Max 10mm offset
    nl_model = make_nonlinear_offset_model(config.L_meters)
    popt_nl, pcov_nl = curve_fit(nl_model, wavelengths, y_empirical,
                                 p0=[1.0 / popt_lin[0], 0.0], sigma=y_err,
                                 absolute_sigma=True, bounds=bounds)
    y_pred_nl = nl_model(wavelengths, *popt_nl)
    res_nl = y_empirical - y_pred_nl
    chi2_nl = calculate_reduced_chi_squared(y_empirical, y_pred_nl, y_err, num_params=2)

    return FitResults(
        popt_lin=popt_lin, pcov_lin=pcov_lin, res_lin=res_lin, chi2_lin=chi2_lin,
        popt_nl=popt_nl, pcov_nl=pcov_nl, res_nl=res_nl, chi2_nl=chi2_nl
    )

# =============================================================================
# VISUALIZATION & LOGGING
# =============================================================================

def generate_diagnostics(df: pd.DataFrame, stats: FitResults) -> None:
    """Generates publication-quality diagnostic plots stored strictly as PDFs."""
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    print("[*] Generating high-resolution diagnostic PDFs...")

    wavelength_nm = df['lambda_nm'].values
    y_empirical = df['sin_theta'].values
    y_err = df['delta_sin_theta'].values
    lambda_fit = np.linspace(400, 700, 200)

    # =========================================================================
    # --- Plot 1: Linear Fit & Residual Inset (Matched to Target Image) ---
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.5))
    ax1.set_facecolor('#F6F6F6') # Light grey background

    # Ticks & Grid Formatting
    ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, top=True, right=True)
    ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0, top=True, right=True)
    ax1.grid(True, which='major', color='#E0E0E0', linestyle='-', linewidth=1.0)
    ax1.grid(True, which='minor', color='#EBEBEB', linestyle='-', linewidth=0.5)
    ax1.minorticks_on()
    
    # Bold Spines
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Data & Fits
    ax1.plot(lambda_fit, linear_model(lambda_fit * 1e-9, stats.popt_lin[0]), 'r-', 
             linewidth=2.0, label=r'Weighted Linear Fit (Naive Model)')
    ax1.errorbar(wavelength_nm, y_empirical, yerr=y_err, fmt='o',
                 markersize=7, markerfacecolor='#1f77b4', markeredgecolor='k', ecolor='k',
                 elinewidth=1.5, capsize=4, label=r'Experimental Data')
    
    ax1.set_title(r'\textbf{Initial Analysis via Standard Linear Model}', fontsize=18, pad=15)
    ax1.set_xlabel(r'Wavelength, $\lambda$ (nm)', fontsize=15)
    ax1.set_ylabel(r'Sine of Diffraction Angle, $\sin(\theta)$', fontsize=15)
    ax1.set_xlim(400, 700)
    ax1.set_ylim(0.00, 0.37)

    # Statistics Calculations for Textbox
    m_fit = stats.popt_lin[0]
    dm_fit = np.sqrt(stats.pcov_lin[0, 0])
    d_exp = (1.0 / m_fit) * 1e6
    dd_exp = (dm_fit / (m_fit**2)) * 1e6
    ss_res = np.sum(stats.res_lin**2)
    ss_tot = np.sum((y_empirical - np.mean(y_empirical))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    textstr = (
        r'{\Large $\sin(\theta) = m \cdot \lambda$}' + '\n\n'
        rf'$R^2 = {r_squared:.4f}$' + '\n'
        rf'$\chi^2/\nu = {stats.chi2_lin:.2f}$' + '\n'
        rf'$m_{{\mathrm{{fit}}}} = {m_fit:.2f} \pm {dm_fit:.2f} \mathrm{{ \ m}}^{{-1}}$' + '\n'
        rf'$d_{{\mathrm{{exp}}}} = {d_exp:.3f} \pm {dd_exp:.3f} \mathrm{{ \ \mu m}}$'
    )
    props = dict(boxstyle='round,pad=1.0', facecolor='#FDFDFD', edgecolor='#A0A0A0', alpha=0.95)
    ax1.text(0.04, 0.96, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # Inset axes for deep zoom on the Green spectrum point (centered)
    axins = inset_axes(ax1, width="45%", height="35%", loc='center')
    axins.set_facecolor('#F6F6F6')
    for spine in axins.spines.values():
        spine.set_linewidth(1.5)
    
    idx_zoom = 2  # Green point index
    axins.plot(lambda_fit, linear_model(lambda_fit * 1e-9, stats.popt_lin[0]), 'r-', linewidth=1.5)
    axins.errorbar(wavelength_nm[idx_zoom], y_empirical[idx_zoom], yerr=y_err[idx_zoom], 
                   fmt='o', markersize=6, markerfacecolor='#1f77b4', markeredgecolor='k', 
                   ecolor='k', elinewidth=1.5, capsize=4)
    
    xlim_c, ylim_c = wavelength_nm[idx_zoom], y_empirical[idx_zoom]
    axins.set_xlim(xlim_c - 15, xlim_c + 15)
    axins.set_ylim(ylim_c - 4 * y_err[idx_zoom], ylim_c + 4 * y_err[idx_zoom])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(True, which='major', color='#E0E0E0', linestyle='--')
    
    # Drawing diagonal target connectors
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="#808080", lw=1.2)
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower right', frameon=True, facecolor='#FDFDFD', 
               edgecolor='#A0A0A0', borderpad=0.8, fontsize=12)

    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'fig_diffraction_linear_fit.pdf'), dpi=300)
    plt.close(fig1)

    # =========================================================================
    # --- Plot 2: Comparative Residual Analysis (LOWESS) (Unchanged) ---
    # =========================================================================
    fig2, ax2 = plt.subplots()
    ax2.axhline(0, color='k', linestyle='--', linewidth=1.5, label=r'Ideal Model Limit')
    
    ax2.errorbar(wavelength_nm, stats.res_nl, yerr=y_err, fmt='s',
                 markersize=6, markerfacecolor='#2ca02c', markeredgecolor='k', ecolor='k', 
                 elinewidth=1.2, capsize=3, label=r'Residuals ($\mathcal{H}_1$: Offset Correction)')
    ax2.errorbar(wavelength_nm, stats.res_lin, yerr=y_err, fmt='o',
                 markersize=6, markerfacecolor='#d62728', markeredgecolor='k', ecolor='k', 
                 elinewidth=1.2, capsize=3, label=r'Residuals ($\mathcal{H}_0$: Naive Linear)')
    
    lowess_trend = sm.nonparametric.lowess(stats.res_lin, wavelength_nm, frac=0.8)
    ax2.plot(lowess_trend[:, 0], lowess_trend[:, 1], '-.', color='#9467bd',
             linewidth=2.0, label=r'LOWESS Systematic Trend')
    
    ax2.set_xlabel(r'Wavelength, $\lambda$ (nm)')
    ax2.set_ylabel(r'Residuals of $\sin(\theta)$')
    
    y_lim_max = np.max(np.abs(stats.res_lin) + y_err) * 1.5
    ax2.set_ylim(-y_lim_max, y_lim_max)
    ax2.set_xlim(400, 700)
    
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend([handles[i] for i in [3, 2, 1, 0]], [labels[i] for i in [3, 2, 1, 0]], loc='best')
    
    ax2.grid(True, which='both', color='#E0E0E0', linestyle='--', linewidth=0.5)
    ax2.minorticks_on()
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'fig_diffraction_residuals.pdf'), dpi=300)
    plt.close(fig2)
    
    print(f"[+] Output visualizations successfully rendered to '{output_dir}/'.")

def print_academic_summary(stats: FitResults) -> None:
    """Logs the statistical conclusions and epistemological verdict."""
    d_lin = 1.0 / stats.popt_lin[0]
    
    print("\n" + "="*75)
    print(" STATISTICAL INFERENCE & MODEL FALSIFICATION REPORT")
    print("="*75)
    print(r" [1] Null Hypothesis H0 (Naive Linear Model):")
    print(f"     - Extracted Grating Period: {d_lin * 1e6:.4f} μm")
    print(f"     - Reduced Chi-Squared (χ²/ν): {stats.chi2_lin:.2f}")
    
    print(r" [2] Alternate Hypothesis H1 (Geometric Zero-Point Offset):")
    print(f"     - Offset Parameter (Δx): {stats.popt_nl[1] * 1e3:.2f} mm (Hit Boundary Constraint)")
    print(f"     - Reduced Chi-Squared (χ²/ν): {stats.chi2_nl:.2f}")
    
    print("\n [3] Epistemological Verdict:")
    if stats.chi2_nl > stats.chi2_lin:
        print("     => Alternate Hypothesis FALSIFIED.")
        print("        The non-linear zero-point offset model yields a statistically inferior")
        print("        description of the empirical data despite occupying higher degrees of")
        print("        freedom. The persistent arc-like residual structure strongly implies")
        print("        higher-order optical aberrations rather than geometric misalignment.")
    print("="*75 + "\n")

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def main() -> None:
    """Main execution pipeline."""
    configure_journal_style()
    config = ExperimentConfig()
    
    df = load_empirical_data()
    df = propagate_uncertainties(df, config)
    
    fit_results = execute_statistical_analysis(df, config)
    
    generate_diagnostics(df, fit_results)
    print_academic_summary(fit_results)

if __name__ == '__main__':
    main()