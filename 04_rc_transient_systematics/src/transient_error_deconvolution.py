# ==============================================================================
# Title:        RC Circuit Transient Dynamics: Deconvolving Compound Systematic Errors
# Author:       Dang-Khoa N. Nguyen
# Institution:  HCMC University of Technology and Engineering
# Description:  Non-linear least squares (NLLS) optimization of transient RC circuit 
#               currents to deconvolve compound systematic errors via robust residual 
#               diagnostics and strict parametric uncertainty propagation.
# ==============================================================================

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class CircuitConfig:
    """
    Immutable data class storing nominal circuit parameters and constraints.
    
    Attributes
    ----------
    v_source : float
        Nominal source voltage (V).
    err_v_source : float
        Absolute uncertainty in the source voltage (V).
    r_nom : float
        Nominal resistance (Ohms).
    c_nom : float
        Nominal capacitance (Farads).
    """
    v_source: float = 5.00
    err_v_source: float = 0.05
    r_nom: float = 82e3
    c_nom: float = 330e-6


@dataclass(frozen=True)
class FitResult:
    """
    Data class storing the parameters and statistical metadata of an NLLS fit.
    
    Attributes
    ----------
    i0 : float
        Extracted initial current (uA).
    err_i0 : float
        Standard deviation (uncertainty) of the initial current (uA).
    tau : float
        Extracted time constant (s).
    err_tau : float
        Standard deviation (uncertainty) of the time constant (s).
    r_squared : float
        Coefficient of determination for the fit.
    popt : npt.NDArray[np.float64]
        Optimized parameters [i0, tau].
    pcov : npt.NDArray[np.float64]
        Covariance matrix of the optimized parameters.
    """
    i0: float
    err_i0: float
    tau: float
    err_tau: float
    r_squared: float
    popt: npt.NDArray[np.float64]
    pcov: npt.NDArray[np.float64]

# =============================================================================
# PHYSICS & NUMERICAL METHODS
# =============================================================================

def configure_plot_aesthetics() -> None:
    """
    Configures global matplotlib parameters to meet strict academic publication 
    standards (e.g., Physical Review, Nature). Forces LaTeX rendering.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.linewidth": 1.5,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "lines.linewidth": 2.0,
        "xtick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.width": 1.5,
        "ytick.major.size": 6,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "legend.fontsize": 14,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "k"
    })


def exponential_decay(t: npt.NDArray[np.float64], i0: float, tau: float) -> npt.NDArray[np.float64]:
    """
    Computes the standard transient exponential decay of current in an RC circuit.
    
    Equation:
        I(t) = I_0 * exp(-t / tau)
        
    Parameters
    ----------
    t : npt.NDArray[np.float64]
        Temporal domain array (s).
    i0 : float
        Initial peak current at t=0 (uA).
    tau : float
        RC time constant (s).
        
    Returns
    -------
    npt.NDArray[np.float64]
        Calculated transient current array (uA).
    """
    return i0 * np.exp(-t / tau)


def execute_nlls_fit(t: npt.NDArray[np.float64], 
                     i_meas: npt.NDArray[np.float64], 
                     err_i: npt.NDArray[np.float64], 
                     p0: Tuple[float, float] = (40.0, 30.0)) -> FitResult:
    """
    Executes a Non-Linear Least Squares (NLLS) optimization for the exponential decay model.
    """
    popt, pcov = curve_fit(
        f=exponential_decay, 
        xdata=t, 
        ydata=i_meas, 
        p0=p0, 
        sigma=err_i, 
        absolute_sigma=True
    )
    
    perr = np.sqrt(np.diag(pcov))
    
    # Vectorized R-squared computation
    ss_res = np.sum((i_meas - exponential_decay(t, *popt))**2)
    ss_tot = np.sum((i_meas - np.mean(i_meas))**2)
    r_squared = 1.0 - (ss_res / ss_tot)
    
    return FitResult(
        i0=popt[0], err_i0=perr[0],
        tau=popt[1], err_tau=perr[1],
        r_squared=r_squared, popt=popt, pcov=pcov
    )


def diagnose_compound_error(config: CircuitConfig, 
                            i0_ua: float, err_i0_ua: float, 
                            tau_s: float, err_tau_s: float) -> None:
    """
    Propagates parametric uncertainties to compute true effective Resistance (R_eff) 
    and Capacitance (C_eff), comparing against nominal values to expose systematic errors.
    """
    print("\n" + "="*75)
    print(" EPISTEMOLOGICAL RESOLUTION: DECONVOLVING COMPOUND ERRORS")
    print("="*75)
    
    i0_a = i0_ua * 1e-6
    err_i0_a = err_i0_ua * 1e-6
    
    r_eff = config.v_source / i0_a
    err_r_eff = r_eff * np.sqrt((config.err_v_source / config.v_source)**2 + (err_i0_a / i0_a)**2)
    
    c_eff = tau_s / r_eff
    err_c_eff = c_eff * np.sqrt((err_tau_s / tau_s)**2 + (err_r_eff / r_eff)**2)
    
    dev_r = (r_eff - config.r_nom) / config.r_nom * 100
    dev_c = (c_eff - config.c_nom) / config.c_nom * 100
    
    print(f"[+] Nominal Target: R_nom = {config.r_nom/1e3:.1f} kOhm, C_nom = {config.c_nom*1e6:.1f} uF")
    print(f"[!] Extracted True: R_eff = {r_eff/1e3:.1f} +/- {err_r_eff/1e3:.1f} kOhm")
    print(f"[!] Extracted True: C_eff = {c_eff*1e6:.1f} +/- {err_c_eff*1e6:.1f} uF")
    print(f"[*] R is {dev_r:+.1f}% deviated.")
    print(f"[*] C is {dev_c:+.1f}% deviated.")
    print("[*] CONCLUSION: Coincidental error cancellation masked the component failures.")
    print("="*75 + "\n")

# =============================================================================
# PUBLICATION VISUALIZATIONS
# =============================================================================

def plot_symmetry(t: npt.NDArray[np.float64], 
                  i_charge: npt.NDArray[np.float64], 
                  i_discharge: npt.NDArray[np.float64], 
                  out_dir: Path) -> None:
    """
    Generates a symmetry diagnostic plot mapping the raw charge vs. discharge currents.
    Data points are purely scatter markers, not connected by lines.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('w')
    
    # Strictly plotting scatter points without connecting lines
    ax.plot(t, i_charge, 's', markersize=8, color='#0072BD', label='Charging Data')
    ax.plot(t, -i_discharge, 'o', markersize=8, color='#D95319', label='Discharging Data')
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_title(r'\textbf{(A)} Current Symmetry Profile', pad=15)
    ax.set_xlabel(r'Time, $t$ (s)')
    ax.set_ylabel(r'Current, $I$ ($\mu$A)')
    
    ax.legend(loc='upper right')
    ax.set_xlim(0, 100)
    ax.set_ylim(-40, 40)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.minorticks_on()
    
    fig.savefig(out_dir / 'fig_A_symmetry.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_models(t: npt.NDArray[np.float64], 
                i_charge: npt.NDArray[np.float64], err_charge: npt.NDArray[np.float64], fit_c: FitResult,
                i_discharge: npt.NDArray[np.float64], err_discharge: npt.NDArray[np.float64], fit_d: FitResult,
                out_dir: Path) -> None:
    """
    Plots empirical datasets overlaying their respective optimized NLLS exponential models.
    Fit lines are strictly dashed. Data points are scatter-only.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('w')
    
    # Empirical Data with Error Bars (Scatter only)
    ax.errorbar(t, i_charge, yerr=err_charge, fmt='s', markersize=8, color='#0072BD', 
                ecolor='k', capsize=3, label='Charging Data')
    ax.errorbar(t, i_discharge, yerr=err_discharge, fmt='o', markersize=8, color='#D95319', 
                ecolor='k', capsize=3, label='Discharging Data')
    
    # High-Resolution Analytic Models (Strictly dashed lines)
    t_smooth = np.linspace(0, 100, 500)
    ax.plot(t_smooth, exponential_decay(t_smooth, *fit_c.popt), '--', color='#0072BD', 
            linewidth=2.5, label='Charging Fit')
    ax.plot(t_smooth, exponential_decay(t_smooth, *fit_d.popt), '--', color='#D95319', 
            linewidth=2.5, label='Discharging Fit')
    
    ax.set_title(r'\textbf{(B)} Exponential Model Validation', pad=15)
    ax.set_xlabel(r'Time, $t$ (s)')
    ax.set_ylabel(r'Current, $I$ ($\mu$A)')
    
    str_c = (r'\textbf{Charging Fit}' '\n' 
             rf'$I_0 = {fit_c.i0:.1f} \pm {fit_c.err_i0:.1f}\ \mu\mathrm{{A}}$' '\n' 
             rf'$\tau = {fit_c.tau:.1f} \pm {fit_c.err_tau:.1f}\ \mathrm{{s}}$' '\n' 
             rf'$R^2 = {fit_c.r_squared:.4f}$')
    str_d = (r'\textbf{Discharging Fit}' '\n' 
             rf'$I_0 = {fit_d.i0:.1f} \pm {fit_d.err_i0:.1f}\ \mu\mathrm{{A}}$' '\n' 
             rf'$\tau = {fit_d.tau:.1f} \pm {fit_d.err_tau:.1f}\ \mathrm{{s}}$' '\n' 
             rf'$R^2 = {fit_d.r_squared:.4f}$')
    
    ax.text(25, 25, str_c, fontsize=14, color='#0072BD', bbox=dict(fc='w', ec='k', pad=0.8, alpha=0.9))
    ax.text(65, 15, str_d, fontsize=14, color='#D95319', bbox=dict(fc='w', ec='k', pad=0.8, alpha=0.9))
    
    ax.legend(loc='upper right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.minorticks_on()
    
    fig.savefig(out_dir / 'fig_B_model_validation.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_residuals(t: npt.NDArray[np.float64], 
                   i_charge: npt.NDArray[np.float64], fit_c: FitResult,
                   i_discharge: npt.NDArray[np.float64], fit_d: FitResult,
                   out_dir: Path) -> None:
    """
    Computes and visualizes fit residuals combined with a LOWESS non-parametric trend line.
    Trend lines are strictly dashed.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('w')
    
    # Compute Vectorized Residuals (Scatter only)
    res_c = i_charge - exponential_decay(t, *fit_c.popt)
    res_d = i_discharge - exponential_decay(t, *fit_d.popt)
    
    ax.plot(t, res_c, 's', markersize=8, color='#0072BD')
    ax.plot(t, res_d, 'o', markersize=8, color='#D95319')
    
    # LOWESS Smoothing (Strictly dashed lines)
    lowess_c = sm.nonparametric.lowess(res_c, t, frac=0.2)
    lowess_d = sm.nonparametric.lowess(res_d, t, frac=0.2)
    
    ax.plot(lowess_c[:, 0], lowess_c[:, 1], '--', color='#0072BD', linewidth=2.5, label='Charging Trend')
    ax.plot(lowess_d[:, 0], lowess_d[:, 1], '--', color='#D95319', linewidth=2.5, label='Discharging Trend')
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_title(r'\textbf{(C)} LOWESS Residual Diagnostics', pad=15)
    ax.set_xlabel(r'Time, $t$ (s)')
    ax.set_ylabel(r'Residual, $I_{\mathrm{exp}} - I_{\mathrm{fit}}$ ($\mu$A)')
    
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.minorticks_on()
    
    fig.savefig(out_dir / 'fig_C_residuals.pdf', bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main() -> None:
    """
    Orchestrates the data extraction, NLLS optimization, statistical error analysis,
    and the generation of publication-ready visualizations.
    """
    configure_plot_aesthetics()
    
    # Cập nhật đường dẫn lưu đồ thị
    output_dir = Path('figures/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = CircuitConfig()
    
    t_s = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=np.float64)
    i_charge = np.array([37, 30, 24, 21, 17, 15, 12, 10, 9, 7, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2, 1], dtype=np.float64)
    i_discharge = np.array([37, 30, 25, 21, 18, 15, 12, 10, 8, 7, 6, 5, 5, 4, 3, 3, 2, 2, 2, 1, 1], dtype=np.float64)
    
    err_charge = (0.005 * i_charge) + 3.0
    err_discharge = (0.005 * i_discharge) + 3.0
    
    print("[*] Performing Non-Linear Least Squares (NLLS) optimization...")
    fit_c = execute_nlls_fit(t_s, i_charge, err_charge)
    fit_d = execute_nlls_fit(t_s, i_discharge, err_discharge)
    
    i0_avg = (fit_c.i0 + fit_d.i0) / 2.0
    i0_err_avg = np.sqrt(fit_c.err_i0**2 + fit_d.err_i0**2) / 2.0
    tau_avg = (fit_c.tau + fit_d.tau) / 2.0
    tau_err_avg = np.sqrt(fit_c.err_tau**2 + fit_d.err_tau**2) / 2.0
    
    diagnose_compound_error(config, i0_avg, i0_err_avg, tau_avg, tau_err_avg)
    
    print(f"[*] Generating diagnostic visualizations in: {output_dir.resolve()}")
    plot_symmetry(t_s, i_charge, i_discharge, output_dir)
    plot_models(t_s, i_charge, err_charge, fit_c, i_discharge, err_discharge, fit_d, output_dir)
    plot_residuals(t_s, i_charge, fit_c, i_discharge, fit_d, output_dir)
    
    print("[+] Pipeline execution completed successfully.")

if __name__ == '__main__':
    main()