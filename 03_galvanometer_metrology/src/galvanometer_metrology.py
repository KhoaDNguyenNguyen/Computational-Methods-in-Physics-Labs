# ==============================================================================
# Title:        Ballistic Galvanometry: Metrological Calibration and Error Propagation
# Author:       Dang-Khoa N. Nguyen
# Institution:  HCMC University of Technology and Engineering
# Description:  Performs metrological calibration of a ballistic galvanometer 
#               via Monte Carlo simulations. Determines unknown capacitances through 
#               vectorized Ordinary Least Squares (OLS) regression while strictly 
#               propagating instrumental uncertainties.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict

# =============================================================================
# DATA STRUCTURES & CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ExperimentData:
    """Immutable data structure containing empirical constants and measurements."""
    C0_uF: float = 30.0
    ERR_C0: float = 0.5
    ERR_N: float = 0.5
    
    # Calibration Measurements
    V_cal: np.ndarray = field(default_factory=lambda: np.array([2.5, 5.0, 7.5, 10.0, 12.5]))
    n_cal: np.ndarray = field(default_factory=lambda: np.array([4.2, 8.5, 12.5, 17.0, 21.0]))
    
    # Unknown Capacitance Measurements
    V_meas: np.ndarray = field(default_factory=lambda: np.array([2.5, 5.0, 7.5, 10.0, 12.5]))
    n_c1: np.ndarray = field(default_factory=lambda: np.array([7.0, 13.0, 19.0, 25.0, 32.0]))
    n_c2: np.ndarray = field(default_factory=lambda: np.array([9.0, 17.0, 25.0, 32.0, 42.0]))

@dataclass
class MCResults:
    """Data structure for storing generalized Monte Carlo simulation statistics."""
    mean_slope: float
    std_slope: float
    mean_intercept: float
    std_intercept: float
    sim_slopes: np.ndarray
    sim_intercepts: np.ndarray

# =============================================================================
# NUMERICAL & PHYSICAL METHODS
# =============================================================================

def get_voltage_uncertainty(V: np.ndarray) -> np.ndarray:
    """Computes instrumental voltage uncertainty based on experimental specifications."""
    return (0.005 * V) + 0.3


def vectorized_ols_regression(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs fully vectorized Ordinary Least Squares (OLS) linear regression
    across thousands of Monte Carlo simulations simultaneously.
    """
    x_mean = np.mean(x, axis=1, keepdims=True)
    y_mean = np.mean(y, axis=1, keepdims=True)
    
    numerator = np.sum((x - x_mean) * (y - y_mean), axis=1)
    denominator = np.sum((x - x_mean) ** 2, axis=1)
    
    slopes = numerator / denominator
    intercepts = np.squeeze(y_mean) - slopes * np.squeeze(x_mean)
    
    return slopes, intercepts


def calibrate_galvanometer(data: ExperimentData, num_sims: int) -> Tuple[MCResults, np.ndarray, np.ndarray]:
    """Executes Phase 1: Derives the Ballistic Constant (B) via Monte Carlo methods."""
    print(f"[*] Executing Vectorized Galvanometer Calibration (N={num_sims:,})...")
    
    Q_true = data.C0_uF * data.V_cal
    err_V = get_voltage_uncertainty(data.V_cal)
    err_Q = Q_true * np.sqrt((data.ERR_C0 / data.C0_uF)**2 + (err_V / data.V_cal)**2)
    
    n_sim = data.n_cal + data.ERR_N * np.random.randn(num_sims, len(data.n_cal))
    Q_sim = Q_true + err_Q * np.random.randn(num_sims, len(Q_true))
    
    sim_B, sim_Q0 = vectorized_ols_regression(n_sim, Q_sim)
    
    results = MCResults(
        mean_slope=float(np.mean(sim_B)), std_slope=float(np.std(sim_B)),
        mean_intercept=float(np.mean(sim_Q0)), std_intercept=float(np.std(sim_Q0)),
        sim_slopes=sim_B, sim_intercepts=sim_Q0
    )
    
    return results, Q_true, err_Q


def determine_capacitances(data: ExperimentData, cal_res: MCResults, num_sims: int) -> Tuple[MCResults, MCResults]:
    """Executes Phase 2: Resolves unknown capacitances propagating B's uncertainty."""
    print(f"[*] Executing Vectorized Capacitance Determination (N={num_sims:,})...")
    
    def simulate_capacitor(n_data: np.ndarray) -> MCResults:
        B_sim = cal_res.mean_slope + cal_res.std_slope * np.random.randn(num_sims, 1)
        n_sim = n_data + data.ERR_N * np.random.randn(num_sims, len(n_data))
        V_sim = data.V_meas + get_voltage_uncertainty(data.V_meas) * np.random.randn(num_sims, len(data.V_meas))
        
        Q_sim = B_sim * n_sim
        sim_C, sim_Q0 = vectorized_ols_regression(V_sim, Q_sim)
        
        return MCResults(
            mean_slope=float(np.mean(sim_C)), std_slope=float(np.std(sim_C)),
            mean_intercept=float(np.mean(sim_Q0)), std_intercept=float(np.std(sim_Q0)),
            sim_slopes=sim_C, sim_intercepts=sim_Q0
        )
        
    return simulate_capacitor(data.n_c1), simulate_capacitor(data.n_c2)

# =============================================================================
# VISUALIZATION PIPELINE
# =============================================================================

def configure_academic_aesthetics() -> Dict[str, str]:
    """Applies rigorous academic journal formatting to matplotlib."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
    })
    
    # Exact colors from the original MATLAB/Python plots
    return {
        'data_blue': '#0072BD',
        'data_orange': '#D95319',
        'hist_blue': '#87CEFA',
        'hist_green': '#AED581',
        'hist_orange': '#F4A582',
        'gray_err': '#333333'
    }


def plot_histogram(ax: plt.Axes, data: np.ndarray, title: str, xlabel: str, 
                   mean: float, std: float, facecolor: str, n_sims: int = 100000,
                   stats_title: str = "Statistics") -> None:
    """Helper function to plot standardized statistical histograms."""
    ax.hist(data, bins=50, color=facecolor, edgecolor='black', linewidth=1.2)
    ax.axvline(mean, color='red', linestyle='--', linewidth=3.0)
    
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Frequency")
    
    stats = (fr"\textbf{{{stats_title}}}" "\n"
             fr"Mean = \textbf{{{mean:.2f}}}" "\n" 
             fr"Std Dev = \textbf{{{std:.2f}}}" "\n"
             fr"N = \textbf{{{n_sims}}}")
    # Omit N in some subplots if desired, but adding it universally matches Fig B.
    if "Intercept" in title and "Q_{0" in title:  # For Fig D where N isn't shown
        stats = (fr"\textbf{{{stats_title}}}" "\n"
                 fr"Mean = \textbf{{{mean:.2f}}}" "\n" 
                 fr"Std Dev = \textbf{{{std:.2f}}}")

    ax.text(0.04, 0.96, stats, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black', lw=1.5))
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.minorticks_on()


def export_figures(data: ExperimentData, cal_res: MCResults, Q_true: np.ndarray, err_Q: np.ndarray, 
                   c1_res: MCResults, c2_res: MCResults, num_sims: int, output_dir: Path) -> None:
    """Orchestrates the generation and PDF export of all diagnostic plots."""
    print("[*] Rendering high-resolution PDF outputs...")
    colors = configure_academic_aesthetics()
    
    # ---------------------------------------------------------
    # Figure A: Calibration Curve
    # ---------------------------------------------------------
    fig_A, ax = plt.subplots(figsize=(8, 6))
    
    # R-squared calculation
    y_mean = np.mean(Q_true)
    ss_tot = np.sum((Q_true - y_mean)**2)
    ss_res = np.sum((Q_true - (cal_res.mean_slope * data.n_cal + cal_res.mean_intercept))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # STRICTLY NO CONNECTING LINES: ls='none'
    ax.errorbar(data.n_cal, Q_true, yerr=err_Q, xerr=data.ERR_N, fmt='s', ls='none', markersize=8,
                markerfacecolor=colors['data_blue'], markeredgecolor='black', markeredgewidth=1.0, 
                ecolor=colors['gray_err'], elinewidth=1.5, capsize=4, label=r"Experimental Data")
    
    # STRICTLY DASHED TRENDLINE: linestyle='--'
    n_line = np.linspace(0, 24, 100)
    ax.plot(n_line, cal_res.mean_slope * n_line + cal_res.mean_intercept, 
            linestyle='--', color=colors['data_orange'], linewidth=3, 
            label=fr"Best-Fit: $Q = {cal_res.mean_slope:.2f}n - {abs(cal_res.mean_intercept):.2f}$")
    
    ax.set_title(r"\textbf{Figure A: Galvanometer Calibration}")
    ax.set_xlabel(r"Galvanometer Deflection, $n$ (divisions)")
    ax.set_ylabel(r"Stored Charge, $Q$ ($\mu$C)")
    ax.set_xlim(0, 23.5)
    ax.set_ylim(0, 420)
    ax.legend(loc='lower right', edgecolor='black', framealpha=1.0)
    ax.grid(True, linestyle='-', alpha=0.3); ax.minorticks_on()
    
    fit_txt = (fr"\textbf{{Fit Results}}" "\n"
               fr"$B = \mathbf{{{cal_res.mean_slope:.2f} \pm {cal_res.std_slope:.2f}}}\ \mu$C/division" "\n"
               fr"$Q_0 = \mathbf{{{cal_res.mean_intercept:.2f} \pm {cal_res.std_intercept:.2f}}}\ \mu$C" "\n"
               fr"$R^2 = \mathbf{{{r2:.5f}}}$")
    ax.text(0.1, 0.90, fit_txt, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black', lw=2.0))
    
    fig_A.savefig(output_dir / 'fig_galvanometer_calibration.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig_A)

    # ---------------------------------------------------------
    # Figure B: Calibration MC Distributions
    # ---------------------------------------------------------
    fig_B, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig_B.suptitle(r"\textbf{Figure B: Monte Carlo Parameter Distributions}", fontsize=18, y=0.98)
    
    plot_histogram(ax1, cal_res.sim_slopes, r"Distribution of Ballistic Constant ($B$)", r"$B$ ($\mu$C/division)", 
                   cal_res.mean_slope, cal_res.std_slope, colors['hist_blue'], num_sims, "Statistics")
    plot_histogram(ax2, cal_res.sim_intercepts, r"Distribution of Y-Intercept ($Q_0$)", r"$Q_0$ ($\mu$C)", 
                   cal_res.mean_intercept, cal_res.std_intercept, colors['hist_green'], num_sims, "Statistics")
    
    fig_B.tight_layout(rect=[0, 0, 1, 0.95])
    fig_B.savefig(output_dir / 'fig_mc_calibration_distributions.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig_B)

    # ---------------------------------------------------------
    # Figure C: Unknown Capacitances Fit
    # ---------------------------------------------------------
    fig_C, ax = plt.subplots(figsize=(8, 6.5))
    err_V_arr = get_voltage_uncertainty(data.V_meas)
    
    Q1_true, Q2_true = cal_res.mean_slope * data.n_c1, cal_res.mean_slope * data.n_c2
    err_Q1 = Q1_true * np.sqrt((cal_res.std_slope/cal_res.mean_slope)**2 + (data.ERR_N/data.n_c1)**2)
    err_Q2 = Q2_true * np.sqrt((cal_res.std_slope/cal_res.mean_slope)**2 + (data.ERR_N/data.n_c2)**2)
    
    # STRICTLY NO CONNECTING LINES: ls='none'
    ax.errorbar(data.V_meas, Q1_true, yerr=err_Q1, xerr=err_V_arr, fmt='s', ls='none', markersize=8,
                markerfacecolor=colors['data_blue'], markeredgecolor='black', ecolor=colors['gray_err'], capsize=4, label=r"Data Points ($C_1$)")
    ax.errorbar(data.V_meas, Q2_true, yerr=err_Q2, xerr=err_V_arr, fmt='o', ls='none', markersize=9,
                markerfacecolor=colors['data_orange'], markeredgecolor='black', ecolor=colors['gray_err'], capsize=4, label=r"Data Points ($C_2$)")
    
    # STRICTLY DASHED TRENDLINES: linestyle='--'
    V_line = np.linspace(0, 13.5, 100)
    ax.plot(V_line, c1_res.mean_slope * V_line + c1_res.mean_intercept, 
            linestyle='--', color=colors['data_blue'], linewidth=3, 
            label=fr"Fit ($C_1$): $Q = {c1_res.mean_slope:.2f}V - {abs(c1_res.mean_intercept):.2f}$")
    ax.plot(V_line, c2_res.mean_slope * V_line + c2_res.mean_intercept, 
            linestyle='--', color=colors['data_orange'], linewidth=3, 
            label=fr"Fit ($C_2$): $Q = {c2_res.mean_slope:.2f}V - {abs(c2_res.mean_intercept):.2f}$")
    
    ax.set_title(r"\textbf{Figure C: Determination of Unknown Capacitances}")
    ax.set_xlabel(r"Charging Voltage, $V$ (V)")
    ax.set_ylabel(r"Calculated Charge, $Q$ ($\mu$C)")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 800)
    ax.legend(loc='lower right', edgecolor='black', framealpha=1.0)
    ax.grid(True, linestyle='-', alpha=0.3); ax.minorticks_on()
    
    res_txt = (fr"\textbf{{Final Fit Parameters (Monte Carlo)}}" "\n"
               fr"\underline{{Capacitor 1:}}" "\n"
               fr"$C_1 = \mathbf{{{c1_res.mean_slope:.2f} \pm {c1_res.std_slope:.2f}}}\ \mu$F" "\n"
               fr"$Q_{{0,1}} = \mathbf{{{c1_res.mean_intercept:.2f} \pm {c1_res.std_intercept:.2f}}}\ \mu$C" "\n"
               fr"\underline{{Capacitor 2:}}" "\n"
               fr"$C_2 = \mathbf{{{c2_res.mean_slope:.2f} \pm {c2_res.std_slope:.2f}}}\ \mu$F" "\n"
               fr"$Q_{{0,2}} = \mathbf{{{c2_res.mean_intercept:.2f} \pm {c2_res.std_intercept:.2f}}}\ \mu$C")
    ax.text(0.04, 0.96, res_txt, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black', lw=2.0))
    
    fig_C.savefig(output_dir / 'fig_capacitance_determination.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig_C)

    # ---------------------------------------------------------
    # Figure D: Capacitance MC Distributions
    # ---------------------------------------------------------
    fig_D, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig_D.suptitle(r"\textbf{Figure D: Monte Carlo Parameter Distributions}", fontsize=20, y=0.96)
    
    plot_histogram(axes[0,0], c1_res.sim_slopes, r"Distribution of Capacitance ($C_1$)", r"$C_1$ ($\mu$F)", 
                   c1_res.mean_slope, c1_res.std_slope, colors['hist_blue'], num_sims, r"Statistics ($C_1$)")
    plot_histogram(axes[0,1], c2_res.sim_slopes, r"Distribution of Capacitance ($C_2$)", r"$C_2$ ($\mu$F)", 
                   c2_res.mean_slope, c2_res.std_slope, colors['hist_orange'], num_sims, r"Statistics ($C_2$)")
    plot_histogram(axes[1,0], c1_res.sim_intercepts, r"Distribution of Intercept ($Q_{0,1}$)", r"$Q_{0,1}$ ($\mu$C)", 
                   c1_res.mean_intercept, c1_res.std_intercept, colors['hist_blue'], num_sims, r"Statistics ($Q_{0,1}$)")
    plot_histogram(axes[1,1], c2_res.sim_intercepts, r"Distribution of Intercept ($Q_{0,2}$)", r"$Q_{0,2}$ ($\mu$C)", 
                   c2_res.mean_intercept, c2_res.std_intercept, colors['hist_orange'], num_sims, r"Statistics ($Q_{0,2}$)")
    
    fig_D.tight_layout(rect=[0, 0, 1, 0.93])
    fig_D.subplots_adjust(hspace=0.35)
    fig_D.savefig(output_dir / 'fig_mc_capacitance_distributions.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig_D)

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main() -> None:
    NUM_SIMULATIONS = 100_000
    OUTPUT_DIR = Path('figures')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    experiment = ExperimentData()
    
    # Phase 1: Metrological Calibration
    calibration_results, true_charge, charge_error = calibrate_galvanometer(experiment, NUM_SIMULATIONS)
    
    # Phase 2: Unknown Target Determinations
    c1_results, c2_results = determine_capacitances(experiment, calibration_results, NUM_SIMULATIONS)
    
    # Data Visualization & Scientific Export
    export_figures(
        data=experiment, 
        cal_res=calibration_results, 
        Q_true=true_charge, 
        err_Q=charge_error, 
        c1_res=c1_results, 
        c2_res=c2_results, 
        num_sims=NUM_SIMULATIONS,
        output_dir=OUTPUT_DIR
    )
    print(f"[+] Processing complete. High-resolution publication PDFs exported to '/{OUTPUT_DIR.name}/'.")

if __name__ == '__main__':
    main()