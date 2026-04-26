# Computational Methods in Experimental Physics: Stochastic Validation & Systematic Deconvolution

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Strict%20OOP%20%7C%20Vectorized-success)](https://numpy.org/)
[![Rendering](https://img.shields.io/badge/Rendering-LaTeX%20%7C%20APS%20Standards-purple)](https://www.latex-project.org/)

## Abstract
This repository orchestrates a suite of advanced computational investigations into classical physical systems. The central thesis of this portfolio is the rigorous examination of the epistemological gap between idealized theoretical frameworks and empirical reality. Beyond deterministic parameter extraction, the analytical architectures engineered herein are designed to explicitly formulate physical boundaries, systematically propagate analytical uncertainty, and falsify naive theoretical models. By coupling numerical optimization with high-density stochastic validation, this repository systematically deconvolves compound experimental errors and exposes latent phenomenology across solid-state physics, optics, electromagnetism, and electrothermal dynamics.

## Table of Contents
1. [Core Methodological Pillars](#core-methodological-pillars)
2. [Research Portfolio & Modules](#research-portfolio--modules)
3. [Repository Architecture](#repository-architecture)
4. [Execution & Environment](#execution--environment)
5. [License](#license)

---

## Core Methodological Pillars

This repository adheres to stringent paradigms of modern computational physics and scientific software engineering, delineated by three foundational pillars:

### I. Epistemological Rigor & Model Falsification
The computational pipelines prioritize critical analytical thinking over rudimentary calculation. Algorithms are specifically engineered to probe the limitations of theoretical models. Methodologies include:
*   **Residual Diagnostics:** Implementation of non-parametric smoothing (LOWESS) to isolate systematic model-mismatch from stochastic noise, effectively unmasking unmodeled physics such as nonlinear mechanical damping or higher-order optical aberrations.
*   **Methodological Critique:** Mathematical demonstrations of how specific empirical paradigms (e.g., two-step calibration vs. ratiometric measurement) structurally bound the ultimate metrological precision of physical constants.
*   **Hypothesis Testing:** Rigorous utilization of reduced chi-squared ($\chi^2/\nu$) statistics and fundamental physical boundary conditions to formally falsify statistically robust but physically invalid empirical approximations.

### II. Advanced Computational Architecture
Legacy iterative paradigms have been entirely refactored into production-grade scientific software:
*   **Stochastic Validation:** Deployment of highly parallelized Monte Carlo simulations ($N=100,000$ iterations) to bypass first-order Taylor expansion limits, capturing the true underlying distribution of non-linear error propagation.
*   **Vectorization & JIT Compilation:** Core metrological equations and Ordinary Least Squares (OLS) regressions are fully vectorized via `numpy` and accelerated via `numba` Just-In-Time compilation, effectively eliminating computational bottlenecks.
*   **Immutable State Management:** Thread-safe execution during multi-core processing (`joblib`) is guaranteed via strict object-oriented architectures and Python `dataclasses`.

### III. Publication-Grade Aesthetics
All graphical outputs are generated programmatically without interactive blocking, strictly adhering to American Physical Society (APS) and Physical Review Letters (PRL) double-column formatting standards. Visualizations utilize global `matplotlib rcParams` to enforce Computer Modern serif fonts, flawless LaTeX rendering, and colorblind-compliant palettes, exporting exclusively to high-resolution PDF formats.

---

## Research Portfolio & Modules

The repository synthesizes five distinct physical investigations, unified by a rigorous computational framework:

### [Module 01: Carrier Transport Dynamics in High-Power LEDs](./01_carrier_transport_in_leds)
Investigates non-ideal charge carrier transport mechanisms in narrow-bandgap (AlGaInP) and wide-bandgap (InGaN) light-emitting diodes. Formulates a heavily bounded, non-linear least squares Monte Carlo pipeline to extract phenomenological diode parameters ($n$, $I_s$, $R_s$). Employs digital signal processing (Savitzky-Golay filtering) to evaluate the discrete local ideality factor $n(I)$, probing transitional regimes between non-radiative Sah-Noyce-Shockley recombination and ideal diffusion.

### [Module 02: Diffraction Grating Systematics & Hypothesis Testing](./02_diffraction_grating_systematics)
Utilizes Weighted Least Squares (WLS) regression and non-parametric residual analysis to evaluate optical diffraction models. Systematically propagates spatial-to-angular uncertainties to formally test an ideal linear grating theory against a non-linear geometric zero-point offset hypothesis. The analysis mathematically falsifies the misalignment theory, proving that the latent residuals originate from higher-order optical aberrations rather than apparatus offset.

###[Module 03: Ballistic Galvanometer Metrology & Error Propagation](./03_galvanometer_metrology)
Executes a multi-dimensional, fully vectorized OLS regression across a $100,000$-trial Monte Carlo framework to derive fundamental metrological constants. The investigation critically evaluates the empirical methodology, proving analytically how structural uncertainty in the instrument's ballistic constant dominantly amplifies subsequent capacitance error manifolds, presenting a rigorous argument for theoretical ratiometric measurement over step-wise calibration.

### [Module 04: RC Transient Dynamics & Error Deconvolution](./04_rc_transient_systematics)
Investigates the transient temporal evolution of a series RC circuit. Resolves a profound statistical paradox where extracted initial currents deviate by $5.6\sigma$ while macroscopic time constants remain in near-perfect nominal agreement. By formulating a strict multi-variable partial-derivative error propagation matrix, the computational pipeline successfully deconvolves a compound systematic error, serving as a definitive demonstration of coincidental error cancellation and the perils of confirmation bias.

###[Module 05: Tungsten Electrothermal Dynamics & Polynomial Falsification](./05_tungsten_electrothermal_dynamics)
Models the complex non-Ohmic interplay between Joule heating and temperature-dependent resistivity in an incandescent tungsten filament. Employs analytical Jacobian-based error propagation coupled with non-uniform domain sampling. The investigation decisively falsifies second-order empirical polynomial models - despite near-unity $R^2$ values - by revealing mathematically rigorous but physically absurd boundary violations, such as non-zero current intercepts and unphysical inverted resistance concavities.

---

## Repository Architecture

```text
.
├── 01_carrier_transport_in_leds/
│   ├── figures/
│   ├── src/
│   ├── README.md
│   └── report_led_carrier_transport.pdf
├── 02_diffraction_grating_systematics/
│   ├── figures/
│   ├── src/
│   ├── README.md
│   └── Diffraction_Grating_Analysis_Report.pdf
├── 03_galvanometer_metrology/
│   ├── figures/
│   ├── src/
│   └── README.md
├── 04_rc_transient_systematics/
│   ├── figures/
│   ├── src/
│   └── README.md
├── 05_tungsten_electrothermal_dynamics/
    ├── figures/
    ├── src/
    └── README.md
```
*(Directory contents abridged for brevity. See individual modules for complete structures).*

---

## Execution & Environment

To guarantee absolute reproducibility across different computational environments, all physical dependencies have been locked. 

**System Requirements:**
A standard local TeX distribution (e.g., *TeX Live*, *MiKTeX*, or *MacTeX*) must be installed and accessible via the system PATH to permit the native rendering of LaTeX strings within `matplotlib`, as `text.usetex = True` is strictly enforced across all modules.

**Installation & Execution:**
1. Clone the repository and navigate to the root directory:
   ```bash
   git clone https://github.com/KhoaDNguyenNguyen/Computational-Methods-in-Physics-Labs.git
   cd Computational-Methods-in-Physics-Labs
   ```
2. Install the required computational and numerical libraries using the provided environment file:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to any specific module and execute the pipeline analytically:
   ```bash
   cd 01_carrier_transport_in_leds
   python src/algainp_transport_analysis.py
   ```

The scripts are engineered to execute autonomously - allocating computational threads, propagating stochastic uncertainties, outputting statistical inferences to the terminal, and regenerating publication-grade PDFs directly into the respective `figures/` directories.

---

## License

This project is open-sourced under the MIT License - see the [LICENSE](LICENSE) file for details. Academic referencing and forking are highly encouraged.

---
**Author:** [Dang-Khoa N. Nguyen](mailto:khoadnguyennguyen@gmail.com)  
**Institution:** HCMC University of Technology and Engineering  
**ORCID:** [0009-0002-9631-0734](https://orcid.org/0009-0002-9631-0734)