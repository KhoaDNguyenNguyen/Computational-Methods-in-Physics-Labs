[Return to Main Laboratory Repository](../README.md)

---

# Module 04 | RC Transient Dynamics: Deconvolving Compound Systematic Errors via NLLS Optimization

## Abstract
This module presents a computational investigation into the transient dynamics of a series Resistor-Capacitor (RC) circuit, leveraging Non-Linear Least Squares (NLLS) optimization and rigorous parametric uncertainty propagation to deconvolve a profound compound systematic error. Initial empirical evaluation presented a significant statistical paradox: the extracted initial current deviated from nominal expectations by $5.6\sigma$, while the time constant exhibited near-perfect agreement ($0.4\sigma$ deviation). By systematically falsifying single-cause anomalies and applying multi-variable partial-derivative error propagation, the analysis successfully isolated simultaneous, counteracting component failures. The investigation ultimately serves as a rigorous mathematical demonstration of "coincidental error cancellation," highlighting the critical epistemological perils of confirmation bias when validating physical models against a highly constrained subset of experimental parameters.

## Theoretical Framework
The temporal evolution of a series RC circuit is governed by Kirchhoff's Voltage Law (KVL). Differentiating the fundamental KVL relation with respect to time and substituting the current-charge differential relationship ($I = dQ/dt$) yields a first-order homogeneous differential equation:

$$ R \frac{dI}{dt} + \frac{1}{C}I(t) = 0 $$

Solving this linear differential equation via the method of separation of variables and integrating from an initial fully-discharged boundary condition yields the explicit exponential decay model:

$$ I(t) = I_0 e^{-t/\tau} $$

where the physical constraints of the system dictate two independent parameters:
*   $I_0 = \frac{V}{R}$ is the peak initial current at $t = 0$.
*   $\tau \equiv RC$ is the characteristic time constant governing the macroscopic decay timescale.

## Computational & Analytical Methodology
*   **Non-Linear Least Squares (NLLS) Optimization:** The transient charging and discharging profiles were fitted directly to the analytical exponential model using robust NLLS algorithms, avoiding the statistical distortions inherent in logarithmic linearization.
*   **Non-Parametric Residual Diagnostics:** Fit fidelity was evaluated using locally weighted scatterplot smoothing (LOWESS) applied to the residual domains ($I_{\mathrm{exp}} - I_{\mathrm{fit}}$) to detect and quantify subtle systematic deviations from the ideal exponential decay model.
*   **Rigorous Uncertainty Propagation:** The effective resistance ($R_{\mathrm{eff}}$) and capacitance ($C_{\mathrm{eff}}$) were extracted by propagating empirical uncertainties via partial derivatives:
    $$ \sigma_{R_{\mathrm{eff}}} = R_{\mathrm{eff}} \sqrt{\left(\frac{\sigma_V}{V}\right)^2 + \left(\frac{\sigma_{I_0}}{I_0}\right)^2} $$
    $$ \sigma_{C_{\mathrm{eff}}} = C_{\mathrm{eff}} \sqrt{\left(\frac{\sigma_\tau}{\tau}\right)^2 + \left(\frac{\sigma_{R_{\mathrm{eff}}}}{R_{\mathrm{eff}}}\right)^2} $$

## Scientific Epistemology & Critical Discussion
The core contribution of this computational analysis lies in the resolution of an apparent physical paradox. Empirical NLLS extraction yielded an initial current ($I_0$) that deviated from the nominal theoretical prediction by an untenable $5.6\sigma$, whereas the extracted time constant ($\tau$) agreed remarkably well with nominal limits (deviating by only $0.4\sigma$). 

A naive empirical interpretation, driven by confirmation bias, might erroneously validate the circuit's structural integrity based solely on the highly accurate time constant, dismissing the initial current deviation as a secondary artifact. However, a multi-variable deconvolution approach mandated the falsification of simpler hypotheses (such as minimal voltage source instability or negligible ammeter internal resistance). 

Applying the strict partial-derivative propagation framework detailed above, the true effective parameters of the system were unmasked, revealing a **compound systematic error**. The physical system contained a resistance that was $+68\%$ higher than its nominal rating, coupled with a capacitance that was $-39\%$ deficient. 

The profound agreement in the time constant was mathematically proven to be a remarkable artifact of coincidental error cancellation:

$$ \tau_{\mathrm{exp}} = R_{\mathrm{eff}} \times C_{\mathrm{eff}} $$
$$ \tau_{\mathrm{exp}} \approx (1.68 \times R_{\mathrm{nom}}) \times (0.61 \times C_{\mathrm{nom}}) $$
$$ \tau_{\mathrm{exp}} \approx 1.025 \times \tau_{\mathrm{nom}} $$

This near-perfect cancellation masked the fundamentally flawed nature of the underlying physical components. Methodologically, this analysis dictates that a physical model's validity is contingent upon its ability to seamlessly predict *all* coupled measurable consequences ($I_0$ and $\tau$ independently). It mathematically demonstrates that internal consistency and agreement with a convenient subset of experimental data do not guarantee the epistemological truth of the physical system.

## Repository Structure

To adhere to software engineering standards, the legacy operational scripts have been refactored and consolidated. The updated directory architecture is as follows:

```text
04_RC_Transient_Compound_Systematics/
├── README.md
├── src/
│   └── transient_error_deconvolution.py
└── figures/
    ├── fig_A_symmetry.pdf
    ├── fig_B_model_validation.pdf
    └── fig_C_residuals.pdf
```

*Note: The `figures/` directory is automatically generated and populated during the execution of the primary analytical script.*

## Dependencies & Environment

Execution of the pipeline requires Python 3.9+ and the following scientific libraries:
* `numpy` (Numerical operations, vectorized equations, and array structuring)
* `scipy` (Non-linear least squares optimization via `curve_fit`)
* `statsmodels` (LOWESS non-parametric smoothing algorithms for residual diagnostics)
* `matplotlib` (High-resolution, LaTeX-typeset visualization)

A standard LaTeX distribution (e.g., TeX Live, MiKTeX) must be installed on the host system to render the publication-quality figures, as the script strictly enforces `text.usetex = True`.

## Execution Instructions

To execute the diagnostic pipeline, navigate to the module directory and run the source script:

```bash
cd 04_RC_Transient_Compound_Systematics
python src/transient_error_deconvolution.py
```

Upon execution, the script will:
1. Initialize nominal circuit configurations and load the empirical transient data arrays.
2. Perform Non-Linear Least Squares (NLLS) regressions for both the charging and discharging phases.
3. Propagate parametric uncertainties via partial derivatives to compute $R_{\mathrm{eff}}$ and $C_{\mathrm{eff}}$, outputting a formalized diagnostic report to the standard output that resolves the compound systematic error.
4. Render and save double-column formatted, publication-ready diagnostic plots (symmetry profiles, exponential model validation, and LOWESS residuals) to the `figures/` directory.

---
**Author:** Dang-Khoa N. Nguyen  
**Institution:** HCMC University of Technology and Engineering