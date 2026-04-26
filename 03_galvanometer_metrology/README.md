[Return to Main Laboratory Repository](../README.md)

---

# Module 03 | Metrological Calibration of a Ballistic Galvanometer: Error Propagation and Methodological Critique via Monte Carlo Simulations

## Abstract
This computational module investigates the metrological calibration of a ballistic galvanometer and its application in deriving unknown electrical capacitances. Utilizing a fully vectorized Ordinary Least Squares (OLS) algorithm coupled with a high-density Monte Carlo simulation framework ($N=100,000$), the analysis executes rigorous analytical uncertainty propagation from baseline instrumental tolerances to final physical parameters. The primary objective is to critically evaluate the epistemological validity of the measurement methodology. The investigation explicitly demonstrates how a two-step "calibrate-then-measure" paradigm fundamentally bounds the ultimate precision of the system by systematically propagating the calibration constant's structural uncertainty - an error pathway that is theoretically obviated in direct ratiometric comparative techniques. 

## Theoretical Framework
The underlying physics governing the measurement system relies on the linear relationship between the total electric charge $Q$ stored in a capacitor and the applied potential difference $V$, defined by the capacitance $C$:

$$ Q = CV $$

When this charged capacitor is discharged through a ballistic galvanometer, the impulse imparts an initial angular velocity to the coil. For small deflections, the maximum mechanical deflection $n$ (in divisions) is strictly proportional to the total transiting charge:

$$ Q = B n $$

where $B$ is the ballistic constant, an intrinsic metrological parameter of the specific instrument integrating its torsional, geometric, and magnetic properties.

Analytically, combining these equations yields a direct relationship between the circuit's electrical properties and the mechanical response of the galvanometer:

$$ CV = B n $$

In theoretical configurations, an unknown capacitance $C_x$ can be isolated by comparing its deflection $n_1$ against the deflection $n_0$ of a standard capacitor $C_0$ under an identical charging voltage. This ratiometric approach algebraically eliminates the ballistic constant $B$:

$$ C_x = C_0 \frac{n_1}{n_0} $$

## Computational & Analytical Methodology
To strictly quantify the physical parameters and their statistical confidence bounds, a multi-tiered computational approach was deployed:

* **Analytical Error Propagation:** Initial uncertainties in the stored charge $\delta Q$ were derived via first-order Taylor series expansion (partial derivatives) of the governing equations, incorporating absolute tolerances from the standard capacitor ($\delta C_0$) and the voltage source ($\delta V$):
  $$ \delta Q = \sqrt{\left(\frac{\partial Q}{\partial C_0} \delta C_0\right)^2 + \left(\frac{\partial Q}{\partial V} \delta V\right)^2} $$
* **Vectorized OLS Regression:** Extraction of the ballistic constant $B$ and the unknown capacitances $C_1, C_2$ was executed via a multi-dimensional Ordinary Least Squares linear regression algorithm, modeling the empirical data against $Q = Bn + Q_0$ and $Q = CV + Q_0$.
* **Monte Carlo Stochastic Validation:** To falsify the assumption of purely Gaussian error propagation and confirm parameter stability, a Monte Carlo simulation utilizing $N=100,000$ iterations was engineered. Instrumental uncertainties were modeled as normally distributed perturbations, mapping the ultimate statistical distributions of both the slopes and the zero-offsets.
* **Residual Diagnostics:** The appropriateness of the first-order linear models was validated through structural residual analysis, ensuring the absence of higher-order systematic phenomenologies (e.g., non-uniform mechanical damping).

## Scientific Epistemology & Critical Discussion
The computational extraction and statistical modeling of the empirical data reveal several profound insights into both the physical apparatus and the chosen experimental methodology:

**1. Methodological Discrepancy and Structural Uncertainty Amplification**
The theoretical framework establishes a direct ratiometric method designed to elegantly eliminate the ballistic constant $B$. However, the computational pipeline models a two-step "calibrate-then-measure" approach. The analysis mathematically proves a significant consequence of this divergence: explicitly determining $B = (17.77 \pm 1.04) \ \mu\text{C/div}$ injects a base relative uncertainty of ~5.9% into the computational framework. According to the partial derivative error matrix, the term $(n \cdot \delta B)^2$ aggressively dominates the error budget for all subsequent capacitance determinations. This methodological choice - while providing a robust, full-scale characterization of the galvanometer - structurally limits the precision of the final measurement compared to a theoretical single-point direct comparison.

**2. Analysis of the Zero-Offset Paradox ($Q_0$)**
The linear regressions yielded non-zero $y$-intercepts in both the calibration phase ($Q_0 = -2.57 \pm 14.18 \ \mu\text{C}$) and the capacitance determination phase ($Q_{0,1} = 11.94 \pm 17.68 \ \mu\text{C}$, $Q_{0,2} = 14.23 \pm 21.75 \ \mu\text{C}$). While the Monte Carlo distributions confirm these intercepts are statistically indistinguishable from zero, their sign fluctuation (negative during calibration, positive during measurement) warrants critical examination. The absence of a persistent non-zero offset definitively falsifies theories of static current leakage. Instead, these subtle intercept fluctuations likely map to transient, uncorrected systematic effects, such as minor temporal drift in the galvanometer's mechanical zero-point or fluctuating stray capacitances introduced when altering the physical circuit topology between phases.

**3. Linear Damping Absorptions**
The idealized model $Q \propto n$ assumes an undamped mechanical response. While physical damping is invariably present in the analogue scale, the high coefficients of determination ($R^2 \approx 0.997$) and the normalized scatter of residuals confirm that the damping force is highly linear across the operational regime. Consequently, the damping effect does not mandate a higher-order corrective polynomial but is entirely absorbed as a scalar modifier into the empirically determined constant $B$.

## Repository Structure

To adhere to software engineering standards, the legacy operational scripts have been refactored and consolidated. The updated directory architecture is as follows:

```text
03_Ballistic_Galvanometer_Calibration/
├── README.md
├── src/
│   └── analyze_galvanometer_metrology.py
└── figures/
    ├── Fig_A_Calibration.pdf
    ├── Fig_B_MC_Calibration.pdf
    ├── Fig_C_Capacitances.pdf
    └── Fig_D_MC_Capacitances.pdf
```

*Note: The `figures/` directory is automatically generated and populated during the execution of the primary analytical script.*

## Dependencies & Environment

Execution of the pipeline requires Python 3.9+ and the following scientific libraries:
* `numpy` (Numerical operations, matrix broadcasting, and vectorized OLS regression)
* `matplotlib` (High-resolution, LaTeX-typeset visualization)

Standard library modules (`dataclasses`, `pathlib`, `typing`) are strictly utilized for immutable state management and robust cross-platform I/O. A standard LaTeX distribution (e.g., TeX Live, MiKTeX) must be installed on the host system to render the publication-quality figures, as the script enforces `text.usetex = True`.

## Execution Instructions

To execute the diagnostic pipeline, navigate to the module directory and run the source script:

```bash
cd 03_Ballistic_Galvanometer_Calibration
python src/analyze_galvanometer_metrology.py
```

Upon execution, the script will:
1. Initialize immutable empirical data structures and metrological boundary constraints.
2. Perform fully vectorized Ordinary Least Squares (OLS) regressions across $N=100,000$ Monte Carlo simulations simultaneously, bypassing inefficient loop architectures.
3. Quantify structural error propagation to extract final physical bounds for the unknown capacitances.
4. Render and export four double-column formatted, publication-ready diagnostic plots directly to the `figures/` directory.

---
**Author:** Dang-Khoa N. Nguyen  
**Institution:** HCMC University of Technology and Engineering