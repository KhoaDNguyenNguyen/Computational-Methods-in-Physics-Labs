[Return to Main Laboratory Repository](../README.md)

---

# Module 02 | Diagnosing Latent Systematic Errors in Diffraction Gratings via Model Falsification

## Abstract
This module provides a rigorous computational pipeline for analyzing experimental data derived from optical diffraction grating systems. Standard undergraduate and introductory graduate treatments of diffraction typically assume an ideal linear relationship governed by the grating equation. This repository implements advanced statistical methodologies - specifically vectorized Monte Carlo uncertainty propagation, Weighted Least Squares (WLS) regression, and locally weighted scatterplot smoothing (LOWESS) - to detect, quantify, and formally falsify underlying systematic errors, such as geometric zero-point offsets.

## Theoretical Framework

The analysis evaluates two competing physical hypotheses regarding the experimental setup:

1. **The Null Hypothesis ($\mathcal{H}_0$) - Ideal Grating Model:** 
   Assumes a perfect alignment where the sine of the diffraction angle is strictly proportional to the incident wavelength:
   $$\sin(\theta) = m \cdot \lambda$$
   where $m = 1/d$ is the reciprocal of the grating period.

2. **The Alternate Hypothesis ($\mathcal{H}_1$) - Geometric Misalignment Model:**
   Introduces a structural zero-point offset parameter ($\Delta x$) to account for potential lateral misalignment of the projection screen:
   $$\theta_{\text{meas}} = \arctan\left( \frac{L \tan(\arcsin(\lambda / d)) + \Delta x}{L} \right)$$

Model falsification is executed by comparing the reduced chi-squared ($\chi^2/\nu$) statistics of both models. If the highly constrained non-linear model ($\mathcal{H}_1$) yields a statistically inferior fit compared to the naive model ($\mathcal{H}_0$), the offset hypothesis is falsified, indicating that the observed systematic residuals are likely dominated by higher-order optical aberrations (e.g., spherical aberration, lens distortion) rather than simple physical misalignment.

## Computational Methodology

* **Monte Carlo Uncertainty Propagation:** Non-linear spatial measurement uncertainties ($\Delta x, \Delta L$) are propagated into angular uncertainties ($\Delta \sin(\theta)$) using a fully vectorized Monte Carlo simulation ($N = 100,000$ trials), avoiding the approximations of standard first-order Taylor expansion methods.
* **WLS Regression:** Parameter extraction is performed via inverse-variance weighting (`scipy.optimize.curve_fit`), ensuring that data points with higher propagated variance exert proportionally less leverage on the regression manifold.
* **Non-parametric Residual Analysis:** The `statsmodels` implementation of LOWESS is applied to the regression residuals to empirically unmask latent non-linear dependencies without imposing prior analytical structures.

## Repository Structure

To adhere to software engineering standards, the legacy operational scripts have been refactored and consolidated. The updated directory architecture is as follows:

```text
02_Diffraction_Grating_Residual_Analysis/
├── README.md
├── Diffraction_Grating_Analysis_Report.pdf
├── src/
│   └── diffraction_systematics.py
├── data/
│   └── mc_propagation_cache.npz
└── figures/
    ├── fig_diffraction_linear_fit.pdf
    └── fig_diffraction_residuals.pdf
```

*Note: The `data/` and `figures/` directories are automatically generated or populated during the execution of the primary analytical script.*

## Dependencies & Environment

Execution of the pipeline requires Python 3.9+ and the following scientific libraries:
* `numpy` (Numerical operations and vectorization)
* `pandas` (Data structuring)
* `scipy` (Non-linear optimization constraints)
* `statsmodels` (Non-parametric smoothing algorithms)
* `matplotlib` (High-resolution, LaTeX-typeset visualization)

A standard LaTeX distribution (e.g., TeX Live, MiKTeX) must be installed on the host system to render the publication-quality figures, as the script enforces `text.usetex = True`.

## Execution Instructions

To execute the diagnostic pipeline, navigate to the module directory and run the source script:

```bash
cd 02_Diffraction_Grating_Residual_Analysis
python src/diffraction_systematics.py
```

Upon execution, the script will:
1. Load empirical data and compute angular means/variances via Monte Carlo integration (caching the results to `data/mc_propagation_cache.npz` to minimize subsequent runtime).
2. Perform WLS regressions for both $\mathcal{H}_0$ and $\mathcal{H}_1$.
3. Render and save double-column formatted, publication-ready diagnostic plots to the `figures/` directory.
4. Output a formalized statistical inference report to the standard output, detailing the epistemological verdict of the hypothesis test.

---
**Author:** Dang-Khoa N. Nguyen  
**Institution:** HCMC University of Technology and Engineering