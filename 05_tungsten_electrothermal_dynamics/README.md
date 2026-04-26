[Return to Main Laboratory Repository](../README.md)

---

# Module 05 | Diagnosing Electro-Thermal Dynamics and Falsifying Empirical Models in Tungsten Filaments

## Abstract
This computational module investigates the non-Ohmic electro-thermal dynamics of a tungsten filament via rigorous statistical modeling and residual analysis. By coupling analytical Jacobian-based error propagation with highly vectorized Monte Carlo simulations, the current-voltage (I-V) and resistance-voltage (R-V) dependencies are evaluated against second-order empirical polynomial models. Although these mathematical approximations yield exceptionally high coefficients of determination ($R^2 > 0.99$), non-parametric residual diagnostics and fundamental boundary analyses reveal profound systemic inadequacies - specifically, unphysical intercepts and inverted concavities. This investigation highlights the critical epistemological distinction between statistically robust curve-fitting and physically consistent phenomenological modeling.

## Theoretical Framework
The electro-thermal behavior of an incandescent tungsten filament is governed by the interplay between Joule heating and the temperature-dependent resistivity of the metallic conductor. An applied voltage $V$ drives a current $I$, resulting in electrical power dissipation $P$ according to the Joule-Lenz law:

$$ P = VI = I^2R $$

This dissipated power elevates the filament's temperature $T$. For a transition metal such as tungsten, electrical resistance is a strongly increasing function of temperature, accurately described over a wide operational range by a Taylor-series polynomial expansion:

$$ R(T) = R_0 \left[ 1 + \alpha(T - T_0) + \beta(T - T_0)^2 + \dots \right] $$

where $R_0$ is the baseline resistance at reference temperature $T_0$, while $\alpha$ and $\beta$ represent the first- and second-order temperature coefficients of resistance. The coupling of these principles establishes a dynamic feedback loop: an increase in applied voltage enhances power dissipation, thereby elevating the temperature and consequently the resistance. This continuous elevation in resistance prevents the current from rising proportionally with voltage, resulting in a distinctly non-linear I-V characteristic with a monotonically decreasing derivative ($dI/dV$).

## Computational & Analytical Methodology
*   **Analytical Error Propagation:** Total absolute uncertainties for derived quantities (e.g., dynamic resistance) were computed via strict Jacobian-based partial derivative propagation, confirming that the fractional uncertainty in resistance operates as the quadrature sum of fractional instrumental uncertainties:
    $$ \Delta R_i = R_i \sqrt{\left(\frac{\Delta V_i}{V_i}\right)^2 + \left(\frac{\Delta I_i}{I_i}\right)^2} $$
*   **Non-Uniform Domain Sampling:** The experimental space was sampled with an intentionally non-uniform density, allocating higher resolution to the extreme voltage regimes ($V > 8\text{ V}$) to explicitly resolve subtle systematic structural deviations in the highly non-linear domain.
*   **Vectorized Stochastic Modeling:** Empirical parameters and their statistical variances were extracted using a fully vectorized Ordinary Least Quares (OLS) algorithm operating within a Monte Carlo framework, bypassing traditional loop bottlenecks to efficiently propagate input variance to polynomial coefficients.
*   **Non-Parametric Residual Diagnostics:** LOWESS (Locally Weighted Scatterplot Smoothing) was applied to the residual vectors to decouple fundamental systematic model-mismatch from stochastic measurement noise.

## Scientific Epistemology & Critical Discussion
The most profound outcome of this analysis is the decisive falsification of the empirical polynomial models, despite their near-unity coefficients of determination ($R^2 > 0.994$). This investigation proves that $R^2$ is an insufficient metric for validating the physical truth of a computational model.

**Critique of Empirical Boundary Conditions:**
The primary failing of the empirical approach is exposed by the physical impossibility of its extrapolated parameters. For the current-voltage relationship, the optimized model predicts a mathematically rigorous but physically absurd non-zero current intercept of $c = (0.43 \pm 0.04)\text{ A}$. The system must strictly adhere to the fundamental boundary condition that zero voltage yields zero current ($I(0) = 0$). This artifact reveals that the quadratic fit is merely a local interpolation tool, thoroughly invalid for extrapolation.

**Falsification via Unphysical Concavity:**
More egregious is the structural failure of the resistance-voltage model. The optimization forces a negative leading coefficient ($p = -0.024 \pm 0.005$), yielding a concave-down parabola. Mathematically, this dictates that the filament's resistance will eventually reach a global maximum and subsequently decrease at higher voltages. This behavior directly contradicts the core physics of tungsten. Under the Stefan-Boltzmann law, radiative heat loss scales aggressively ($P \propto T^4$), necessitating progressively larger power increments to achieve equivalent temperature (and thus resistance) gains. Therefore, the true phenomenological model must be concave-up. 

**The Epistemological Value of Residuals:**
The deliberate, non-uniform sampling strategy successfully resolved the "fingerprint" of this model mismatch. The residuals do not exhibit the zero-mean, normally distributed scatter indicative of random instrumental noise. Instead, they form a clear, wave-like systematic pattern. This structure is not an error in measurement; it is the physical remainder left behind when a complex electro-thermal function is forcibly constrained into a simplified mathematical approximation.

## Repository Structure

To adhere to software engineering standards, the legacy operational scripts have been refactored and consolidated. The updated directory architecture is as follows:

```text
05_electrothermal_dynamics_tungsten/
├── README.md
├── src/
│   └── electrothermal_dynamics_analysis.py
└── figures/
    ├── fig_IV_fit.pdf
    ├── fig_IV_residuals.pdf
    ├── fig_RV_fit.pdf
    └── fig_RV_residuals.pdf
```

*Note: The `figures/` directory is automatically generated or populated during the execution of the primary analytical script.*

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
cd 05_electrothermal_dynamics_tungsten
python src/electrothermal_dynamics_analysis.py
```

Upon execution, the script will:
1. Instantiate the experimental dataset and compute derived quantities (Resistance) with strictly propagated analytical uncertainties.
2. Execute the highly vectorized Monte Carlo pipeline ($N = 100,000$ iterations) to extract polynomial coefficients and their corresponding standard deviations.
3. Render and save double-column formatted, publication-ready I-V and R-V plots (including LOWESS non-parametric residual diagnostics) to the `figures/` directory.
4. Output pipeline execution milestones and completion status to the standard output.

---
**Author:** Dang-Khoa N. Nguyen  
**Institution:** HCMC University of Technology and Engineering