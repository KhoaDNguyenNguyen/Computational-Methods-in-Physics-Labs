[Return to Main Laboratory Repository](../README.md)

---

# Module 01 | Probing Carrier Transport Mechanisms in High-Power LEDs via Monte Carlo Uncertainty Analysis

## Overview

This module investigates the non-ideal charge carrier transport mechanisms in narrow-bandgap (AlGaInP) and wide-bandgap (InGaN) light-emitting diodes (LEDs). Through the numerical extraction of phenomenological diode parameters - specifically the ideality factor ($n$), reverse saturation current ($I_s$), and series resistance ($R_s$) - the physical origins of non-radiative recombination, tunneling phenomena, and resistive losses are evaluated. 

To overcome the inherent limitations of deterministic curve fitting, a highly parallelized, weighted Monte Carlo simulation framework is employed. This methodology rigorously propagates experimental measurement uncertainties to ascertain statistically robust bounds for all extracted physical parameters.

## Computational Methodology

The analytical suite is partitioned into three primary computational workflows:

1. **Nonlinear Carrier Transport Modeling (AlGaInP)**
   Evaluates the I-V characteristics of Red LEDs. Initial parameter bounds are derived analytically via linear derivative fitting in the high-current regime. The optimization is executed utilizing the Levenberg-Marquardt algorithm coupled with $10^4$ Monte Carlo iterations to establish parameter confidence intervals.

2. **Wide-Bandgap Boundary Adaptation (InGaN)**
   Adapts the aforementioned numerical model for Blue LEDs. Due to elevated trap-assisted tunneling currents inherent to InGaN heterostructures, the non-ideal diode equation is constrained using bounded non-linear least squares, preventing algorithmic divergence typically caused by sub-nanoampere saturation currents.

3. **Digital Signal Processing for Local Ideality Factor $n(I)$**
   Implements a discrete analysis of the local ideality factor over varying injection currents. To mitigate numerical noise amplification inherent in discrete differentiation, experimental voltage arrays are subjected to a Savitzky-Golay filter prior to the computation of $dV/dI$. This permits the precise identification of transitional regimes between Sah-Noyce-Shockley recombination ($n \approx 2$) and ideal diffusion ($n \approx 1$).

## Software Architecture

The Python codebase is engineered in strict adherence to software engineering standards for scientific computing:
- **Object-Oriented Integrity**: Computational logic is encapsulated within specific classes (`DiodeExperiment`, `MonteCarloFitter`), eliminating global state variables and ensuring thread safety during multiprocessing operations.
- **Computational Efficiency**: Core physics equations are vectorized and compiled Just-In-Time (JIT) utilizing the `numba` library, significantly reducing Monte Carlo execution latency.
- **Publication-Standard Outputs**: All graphical outputs are generated programmatically without interactive blocking. Matplotlib configuration parameters (`rcParams`) are enforced globally to utilize the Computer Modern serif font, strict LaTeX rendering, and precise dimension scaling suitable for double-column high-impact journals.

## Repository Structure

```text
.
├── figures/                            # High-resolution analytical plots (PDF)
├── src/
│   ├── algainp_transport_analysis.py   # Monte Carlo fitting for Red LED
│   ├── ingan_transport_analysis.py     # Monte Carlo fitting for Blue LED
│   └── local_ideality_dsp.py           # SG-filtering and numerical differentiation
├── report_led_carrier_transport.pdf    # Comprehensive laboratory report
└── README.md
```

## Execution Instructions

Ensure that the required scientific libraries (`numpy`, `scipy`, `matplotlib`, `statsmodels`, `joblib`, and `numba`) are installed in the computational environment. Execute the analytical scripts directly from the terminal. 

For instance, to initiate the AlGaInP transport analysis:

```bash
python3 src/algainp_transport_analysis.py
```

Upon execution, the scripts will parallelize the Monte Carlo iterations across all available logical CPU cores, output numerical parameters to the standard output (with associated standard errors), and compile the diagnostic plots into the `figures/` directory. Cached numpy archives (`.npz`) are automatically generated to bypass redundant computations in subsequent executions.