# Synthetic Data Experiments

This folder contains the synthetic-data experiments used to evaluate two competing models:

- Mixture of cylindrical distributions
- Two-layer hierarchical mixture model (mixture of mixtures, MoM)

Both experiments are run under clean and noisy settings to assess robustness to model deviations.

## Experiment 01: Cylindrical Mixture as Data-Generating Process

### Objective

Assess how well each model recovers structure when data are generated from a finite mixture of cylindrical distributions (correct specification for the cylindrical model), and evaluate robustness under controlled noise.

### Design

**Data generation**

Synthetic datasets are sampled from a cylindrical mixture with predefined:

- Mixture weights
- Gaussian means and conditional covariances
- Cross-covariance structure
- Directional means and concentration parameters

Some components are intentionally configured to share Gaussian means while differing in covariance geometry and/or directional structure to produce non-trivial clustering scenarios.

**Noise robustness**

Robustness is assessed with controlled perturbations by adding Gaussian noise on the Euclidean component

**Evaluation Metric**

Bayesian Information Criterion and unpenalized log-likelihood are recorded at this stage.

### Interpretation goals

- Can the cylindrical mixture recover its own generating parameters?
- Can the hierarchical model approximate a cylindrical structure?
- How stable are both models under moderate noise?

## Experiment 02: Hierarchical Mixture (MoM) as Data-Generating Process

### Objective

Evaluate model behavior when data are generated from a hierarchical latent structure (correct specification for MoM, misspecification for the cylindrical mixture).

### Design

We retain the same design, only changing the data generation: Synthetic datasets are sampled from a two-layer hierarchy:

- First latent layer selects a group
- Second latent layer selects a subcomponent within that group
- Observations are generated conditionally on both layers

Component distributions are cylindrical, allowing Euclidean and directional variability at each level.

### Interpretation goals

- How well does MoM recover multi-level latent structure?
- To what extent can a cylindrical mixture approximate hierarchical data?
- What is the trade-off between flexibility and interpretability?

## Folder contents

- `01_cylindrical_mixture.ipynb`: Experiment 01 workflow and analyses
- `02_hierarchical_mixture.ipynb`: Experiment 02 workflow and analyses
- `experiment_helper.py`: Shared helper functions for synthetic experiments
