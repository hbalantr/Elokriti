# EloKriti

Chemistry-aware active learning for the discovery of BPA-free dental resin
backbones using molecular-dynamics-derived network properties.

This repository contains the active-learning and multi-objective
Bayesian-optimization code used in:

**Accelerated Discovery of BPA-Free Dental Resin Candidates through
Active Learning of Network-Emergent Properties**

<p align="center">
  <img src="images/AL_Workflow_Final.png" width="900">
</p>


## Overview

EloKriti uses Gaussian-process surrogate models to prioritize unevaluated
candidate backbones for explicit virtual-curing and molecular-dynamics
evaluation.

The three optimization objectives are:

maximize elastic modulus;

minimize volumetric shrinkage;

minimize fractional free volume.

The surrogate models are used only for candidate selection. Final property
values and Pareto assignments are based on explicit molecular-dynamics
evaluations.

## Repository structure

EloKriti/
├── polymers_bo/
│   ├── data.py
│   ├── surrogate.py
│   ├── bo.py
│   └── suggest.py
├── scripts/
│   └── suggest_next.py
├── images/
│   └── AL_Workflow_Final.png
├── README.md
└── LICENSE

Requirements

The Bayesian-optimization workflow requires:

numpy
pandas
scipy
scikit-learn

Install using:

pip install numpy pandas scipy scikit-learn

Input files

The candidate-pool CSV should contain:

reaction_id
reactant_1
molecular_weight_Boltzmann_average
logP_Boltzmann_average
TPSA_Boltzmann_average
normalized_monomer_phi_Boltzmann_average
normalized_backbone_phi_Boltzmann_average

The evaluated-candidate CSV should contain the same columns plus:

elastic_modulus
volumetric_shrinkage
ffv

Only successfully evaluated candidates with values for all three objectives
are used to train the surrogate models.

Active-learning configuration

The paper configuration uses:

TF-IDF analyzer          character
n-gram range             2-5
maximum TF-IDF features  12000
SVD components           8
random seed              42

GP kernel                ConstantKernel × Matern(nu=2.5) + WhiteKernel
GP optimizer restarts    1

random scalarizations    64
weight distribution      Dirichlet(1,1,1)
expected-improvement xi  0.01

Objective directions are:

elastic modulus          maximize
volumetric shrinkage     minimize
fractional free volume   minimize

Run candidate selection

From the repository root:

python scripts/suggest_next.py \
  --candidate-csv candidate_pool.csv \
  --evaluated-csv evaluated_clean.csv \
  --output-csv next_candidates.csv \
  --top-k 10 \
  --seed 42 \
  --scalarizations 64

The output CSV contains the highest-ranked unevaluated candidates together
with surrogate predictions, predictive uncertainties, and acquisition scores.

After explicit virtual-curing and molecular-dynamics evaluation, successful
candidate results are added to the evaluated CSV and the acquisition step is
repeated.

Data availability

This repository contains the active-learning and Bayesian-optimization code.

Publication-associated molecular-dynamics inputs, processed results, and
simulation data will be separately in a permanent research-data
repository.

Citation

If you use this code, please cite:

Accelerated Discovery of BPA-Free Dental Resin Candidates through
Active Learning of Network-Emergent Properties

The final journal citation and DOI will be added after publication.

License

MIT License. See LICENSE.
