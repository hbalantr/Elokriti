# EloKriti

Chemistry-aware active learning and virtual curing for the discovery of
BPA-free dental resin backbones using molecular-dynamics-derived
network properties.

This repository contains the code used for the active-learning and
multi-objective Bayesian-optimization workflow described in:

**Accelerated Discovery of BPA-Free Dental Resin Candidates through
Active Learning of Network-Emergent Properties**

<p align="center">
  <img src="images/AL_Workflow_Final.png" width="900">
</p>

## Overview

EloKriti is a chemistry-aware active-learning workflow developed to
prioritize BPA-free dental resin backbone candidates for explicit
virtual-curing and molecular-dynamics evaluation.

The workflow combines molecular representations, Gaussian-process
surrogate models, and multi-objective Bayesian optimization to reduce
the number of computationally expensive molecular-dynamics evaluations
required during candidate screening.

Candidate structures are represented using reactant SMILES together
with five molecular descriptors. Independent Gaussian-process models
are trained for three cured-network properties obtained from explicit
molecular-dynamics calculations:

- elastic modulus;
- volumetric shrinkage;
- fractional free volume.

The active-learning objectives are:

```text
Elastic modulus          maximize
Volumetric shrinkage     minimize
Fractional free volume   minimize

Surrogate predictions are used only to prioritize unevaluated
candidates for subsequent simulation. The reported property values and
Pareto assignments in the associated study are based on explicit
virtual-curing and molecular-dynamics evaluations.

EloKriti/
├── README.md
├── LICENSE
│
├── polymers_bo/
│   ├── __init__.py
│   ├── data.py
│   ├── surrogate.py
│   ├── bo.py
│   └── suggest.py
│
├── scripts/
│   └── suggest_next.py
│
└── images/
    └── AL_Workflow_Final.png
