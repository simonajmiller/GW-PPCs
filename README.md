[![Static Badge](https://img.shields.io/badge/arXiv-2604.06090-%23B31B1B)]( 	
https://doi.org/10.48550/arXiv.2604.06090)


# `GW-PPCs`
This repository accompanies **["Posterior Predictive Checks for Gravitational-wave Populations: Limitations and Improvements" by Miller et. al 2026]([url](https://arxiv.org/abs/2604.06090)).** 

The repository is organized as follows: 
* `figures` contains notebooks to reproduce all figures in the paper (aside from the schematics in Figs 2 and 3)
* `data` contains relevant data needed to reproduce figures
* `utils` includes some basic utilities for figure generation
* `toy_example` contains a notebook walking through an analytic toy model example for event vs. data-level posterior predictive checks

We also direct readers to [`measuring-bbh-component-spin`](https://github.com/simonajmiller/measuring-bbh-component-spin) and for additional relevant data from [Miller et. al 2024](https://arxiv.org/abs/2401.05613), and [Zenodo](zenodo.org/records/16911563) for the GWTC-4.0 Rates & Populations data release. 

If you use any of the code or data herein, we request that you cite Miller et. al 2026:
```bibtex
@article{Miller:2026buq,
    author = "Miller, Simona J. and Winney, Sophia and Chatziioannou, Katerina and Meyers, Patrick M.",
    title = "{Posterior Predictive Checks for Gravitational-wave Populations: Limitations and Improvements}",
    eprint = "2604.06090",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P2600144",
    month = "4",
    year = "2026"
}
```
