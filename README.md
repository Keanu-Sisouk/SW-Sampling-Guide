# SW-Sampling-Guide

This github repository contains the code used for the experiment on Gaussian Mixture in the paper "A User's Guide to Sampling Strategies for Sliced Optimal Transport".

## Dependencies

To compute the Sliced Wasserstein distance, the library Python Optimal Transport is needed:
[POT library](https://pythonot.github.io)

Other functions come from the scipy and torch libraries.

Regarding the requirements for the code used in the ICML2024 paper: 
"Sliced-Wasserstein Estimation with Spherical Harmonic as Control Variates", 
Rémi Leluc, Aymeric Dieuleveut, François Portier, Johan Segers and Aigerim Zhuman. [Paper](https://arxiv.org/abs/2402.01493),
please see the following repositories:
[SHCV repository](https://github.com/RemiLELUC/SHCV), 
[Spherical Harmonics](https://github.com/vdutor/SphericalHarmonics).

## Running the scripts

In the folder `testSW`, you can run the following command:
```bash
python3 run_python_scripts.py
```
to run all the methods (beware of the running time). Otherwise choose a file with format name `test_*.py` to run.
Accordingly the ouputs are generated in a npy format to be directly used for plotting.
If you want to get the Fig 4.1 of the paper, you can run the file `plotFig4-1Paper.py` (note that you have to run `run_python_scripts.py` beforehand).
<!-- If you only want to get the plot in dimension 3, you can run `plot3D.py`. -->

## Supplementary notes

The scripts `generated_gaussian_toys.py` and `compute_ground_truths.py` should only be ran to generate new data and new ground truths (true SW). Be aware that the second script has a long computation time (~5 days).
To avoid unecessary computation time, the s-Riesz configuration points are already pre-computed and stored in npy format.

