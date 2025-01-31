# SW-Guide

This github repository contains the code used for the experiment on Gaussian Mixture in the paper "A User's Guide to Sampling Strategies for Sliced Optimal Transport".

## Dependencies

To compute the Sliced Wasserstein distance, the library Python Optimal Transport is needed:
[POT library](https://pythonot.github.io)

Other functions come from the scipy library.

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
to run all the methods. Otherwise choose a file with format name `test_*.py` to run.
Accordingly the ouputs are generated in a npy format to be directly used for plotting.


