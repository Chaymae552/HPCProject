# HPCProject

This repository contains the code and experiments for the High Performance Computing project.  
The goal is to improve and parallelize a simple **Multi-Layer Perceptron (MLP)** implemented in C, trained on a synthetic 2D classification dataset, and to study how different strategies affect performance, runtime, and scalability.

The project includes:

- **Memory debugging and profiling** (Valgrind, Callgrind, KCachegrind)
- **Training improvements** (mini-batch SGD, dynamic learning-rate schedules, multiple activations)
- **OpenMP** (shared-memory parallelism)
- **MPI + OpenMP** (hybrid distributed/shared-memory parallelism)

All experiments were run on the **Toubkal** HPC cluster.
