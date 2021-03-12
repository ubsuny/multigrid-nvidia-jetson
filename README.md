# GPU/CPU Multigrid Comparison for the "Good" Helmholtz for NVIDIA Jetson

### Project Goals:

The goal of this work is to write a GPU and CPU multigrid (GMG or AMG) code for the "Good" Helmholtz equation,

$$ -\Delta u + u = f.$$

Both AmgX and PETSc have working multigrid solvers and are of interest for this project.  While GPUs have potential for major performance increases in multigrid methods, it is expected that GPU performance will only exceed CPU performace at very large scales.

[AmgX](https://developer.nvidia.com/amgx) - AmgX is a GPU accelerated core solver library that speeds up computationally intense linear solver portion of simulations.  AmgX is a NVIDIA Registered Developer Program.


[PETSc](https://www.mcs.anl.gov/petsc/index.html) - PETSc is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations. It supports MPI, and GPUs through CUDA or OpenCL, as well as hybrid MPI-GPU parallelism. 

Useful Resources:
[Dave May Extreme Scale Multigrid Components Within PETSc](https://arxiv.org/pdf/1604.07163.pdf)
