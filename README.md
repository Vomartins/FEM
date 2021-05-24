# FEM
Finite element method solver

This repository has a solver which is in development during the professor's **Maicon Ribeiro Correa** (from *IMECC/UNICAMP*) course of **Finite Element Method**. As the objective of the study here is the finite element method itself the solver's efficiency needs improvement. The developer's native language is *Brazilian Portuguese*, so the scripts have several portuguese words, which'll be changed in the future.

Lately, the solver can solve **1-D ordinary differential equations** of the following type:
> d/dx(a(x) du/dx) + b(x)du/dx + c(x)u = f(x)

on the interval [x_i, x_f] with **Dirichlet boundary conditions**:
> u(x_i) = u_i and u(x_f) = u_f

The user can select which order of the element that will be used to construct the numeric solution (linear, quadratic or cubic).

The ***exemplos.py*** script has some examples with analytical solutions and its derivatives, which can be useful for further developments and understandings.

The ***test.ipynb*** is a simple Jupyter notebook to run the examples and plot the solution.
