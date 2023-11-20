# Density Simulation for McKean-Vlasov Equations

This repository contains Python programs for simulating the density solution of a McKean-Vlasov equation, associated to the paper:

**Hoffmann, Marc, and Yating Liu. "A statistical approach for simulating the density solution of a McKean-Vlasov equation." arXiv preprint arXiv:2305.06876 (2023).** (https://arxiv.org/abs/2305.06876)

In our paper, we explore our simulation method through three examples:

## 1. Linear Interaction example (Section 3.1 of the paper)

We consider a linear interaction example of the form $\tilde{b}(t, x, y) = c(x−y)$. This case is central to several applications and has moreover the advantage to yield an explicit solution for $μ_t(x)$, enabling us to accurately estimate the error of the simulation (see “*1-McKean-Vlasov SDE with linear interaction*”).

## 2. A double layer potential with a possible singular shock in the common drift (Section 3.2 of the paper)

This example involves a double layer potential with a possible singular shock in the common drift, challenging Assumption 2.3. Although the solution is not explicit, the singularity enables us to investigate automated bandwidth choices on repeated samples, shedding light on the effect of our statistical adaptive method (see “*2-McKean-Vlasov SDE with double layer potential*”). 

## 3. Burgers Equation in Dimension d = 1 (Section 3.3 of the paper)

Although not formally within the reach Assumption 2.3, we may still implement our method. While we cannot provide with theoretical guarantees, we again have an explicit solution for $μ_t(x)$ and we can accurately measure the performance of our simulation method (see “*3-Burger equation*”). 

## Usage:

To run the simulations for each example, follow the instructions in the respective directories (`1-McKean-Vlasov SDE with linear interaction`, `2-McKean-Vlasov SDE with double layer potential`, `3-Burger equation`). In each directory, you'll find a Jupyter notebook containing Python code and a .py file alongside simulated particle systems designed for computational purposes.

## References:

Please cite the associated paper if you use or find the code helpful in your research.

## License:

This code is released under the [Creative Commons Zero v1.0 Universal license](LICENSE). You are free to use, modify, and distribute it without any restrictions.

---
