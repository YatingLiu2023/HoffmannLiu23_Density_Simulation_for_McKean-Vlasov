# Density Simulation for McKean-Vlasov Equations

This repository contains Python programs for simulating the density solution of a McKean-Vlasov equation, as detailed in the paper:

**Hoffmann, Marc, and Yating Liu. "A statistical approach for simulating the density solution of a McKean-Vlasov equation." arXiv preprint arXiv:2305.06876 (2023).**

In our paper, we explore our simulation method through three distinct examples:

## 1. Linear Interaction (Assumption 2.3)

We consider a linear interaction of the form eb(t, x, y) = c(x−y). This case is central to various applications, as highlighted in the references in the introduction. Moreover, it provides an explicit solution for μt(x), allowing us to accurately estimate the simulation error.

## 2. Double Layer Potential with Singular Shock

This example involves a double layer potential with a possible singular shock in the common drift, challenging Assumption 2.3. Although the solution is not explicit, the singularity enables us to investigate automated bandwidth choices on repeated samples, shedding light on the effect of our statistical adaptive method. Additional details and results can be found at [bit.ly/3yY332B](bit.ly/3yY332B) and [bit.ly/3ZnvhPj](bit.ly/3ZnvhPj).

## 3. Burgers Equation in Dimension d = 1

While not formally within the reach of Assumption 2.3, we implement our method for the Burgers equation in dimension d = 1. Although theoretical guarantees are not provided, we obtain an explicit solution for μt(x), allowing us to accurately measure the performance of our simulation method.

## Usage:

To run the simulations for each example, follow the instructions in the respective directories (e.g., `linear_interaction`, `double_layer_potential`, `burgers_equation`). Detailed usage and parameter information is available in the documentation within each directory.

## References:

Please cite the associated paper if you use or find the code helpful in your research.

## License:

This code is released under the [Creative Commons Zero v1.0 Universal license](LICENSE). You are free to use, modify, and distribute it without any restrictions.

For any questions or issues, please open an [issue](https://github.com/yourusername/your-repository/issues).

---
