# QBMQSP (Βeta version)

An implementation of a fully-visible quantum Boltzmann machine (QBM) based on quantum signal processing.

> **Methods**: Quantum Botlzmann Machine, Quantum Machine Learning, Quantum Signal Processing, Quantum Eigenvalue Transform, Unitary Block Encoding.

The training of QBMs involves the evaluation of Gibbs state expectation values, which is computationally intractable for classical computers due to the exponentially growing Hilbert space. 
However, this could be remedied through the usage of a quantum computer. 
One approach is to prepare a QBM on a quantum computer by leveraging the quantum eigenvalue transform (QEVT), a framework for realizing matrix polynomials on a quantum computer based on quantum signal processing (QSP).

This project implements a numerical simulation of the QEVT and employs it as a subroutine for the training of a QBM.
The necessary QSP phase factors are solved by interfacing with QSPPACK[^1].
For the unitary block encoding of the QBM Hamiltonian, either a general unitary block encoding or the hardware-compatible LCU[^2] method is performed.


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
      <ul>
        <li><a href="#training-a-qbm">Training a QBM</a></li>
        <li><a href="#preparing-a-qbm">Preparing a QBM</a></li>
      </ul>
    </li>
    <li><a href="#structure">Structure</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#executing-program">Executing program</a></li>
      </ul>
    </li>
    <li><a href="#help">Help</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#version-history">Version History</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>


--------------------------------------------------------
## Description <a name="description"></a>

### Training a QBM <a name="training-a-qbm"></a>
A fully-visible QBM is a machine learning model of the form of a variational Gibbs state, 

$$ \rho\_\theta = \frac{ e^{-\beta \hspace{0.05cm} \mathcal{H}\_\theta} }{ \text{Tr}\[ e^{-\beta  \hspace{0.05cm} \mathcal{H}\_\theta} \] } \hspace{0.1cm}, $$

where $\mathcal{H}\_\theta$ is a Hamiltonian parameterized by variational parameters $\theta$, acting on $n$ qubits. 
The constant inverse temperature $\beta$ could in principle also be absorbed into $\mathcal{H}\_\theta$. 
We restrict ourselves to representations of the form 

$$ \mathcal{H}\_\theta = \sum\_i \theta\_i h\_i  \hspace{0.1cm}, $$

where $\theta\_i \in \mathbb{R}$ and $h\_i = {\bigotimes}_{j=0}^{n-1} \hspace{0.1cm} \sigma\_{j,i}$ with $\sigma\_{j,i} \in \\{ I, X, Y, Z \\}$. 
A QBM can be used for generative modeling of classical and quantum data. 
To that end, the QBM is usually optimized to fit some target quantum state $\chi$, which could encode some classical probability distribution or correspond to some quantum mechanical physical system of interest.
To that end, the QBM is trained by minimizing the quantum relative entropy[^3], 

$$ S( \chi \hspace{0.1cm} \Vert \hspace{0.1cm} \rho\_{\theta} ) = \text{Tr}\[ \chi \hspace{0.1cm} \text{log} \hspace{0.1cm} \chi \] - \text{Tr}\[ \chi \hspace{0.1cm} \hspace{0.05cm} \text{log} \hspace{0.1cm} \rho\_\theta \] \hspace{0.1cm}, $$ 

which is stricly convex[^4] and zero if and only if $\chi = \rho\_{\theta}$. 
$S$ is usually minimized via gradient descent and the gradient is given by the difference of expectation values of $\frac{\partial \hspace{0.05cm} \mathcal{H}\_\theta}{\partial\theta}$ between the target and the model density matrix, which for the above Hamiltonian is given by

$$ \frac{ \partial S( \chi \hspace{0.1cm} \Vert \hspace{0.1cm} \rho\_{\theta})}{\partial\theta\_i} = \beta \hspace{0.1cm} ( \hspace{0.05cm} \langle{ h\_i }\rangle\_{\chi} - \langle{ h\_i }\rangle\_{\rho\_\theta} \hspace{0.05cm} ) \hspace{0.1cm}. $$

Hence, the training depends on the ability to prepare and measure a QBM. 
To that end, this project utilizes the framework of QEVT.


### Preparing a QBM <a name="preparing-a-qbm"></a>
As a mixed quantum state, the QBM is not prepared directly, but rather a purification of it. 
Given an $n$-qubit QBM, we define two $n$-qubit registers, the system register $S$ and the environment register $E$, to prepare $\ket{\phi^{+}}\_{{S, E}}^{\otimes n} \hspace{0.1cm},$ where $\ket{\phi^{+}}\_{{S, E}} = \frac{1}{\sqrt{2}} (\ket{00} + \ket{11})$ denotes the maximally entangled state with the first qubit in $S$ and and the second qubit in $E$. 
We then apply $V\_\theta = e^{- \frac{\beta}{2} \hspace{0.05cm} \mathcal{H}\_\theta}$ and perform measurements in $S$ but not in $E$, i.e. we trace out $E$, resulting in QBM statistics:

$$ \text{Tr}\_{E} \[ \hspace{0.2cm} V\_\theta \hspace{0.2cm} \ket{\phi^{+}}\bra{\phi^{+}}\_{S,E}^{\otimes n} \hspace{0.2cm} V\_\theta^{\dagger} \hspace{0.2cm} \] = \rho\_\theta \hspace{0.1cm}. $$

To apply the non-unitary, imaginary-time evolution operator $V\_\theta = e^{- \frac{\beta}{2} \hspace{0.05cm} \mathcal{H}\_\theta}$ we utilize QEVT. As a first ingredient we perform a unitary block encoding of the Hamiltonian, which requires a third register, the auxiliary register $A$. This project implements the general encoding scheme and the hardware-compatible LCU method. As a second ingredient we need to find the QSP phase factors to realize $e^{- \frac{\beta}{2} x}$ on a quantum computer.
However, since the QSP theorem only allows for realizing polynomials of definite parity, we compute the QSP phase factors $\varphi$ for a polynomial approximation of the even function $f\_\tau (x) = e^{- \tau \hspace{0.05cm} |x|}$, for some $\tau \in \mathbb{R}$. This is achieved by interfacing with QSPPACK, which finds a polynomial approximation of $f\_\tau$ on an interval $\[\delta, \hspace{0.05cm} 1\]$ for some tunable parameter $\delta \in (0, \hspace{0.05cm} 1)$.

Hence, to realize $V\_\theta$, we have to scale the spectrum of $\mathcal{H}\_\theta$ to the interval $\[\delta, \hspace{0.05cm} 1\]$. 
Since $\mathcal{H}\_\theta$ is a linear combination of Pauli string operators, which have eigenvalues $\pm 1$, the operator norm is bounded by $\lVert \mathcal{H}\_\theta \rVert \le \lVert\theta\rVert\_1$. Therefore, the preprocessing

$$ \mathcal{H}\_\theta^\delta = \frac{ \mathcal{H}\_\theta + \lVert\theta\rVert\_1 \mathcal{I} } { 2 \lVert\theta\rVert\_1 } (1-\delta) + \delta \hspace{0.1cm} \mathcal{I} \hspace{0.1cm} $$

results in $\text{spec}( \mathcal{H}\_\theta^{\delta} ) \subset \[\delta, \hspace{0.05cm} 1\]$. 

By computing $\varphi$ for $f\_\tau$ with $\tau = \frac{ \hspace{0.1cm} \beta \hspace{0.05cm} \lVert\theta\rVert\_1}{1-\delta}$ and employing QEVT, we are able to implement a unitary $U\_\varphi$ acting on $A$ and $S$ such that

$$ {}\_A\bra{0}  U\_\varphi  \ket{0}\_A = f\_\tau(\mathcal{H}\_{\theta}^{\delta}) = e^{- \tau  \hspace{0.05cm} \mathcal{H}\_{\theta}^{\delta}} = e^{- \beta \frac{1+\delta}{1-\delta} \lVert\theta\rVert\_1} \hspace{0.2cm} V\_\theta  \hspace{0.1cm}. $$

Hence, by preparing the state $\ket{0}\_{A} \otimes \ket{\phi^{+}}\_{{S, E}}^{\otimes n}$ , applying $U\_\varphi$ on $A$ and $S$ and measuring $\ket{0}\_A$ in $A$, performing any measurement on system $S$ but not $E$, results in the statistics of a QBM:

$$ \text{Tr}\_{E}\[ \hspace{0.2cm} {}\_A\bra{0} \hspace{0.2cm} U\_\varphi \hspace{0.2cm} ( \hspace{0.1cm} \ket{0}\bra{0}\_A \otimes \ket{\phi^{+}}\bra{\phi^{+}}\_{{S, E}}^{\otimes n} \hspace{0.1cm} ) \hspace{0.2cm} U\_\varphi^{\dagger} \hspace{0.2cm} \ket{0}\_A  \hspace{0.2cm} \] \hspace{0.2cm} \sim \hspace{0.2cm} \rho\_\theta \hspace{0.1cm} .$$

The normalization factor is the trace of the LHS and equivalent to the success probability of measuring $\ket{0}\_A$ in $A$, $p_0 = \text{Tr}\[ e^{- \beta \mathcal{H}\_\theta} \] \hspace{0.1cm} 2^{-n} \hspace{0.1cm} e^{- \beta \frac{1-\delta}{1+\delta} \lVert\theta\rVert\_1} \hspace{0.1cm}$, which can be increased by employing amplitude amplification in system $A$.


--------------------------------------------------------
## Structure

``` bash
├── README.md
├── env.yml
├── qbmqsp
│   ├── __init__.py
│   ├── block_encode.py
│   ├── hamiltonian.py
│   ├── phaseangles_qbm.m
│   ├── qbm.py
│   ├── qevt.py
│   ├── qsp.py
│   ├── qsp_phase_engine.py
│   ├── rel_ent.py
│   └── utils.py
└── tests
    ├── __init__.py
    ├── gen_data.py
    ├── test_block_encode.ipynb
    ├── test_qbm.ipynb
    ├── test_qevt.ipynb
    ├── test_qsp.ipynb
    └── test_rel_ent.ipynb
```

--------------------------------------------------------
## Getting Started

### Dependencies

| **Dependency** | **Version** |
| -------------- | ----------- |
 | [python]() | 3.11.0 |
 |        [ipykernel](https://pypi.org/project/ipykernel/) | 6.29.4 |
 |       [matlabengine](https://pypi.org/project/matlabengine/) | 24.1.2 |
 |       [matplotlib](https://pypi.org/project/matplotlib/) | 3.9.0 |
 |       [networkx](https://pypi.org/project/networkx/) | 3.3 |
 |       [pennylane](https://pennylane.ai/install/) |  0.36.0 |
 |       [scipy](https://pypi.org/project/scipy/) | 1.13.1 |
 | [MATLAB](https://de.mathworks.com/products/new_products/latest_features.html) | R2024a |
 |       [chebfun](https://www.chebfun.org/download/) |  |
 |       [CVX](https://cvxr.com/cvx/doc/install.html) |  |
 |       [QSPPACK](https://github.com/qsppack/QSPPACK) |  |
 

### Installation

1. Clone the repository:
```bash
 git clone https://github.com/danielortmann/qbmqsp.git
```

2. Install MATLAB release R2024a.
3. Install QSPPACK: https://github.com/qsppack/QSPPACK
4. and its dependencies: _chebfun_ and _CVX_.
5. Make sure the toolbox and the file `qbmqsp/phaseangles_qbm.m` are on the MATLAB path.
6. Install the required Python virtual environment:
```bash
conda env create -n qbmqsp -f env.yml
```

Next, install the repository as a Python package in development mode into the previously created virtual environment:

7. First, install conda-build in the conda `base` environment:
```bash
conda activate base
conda install conda-build
```

8. Display the path to the virtual environment _qbmqsp_:
```bash
conda env list
```

9. Then run `conda-develop` and specify the path to the repository and the name of the virtual environment:
```bash
conda-develop <path-to-repo> -n qbmqsp
```

10. Activate the virtual environment:
```bash
conda activate qbmqsp
```
    The source files can now be imported into a Python session as any other package, namely via `import qbmqsp`.


### Executing program

...


--------------------------------------------------------
## Help

...


--------------------------------------------------------
## Authors

Daniel Ortmann


--------------------------------------------------------
## Version History

...


--------------------------------------------------------
## License

...


--------------------------------------------------------
## Acknowledgments

...

--------------------------------------------------------
## Reference

[^1]: Dong, Y., Meng, X., Whaley, K.B. and Lin, L., 2021. Efficient phase-factor evaluation in quantum signal processing. Physical Review A, 103(4), p.042419.

[^2]: Childs, A. M. and Wiebe, N., 2012. Hamiltonian simulation using linear combinations of unitary operations. Quantum Information & Computation, 12(11-12), pp. 901–924.

[^3]: Kieferova, M. and Wiebe, N., 2017. Tomography and generative training with quantum Boltzmann machines. Physical Review A, 96(6), p.062327.

[^4]: Coopmans, L. and Benedetti, M., 2024. On the Sample Complexity of Quantum Boltzmann Machine Learning. arXiv:2306.14969 [quant-ph].
