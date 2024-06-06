"""LCU Hamiltonian representation"""
from pennylane import numpy as np


class Hamiltonian(object):
    """Representation of a parameterized LCU Hamiltonian consisting of Pauli string operators.
        
        Such a Hamiltonian is a linear combination of Pauli string operators h_i with real coefficients θ_i.
        It is of the form `H = \\sum_i θ_i * h_i` and the θs correspond to the Hamiltonian parameters.

        This class represents H by storing the coefficients θ_i and Pauli string operators h_i in a list of floats and strings, respectively.

        Example: 
        The TFI model H = - J \\sum_{i} σ^x_i σ^x_{i+1} - g \\sum_{i} σ^z_i acting on 3 qubits could be represented by 
        θ = [-J, -J, -g, -g, -g]
        h = ['XXI', 'IXX', 'ZII', 'IZI', 'IIZ']

    Parameters
    ----------
    h, θ :
        Same as in ``Attributes``.

    Attributes
    ----------
    h : list[str]
        List of Pauli string operator representations, where h[i][j] ∈ {'I', 'X', 'Y', 'Z'}.
    θ : list[float] or np.ndarray[dtype=float, ndim=1]
        Hamiltonian parameters and coefficients of the Pauli string operators.
    n_qubits : int
        Number of qubits the Hamiltonian acts on.
    n_params : int
        Number of Hamiltonian parameters.
    """
    
    def __init__(self, h: list[str], θ: list[float] | np.ndarray[float]) -> None:
        self.θ = np.asarray(θ)
        if self.θ.ndim != 1:
            raise ValueError("__init__: θ must have dimension 1.")
        self.h = h
        self.n_qubits = len(h[0])
        self.n_params = len(h)
        if self.n_params != len(self.θ):
            raise ValueError("__init__: h and θ must have same length.")
        
    def θ_norm(self) -> float:
        """Compute the L1-norm of `θ`."""
        return np.linalg.norm(self.θ, ord=1)
    
    def preprocessing(self, δ: float) -> tuple[list[str], np.ndarray[float]]:
        """Preprocess Hamiltonian to scale its spectrum to the interval [`δ`, 1]."""
        if not δ < 1:
            raise ValueError("preprocessing: δ must be smaller than 1.")
        
        h_δ = self.h + [self.n_qubits * 'I']
        θ_δ = np.append(self.θ * (1-δ)/(2*self.θ_norm()), (1+δ)/2)
        return h_δ, θ_δ
    
    def assemble(self, i: int = -1) -> np.ndarray[float | complex]:
        """Assemble the `i`-th Pauli string operator or the full Hamiltonian (default)."""
        if not -1 <= i < self.n_params:
                raise ValueError("assemble: i must be one of {-1,0,..,%r}." % (self.n_params - 1))
        
        def assemble_pauli_string(pauli_string: str):
            σ = {'I': np.array([[1., 0.], [0., 1.]]), 
                 'X': np.array([[0., 1.], [1., 0.]]), 
                 'Y': np.array([[0., -1j], [1j, 0.]]), 
                 'Z': np.array([[1., 0.], [0., -1.]])}
            pauli_string_operator = 1
            for p in pauli_string[::-1]:
                pauli_string_operator = np.kron(σ[p], pauli_string_operator)
            return pauli_string_operator
        
        if i == -1:
            H_operator = 0
            for i in range(self.n_params):
                H_operator = H_operator + self.θ[i] * assemble_pauli_string(self.h[i])
            return H_operator
        else:
            return assemble_pauli_string(self.h[i])