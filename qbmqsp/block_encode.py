"""Unitary block encoding"""
import pennylane as qml
from pennylane import numpy as np

from .hamiltonian import Hamiltonian
from .utils import int_to_bits


class BlockEncode(object):
    """Unitary block encoding of a LCU Hamiltonian consisting of Pauli string operators.

    Either a general encoding with one auxiliary qubit or the hardware-native LCU method is performed.

    Parameters
    ----------
    h, θ :
        See qbmqsp.hamiltonian.Hamiltonian
    enc :
        Same as in ``Attributes``.

    Attributes
    ----------
    H : qbmqsp.hamiltonian.Hamiltonian
        Hamiltonian constructed from parameters (`h`, `θ`).
    enc : {'general', 'lcu'}
        Unitary block encoding scheme.
    """

    def __init__(self, h: list[str], θ: np.ndarray[float], enc: str) -> None:
        self.H = Hamiltonian(h, θ)
        self.enc = enc
        if enc == 'general':
            # Assemble normalized Hamiltonian operator to be block encoded
            self._H_operator_normalized = self.H.assemble() / self.H.θ_norm()
        elif enc == 'lcu':
            # Transform Hamiltonian parameters to perform amplitude encoding
            self._θ_amplitude = np.sqrt( np.abs( θ/self.H.θ_norm()))
        else:
            raise ValueError("__init__: enc must be one of %r." % {'general', 'lcu'})
        
    def n_qubits(self, registers: str | set[str] = None) -> int:
        """Return total number of qubits used in `registers`.
        
        Parameters
        ----------
        registers : str | set[str]
            Quantum registers whose number of qubits should be returned.
            Must be an element from or a subset of {'enc', 'sys'}.

        Returns
        -------
        n : int
            Total number of qubits used in `registers`.
        """
        if registers is None:
            registers = {'enc', 'sys'}
        elif type(registers) == str:
            registers = {registers}
        if not registers.issubset({'enc', 'sys'}):
            raise ValueError("n_qubits: registers must be an element from or a subset of %r." % {'enc', 'sys'})
        
        n = 0
        for register in registers:
            if register == 'enc':
                if self.enc == 'general':
                    n += 1
                elif self.enc == 'lcu':
                    n += int( np.ceil( np.log2( self.H.n_params)))
            elif register == 'sys':
                n += self.H.n_qubits
        return n

    def _prepare_gate(self, wires: list[int]) -> None:
        """Apply the PREPARE gate on `wires`"""
        qml.AmplitudeEmbedding(self._θ_amplitude, wires, pad_with=0)

    def _prepare_gate_inverse(self, wires: list[int]) -> None:
        """Apply the inverse of the PREPARE gate on `wires`"""
        qml.adjoint(qml.AmplitudeEmbedding(self._θ_amplitude, wires, pad_with=0))

    def _pauli_string_gate(self, pauli_string: str, π_phase: bool, wires: list[int]) -> None:
        """Apply the Pauli string gate of `pauli_string` on `wires`.
        
        Parameters
        ----------
        pauli_string : str
            Pauli string representation in terms of {'I', 'X', 'Y', 'Z'}.
        π_phase : bool
            If True, apply relative phase factor of -1.
        wires : list[int]
            Wires to apply the gate on.
        """
        if len(pauli_string) != len(wires):
            raise ValueError("pauli_string_gate: pauli_string and wires must have same length.")
        
        for wire, string in zip(wires, pauli_string):
            if string == 'X':
                qml.X(wire)
            elif string == 'Y':
                qml.Y(wire)
            elif string == 'Z':
                qml.Z(wire)
            elif string != 'I':
                raise ValueError("pauli_string_gate: pauli_string must be composed of %r." % {'I', 'X', 'Y', 'Z'})
            
        if π_phase:
            qml.RZ(2*np.pi, wires[0])

    def _select_gate(self, enc_wires: list[int], sys_wires: list[int]) -> None:
        """Apply the SELECT gate on encoding wires `enc_wires` and system wires `sys_wires`."""
        for i in range(self.H.n_params):
            qml.ctrl(self._pauli_string_gate, 
                     control=enc_wires, 
                     control_values=int_to_bits(i, len(enc_wires))
                     )(self.H.h[i], self.H.θ[i] < 0, sys_wires)

    def _lcu_circuit(self, enc_wires: list[int], sys_wires: list[int]) -> None:
        """Apply the LCU circuit on encoding wires `enc_wires` and system wires `sys_wires`."""
        self._prepare_gate(enc_wires)
        self._select_gate(enc_wires, sys_wires)
        self._prepare_gate_inverse(enc_wires)

    def circuit(self, enc_wires: list[int], sys_wires: list[int]) -> None:
        """Apply block encoding circuit.
        
        Parameters
        ----------
        enc_wires : list[int]
            Wires used of for block encoding.
        sys_wires: list[int]
            Wires on which the Hamiltonian acts on.
        """
        if self.n_qubits('enc') != len(enc_wires):
            raise ValueError("circuit: length of enc_wires must match number of encoding qubits (=%r)." % self.n_qubits('enc'))
        if self.n_qubits('sys') != len(sys_wires):
            raise ValueError("circuit: length of sys_wires must match number of system qubits (=%r)." % self.n_qubits('sys'))

        if self.enc == 'general':
            qml.BlockEncode(self._H_operator_normalized, enc_wires + sys_wires)
        elif self.enc == 'lcu':
            self._lcu_circuit(enc_wires, sys_wires)