# Cardinality Preservering Variational Quantum Algorithm (CP-VQA) 

In this repo. can be found statevector simulation implementations for CP-VQA and QAOA. Both are built around ABC classes - the structure should be fairly intuitive.
Both classes implements get_cost(), set_circuit(), get_statevector() and callback() (for storing cost values and ground state probabilities during optimization).

Both classes are implemented in both Qiskit, Qulacs and Qsim (Cirq). The Qiskit impl. of CP-VQA, can handle both excact simulation of the hamiltonian (H) and the approximate simulation, where H is approximated by a sequence of Rxx and Ryy gates (higher order terms due to non-commutativity are neglected), while the Qulacs and Qsim (Cirq) implementations only handle the approximate version of H. 

The Qsim implementation also comes with the "get_full_state_vector" attr., which, if set to False, ensures that only the binom(N,k) indices in the statevector corresponding to states w. ||state||_0=k are sampled - making it the efficient choice for larger systems, while Qulacs are the fastest for < 15 (ish) qubit systems - see figure in script folder.

In script/FULL_H_vs_TROTTERIZED_H.ipynb can be found a full example of how to use in conjunction w. scipy, and joblib for parallelization.

### Installation ###
##### MacOS/Unix: #####
env/bin/python -m pip install -r requirements.txt

##### Windows: #####
env\bin\python -m pip install -r requirements.txt

N.B. See [pip 'freeze' documentation](https://pip.pypa.io/en/stable/cli/pip_freeze/) for detailed explanation. 