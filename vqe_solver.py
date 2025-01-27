import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from scipy.optimize import minimize

def create_ansatz(parameters):
    # Контур анзатца из 4 слоев Ry-Rz-CNOT.
    qr = QuantumRegister(2, 'q')
    circuit = QuantumCircuit(qr)
    
    param_idx = 0
    for layer in range(4):
        # Ry
        circuit.ry(parameters[param_idx], 0)
        circuit.ry(parameters[param_idx + 1], 1)
        param_idx += 2
        
        # Rz
        circuit.rz(parameters[param_idx], 0)
        circuit.rz(parameters[param_idx + 1], 1)
        param_idx += 2
        
        # CNOT
        circuit.cx(0, 1)
    
    circuit.save_statevector()
    
    return circuit

def create_hamiltonian():
    # H = 0.3980Y⊗Z - 0.3980Z⊗I - 0.0113Z⊗Z + 0.1910X⊗X
    
    operators = ['YZ', 'ZI', 'ZZ', 'XX']
    coefficients = [0.3980, -0.3980, -0.0113, 0.1910]
    
    hamiltonian = SparsePauliOp.from_list(list(zip(operators, coefficients)))
    return hamiltonian

def get_expectation(circuit, hamiltonian, backend):
    job = backend.run(circuit)
    result = job.result()
    statevector = result.get_statevector()
    sv = Statevector(statevector)
    
    # Математическое ожидание для каждой строки Pauli
    expectation = 0
    for op, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        op_matrix = SparsePauliOp(op)
        expectation += coeff * sv.expectation_value(op_matrix)
    
    return np.real(expectation)

def objective_function(parameters, circuit, hamiltonian, backend):
    bound_circuit = circuit.bind_parameters(parameters)
    expectation = get_expectation(bound_circuit, hamiltonian, backend)
    return expectation

def find_ground_state():
    backend = AerSimulator(method='statevector')
    
    num_parameters = 16  # 4 слоя * (2 Ry + 2 Rz)
    parameters = [Parameter(f'θ_{i}') for i in range(num_parameters)]
    circuit = create_ansatz(parameters)
    
    hamiltonian = create_hamiltonian()
    
    initial_params = np.random.random(num_parameters) * 2 * np.pi
    
    result = minimize(
        objective_function,
        initial_params,
        args=(circuit, hamiltonian, backend),
        method='COBYLA',
        options={'maxiter': 1000}
    )
    
    return result.fun, result.x

if __name__ == "__main__":
    min_energy, optimal_params = find_ground_state()
    print(f"Minimum eigenvalue found: {min_energy:.6f}")
