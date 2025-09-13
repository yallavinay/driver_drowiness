# src/models/qml_hybrid.py
"""
Minimal hybrid model: classical encoder -> small quantum layer (PennyLane) -> output
This is a proof-of-concept; it uses default.qubit simulator for development.
"""

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from pennylane.qnn import TorchLayer

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # inputs has shape [batch_size, n_qubits]
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)   # broadcast rotation across batch
    qml.templates.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (2, n_qubits, 3)}  # 2 variational layers
qlayer = TorchLayer(quantum_circuit, weight_shapes)
 # QNN DEcisiomn
class HybridNet(nn.Module):
    def __init__(self, input_dim=12288, n_qubits=n_qubits):
        super().__init__()
        # Fix: input_dim should match 64x64x3 = 12,288 features from dataset
        self.fc = nn.Linear(input_dim, n_qubits)
        self.ql = qlayer
        self.out = nn.Linear(n_qubits, 2)
    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.ql(x)
        x = self.out(x)
        return x