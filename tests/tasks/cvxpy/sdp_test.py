import numpy as np

import cvxpy as cp

# Generate a random SDP.
n = 40
k = 3

b = np.random.rand(n, n)
b_symm = (b + b.T) / 2

np.fill_diagonal(b_symm, 0)
b_symm = np.abs(b_symm)

adj = np.where(b_symm != 0, np.ones_like(b_symm), np.zeros_like(b_symm))

adj_param = cp.Parameter((n, n))
edges = cp.Variable((n, n), PSD=True)
edge_weight_param = cp.Parameter((n, n))

selected = np.ones_like(adj) * -1
# print(adj[0] * b_symm)


# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
# The operator >> denotes matrix inequality.
constraints = [
    cp.sum(cp.multiply(cp.reshape(adj_param[i], (1, n)), edges)) == 4 * 3 - 2 * n
    for i in range(n)
]
constraints += [
    cp.diag(edges) == 1,
]
print(constraints)
prob = cp.Problem(
    cp.Maximize(cp.sum(cp.multiply(edge_weight_param, edges))), constraints
)
adj_param.value = adj
edge_weight_param.value = b_symm
result = prob.solve()
print(prob.status)
# print(result)
