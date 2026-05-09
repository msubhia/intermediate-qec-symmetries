import numpy as np
import stim
import itertools



def pauli_to_xz(p: stim.PauliString):
    s = str(p).replace("+", "").replace("-", "").replace("_", "I")
    x = np.array([c in "XY" for c in s], dtype=np.uint8)
    z = np.array([c in "ZY" for c in s], dtype=np.uint8)
    return x, z


def xz_to_pauli(x, z):
    chars = []
    for xi, zi in zip(x, z):
        if xi == 0 and zi == 0:
            chars.append("I")
        elif xi == 1 and zi == 0:
            chars.append("X")
        elif xi == 0 and zi == 1:
            chars.append("Z")
        else:
            chars.append("Y")
    return stim.PauliString("".join(chars))


def gf2_inv(A):
    A = A.copy().astype(np.uint8) % 2
    n = A.shape[0]
    I = np.eye(n, dtype=np.uint8)
    aug = np.concatenate([A, I], axis=1)

    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, n):
            if aug[i, c]:
                pivot = i
                break
        if pivot is None:
            raise ValueError("Matrix is not invertible over GF(2).")

        if pivot != r:
            aug[[r, pivot]] = aug[[pivot, r]]

        for i in range(n):
            if i != r and aug[i, c]:
                aug[i] ^= aug[r]

        r += 1

    return aug[:, n:]


def find_hadamard_layer(X, Z):
    """
    Greedy search for a set of qubits to Hadamard (swap X<->Z) so that
    the resulting X-block is full rank.

    For each row, we scan qubits left-to-right and pick whichever of the
    X-column or Z-column gives a pivot first. Choosing Z means we apply H
    on that qubit.

    Returns:
        hadamard_qubits : list of qubit indices where H is applied
        col_perm        : qubit ordering with pivot qubits first
        row_ops         : GF(2) row-operation matrix
    """
    m, n = X.shape
    XZ = np.concatenate([X, Z], axis=1).astype(np.uint8)  # m x 2n
    row_ops = np.eye(m, dtype=np.uint8)

    pivot_info = []   # (qubit_idx, used_z_side)
    used_qubits = set()

    r = 0
    for qubit in range(n):
        if r >= m:
            break
        for z_side in [False, True]:
            col = qubit + (n if z_side else 0)
            pivot = None
            for i in range(r, m):
                if XZ[i, col]:
                    pivot = i
                    break
            if pivot is None:
                continue

            if pivot != r:
                XZ[[r, pivot]] = XZ[[pivot, r]]
                row_ops[[r, pivot]] = row_ops[[pivot, r]]

            for i in range(m):
                if i != r and XZ[i, col]:
                    XZ[i] ^= XZ[r]
                    row_ops[i] ^= row_ops[r]

            pivot_info.append((qubit, z_side))
            used_qubits.add(qubit)
            r += 1
            break

    if r < m:
        raise ValueError(
            f"Could not find full-rank pivot set (rank={r} < {m}). "
            "Stabilizers may not be independent."
        )

    hadamard_qubits = [q for q, from_z in pivot_info if from_z]
    pivot_qubits    = [q for q, _      in pivot_info]
    non_pivot       = [q for q in range(n) if q not in used_qubits]
    col_perm        = pivot_qubits + non_pivot

    return hadamard_qubits, col_perm, row_ops


def stabilizers_to_graph(stabilizers):
    """
    Input: n independent stabilizers of a stabilizer state.

    Automatically finds a local Hadamard layer so the X-block becomes
    invertible, then computes the graph state adjacency matrix A.

    Returns:
        graph_stabilizers  (in original qubit order)
        adjacency matrix A (in original qubit order)
        hadamard_qubits    (qubit indices where H was applied)
    """
    Xs, Zs = [], []
    for g in stabilizers:
        x, z = pauli_to_xz(g)
        Xs.append(x)
        Zs.append(z)

    X = np.array(Xs, dtype=np.uint8)
    Z = np.array(Zs, dtype=np.uint8)
    n = X.shape[1]
    m = X.shape[0]

    hadamard_qubits, col_perm, row_ops = find_hadamard_layer(X, Z)

    # Apply Hadamard on selected qubits: swap X<->Z columns
    X_lc = X.copy()
    Z_lc = Z.copy()
    for q in hadamard_qubits:
        X_lc[:, q], Z_lc[:, q] = Z[:, q].copy(), X[:, q].copy()

    inv_perm = np.argsort(col_perm)

    X_perm = X_lc[:, col_perm]
    Z_perm = Z_lc[:, col_perm]

    X_red = (row_ops @ X_perm) % 2
    Z_red = (row_ops @ Z_perm) % 2

    X_inv = gf2_inv(X_red[:, :m])
    A_perm = (X_inv @ Z_red) % 2
    np.fill_diagonal(A_perm, 0)

    # Unpermute A back to original qubit order
    A = A_perm[np.ix_(inv_perm[:m], inv_perm[:m])]
    np.fill_diagonal(A, 0)

    graph_stabilizers = []
    for i in range(m):
        x = np.zeros(m, dtype=np.uint8)
        z = A[i].copy()
        x[i] = 1
        graph_stabilizers.append(xz_to_pauli(x, z))

    return graph_stabilizers, A, hadamard_qubits


def logicals_to_cws_codewords(logical_Xs, A, hadamard_qubits=None):
    """
    Given logical X operators, convert them to CWS Z-only word operators.

    For graph stabilizers K_i = X_i Z^{A_i},
    any Pauli X^x Z^z is equivalent to Z^{z + x A}.

    If hadamard_qubits is provided, the logical operators are first
    conjugated by the same local Hadamard layer used for the graph state.
    """
    codeword_generators = []

    for L in logical_Xs:
        x, z = pauli_to_xz(L)

        if hadamard_qubits:
            x, z = x.copy(), z.copy()
            for q in hadamard_qubits:
                x[q], z[q] = z[q], x[q]

        c = (z + x @ A) % 2
        codeword_generators.append(c)

    return codeword_generators


def span_binary(gens):
    """
    Return all binary sums of generator vectors.
    """
    gens = [np.array(g, dtype=np.uint8) for g in gens]
    if len(gens) == 0:
        return [np.array([], dtype=np.uint8)]

    k = len(gens)
    words = []

    for mask in range(2**k):
        v = np.zeros_like(gens[0])
        for i in range(k):
            if (mask >> i) & 1:
                v ^= gens[i]
        words.append(v)

    return words


def bits_to_str(v):
    return "".join(str(int(x)) for x in v)


def stabilizer_code_to_cws(stabilizers, logical_Zs, logical_Xs):
    """
    Inputs:
        stabilizers : n-k stabilizer generators
        logical_Zs  : k logical Zs used to fix one stabilizer state
        logical_Xs  : k logical Xs used to generate CWS words

    Output:
        graph_stabilizers
        C               : list of CWS codeword bitstrings
        A               : adjacency matrix
        hadamard_qubits : qubit indices where a local H was applied
    """
    state_stabilizers = stabilizers + logical_Zs

    graph_stabilizers, A, hadamard_qubits = stabilizers_to_graph(state_stabilizers)

    c_gens = logicals_to_cws_codewords(logical_Xs, A, hadamard_qubits)

    C = span_binary(c_gens)
    C = [bits_to_str(c) for c in C]

    return graph_stabilizers, C, A, hadamard_qubits


def apply_ccz_to_cws(C, A, gate_list):
    """
    Apply CCZ gates to a CWS code, updating to a hypergraph CWS code.

    CCZ gates are diagonal and are absorbed into the hypergraph:
    CCZ(a,b,c) toggles the 3-hyperedge {a,b,c}.

    The codewords C are unchanged — CCZ does not affect the word operators,
    only the underlying resource state (graph -> hypergraph).

    Parameters
    ----------
    C        : list of bitstrings
    A        : np.ndarray (n,n), graph adjacency matrix
    gate_list: list of ('CCZ', [a, b, c]) tuples

    Returns
    -------
    C_new      : same as C (unchanged)
    hyperedges : dict mapping frozenset -> 1 (mod 2)
    """
    n = A.shape[0]

    # Load existing 2-edges from A
    hyperedges = {}
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                hyperedges[frozenset({i, j})] = 1

    for gate, qubits in gate_list:
        if gate != "CCZ":
            raise ValueError(f"Unsupported gate: {gate}. Only CCZ is supported.")

        key = frozenset(qubits)
        hyperedges[key] = 1 - hyperedges.get(key, 0)
        if hyperedges[key] == 0:
            del hyperedges[key]

    return list(C), hyperedges



def print_hypergraph(hyperedges):
    """Pretty-print the hypergraph edges by order."""
    edges_2 = sorted([sorted(e) for e in hyperedges if len(e) == 2])
    edges_3 = sorted([sorted(e) for e in hyperedges if len(e) == 3])
    print(f"  2-edges ({len(edges_2)}): {edges_2}")
    print(f"  3-edges ({len(edges_3)}): {edges_3}")

