import stim
import itertools
import numpy as np
import pickle
import time
import os
import re
import pprint
import numpy as np

from autqec.automorphisms   import *
from autqec.utils.qec       import *
from autqec.utils.qiskit    import *
from autqec.graph_auts      import *
from autqec.ZX_dualities    import *
from autqec.ZY_dualities    import *
from autqec.magma_interface import *
from autqec.code_embedding  import *
from magma_online           import *


# ---------------------------------------------------------------------
# Basic Pauli / circuit helpers
# ---------------------------------------------------------------------

def circuits_equivalent(c1: stim.Circuit, c2: stim.Circuit) -> bool:
    return stim.Tableau.from_circuit(c1) == stim.Tableau.from_circuit(c2)

def commute(p1: stim.PauliString, p2: stim.PauliString) -> bool:
    """
    Return True iff two Pauli strings commute.
    """
    return p1.commutes(p2)


def conjugate_stabilizer_by_circuit(
    stabilizer: stim.PauliString | str,
    circuit: stim.Circuit,
) -> stim.PauliString:
    """
    Conjugate a Pauli stabilizer by a Stim circuit.

    Returns:
        U stabilizer U†

    Notes:
        Stim's tableau action applies the circuit's Clifford conjugation.
    """
    if isinstance(stabilizer, str):
        stabilizer = stim.PauliString(stabilizer)

    return circuit.to_tableau()(stabilizer)


def pauli_body(p: stim.PauliString | str) -> str:
    """
    Return the unsigned I/X/Y/Z body of a Pauli string.

    Examples:
        '+X_Z' -> 'XIZ'
        '-XY_' -> 'XYI'
    """
    s = str(p)

    if s.startswith("+i") or s.startswith("-i"):
        s = s[2:]
    elif s.startswith("+") or s.startswith("-"):
        s = s[1:]

    return s.replace("_", "I")

def pauli_sign(p: stim.PauliString | str) -> int:
    """
    Return the overall sign of a Pauli string.

    Returns:
        +1 or -1
    """
    s = str(p)

    if s.startswith("+i") or s.startswith("-i"):
        raise ValueError(f"Pauli has imaginary phase, not just ±1: {p}")

    return -1 if s.startswith("-") else +1


def canonicalize_real_pauli(p):
    s = pauli_body(p)
    return stim.PauliString(s)


def pauli_to_xz(p: stim.PauliString | str) -> np.ndarray:
    """
    Convert a Pauli string to binary symplectic form.

    For n qubits, returns:

        [x_0 ... x_{n-1} | z_0 ... z_{n-1}]

    Encoding:
        I -> (0, 0)
        X -> (1, 0)
        Z -> (0, 1)
        Y -> (1, 1)

    Ignores the overall sign.
    """
    s = pauli_body(p)

    x = np.array([c in "XY" for c in s], dtype=np.uint8)
    z = np.array([c in "ZY" for c in s], dtype=np.uint8)

    return np.concatenate([x, z])


def xz_to_pauli(xz: np.ndarray) -> stim.PauliString:
    """
    Convert binary symplectic form back to a Stim PauliString.

    Input:
        xz = [x_0 ... x_{n-1} | z_0 ... z_{n-1}]

    Output has positive sign.
    """
    if len(xz) % 2 != 0:
        raise ValueError("Symplectic vector length must be even.")

    n = len(xz) // 2
    xs = xz[:n]
    zs = xz[n:]

    chars = []

    for x, z in zip(xs, zs):
        if x == 0 and z == 0:
            chars.append("I")
        elif x == 1 and z == 0:
            chars.append("X")
        elif x == 0 and z == 1:
            chars.append("Z")
        elif x == 1 and z == 1:
            chars.append("Y")
        else:
            raise ValueError("Invalid binary x/z value.")

    return stim.PauliString("".join(chars))


def pauli_to_circuit(P: stim.PauliString | str) -> stim.Circuit:
    """
    Convert a Pauli string into a Stim circuit applying that Pauli.

    Example:
        P = 'IXYZ'

    gives a circuit applying:
        X on qubit 1
        Y on qubit 2
        Z on qubit 3
    """
    c = stim.Circuit()
    s = pauli_body(P)

    for q, op in enumerate(s):
        if op in ["X", "Y", "Z"]:
            c.append(op, [q])

    return c


# ---------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------

def gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b over GF(2).

    Args:
        A: m x n binary matrix.
        b: length-m binary vector.

    Returns:
        One binary solution x of length n.

    Raises:
        ValueError:
            If the system has no solution.

    Notes:
        If the system is underdetermined, this returns one solution by
        setting all free variables to 0.
    """
    A = A.copy().astype(np.uint8) % 2
    b = b.copy().astype(np.uint8) % 2

    m, n = A.shape
    aug = np.concatenate([A, b[:, None]], axis=1)

    pivots = []
    row = 0

    for col in range(n):
        pivots_below = np.where(aug[row:, col])[0]

        if len(pivots_below) == 0:
            continue

        pivot = pivots_below[0] + row
        aug[[row, pivot]] = aug[[pivot, row]]

        for r in range(m):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]

        pivots.append(col)
        row += 1

        if row == m:
            break

    for r in range(row, m):
        if np.all(aug[r, :n] == 0) and aug[r, n]:
            raise ValueError("No GF(2) solution exists.")

    x = np.zeros(n, dtype=np.uint8)

    for r, col in enumerate(pivots):
        x[col] = aug[r, n]

    return x


# ---------------------------------------------------------------------
# Stabilizer span / products
# ---------------------------------------------------------------------

def stabilizer_product(
    gens: list[stim.PauliString],
    coeffs: np.ndarray,
) -> stim.PauliString:
    """
    Multiply selected stabilizer generators.

    Args:
        gens:
            Stabilizer generators [g_0, ..., g_{m-1}].

        coeffs:
            Binary coefficients. If coeffs[j] = 1, include gens[j].

    Returns:
        Product over gens[j]^coeffs[j], including Stim's phase tracking.
    """
    if len(gens) == 0:
        raise ValueError("Generator list cannot be empty.")

    if len(coeffs) != len(gens):
        raise ValueError("coeffs and gens must have the same length.")

    out = stim.PauliString("I" * len(pauli_body(gens[0])))

    for g, c in zip(gens, coeffs):
        if c:
            out *= g

    return out


def decompose_in_stabilizer_basis(
    target: stim.PauliString,
    gens: list[stim.PauliString],
) -> np.ndarray:
    """
    Decompose a Pauli string in the span of stabilizer generators.

    Finds coefficients c_j such that, ignoring phase,

        target ~ product_j gens[j]^c_j

    where "~" means same I/X/Y/Z body.

    Args:
        target:
            Pauli string to decompose.

        gens:
            Stabilizer generators.

    Returns:
        Binary coefficient vector c.

    Raises:
        ValueError:
            If target is not in the GF(2) span of gens.
    """
    if len(gens) == 0:
        raise ValueError("Generator list cannot be empty.")

    H = np.array([pauli_to_xz(g) for g in gens], dtype=np.uint8)
    v = pauli_to_xz(target)

    return gf2_solve(H.T, v)


def equivalent_up_to_stabilizers(
    p1: stim.PauliString,
    p2: stim.PauliString,
    stabilizers: list[stim.PauliString],
):
    """
    Check whether p1 and p2 differ by a stabilizer product, ignoring phase.

    Tests whether:

        p1 * p2 ∈ span(stabilizers)

    in binary symplectic form.

    Returns:
        equivalent:
            True or False.

        used_indices:
            Indices of stabilizers used in the product.

        used_stabilizers:
            Stabilizer generators used.

        product:
            Product of used stabilizers, or None if not equivalent.
    """
    v1 = pauli_to_xz(p1)
    v2 = pauli_to_xz(p2)

    if len(v1) != len(v2):
        raise ValueError("Operators have different numbers of qubits.")

    rhs = v1 ^ v2

    if len(stabilizers) == 0:
        identity = stim.PauliString("I" * len(pauli_body(p1)))
        return bool(np.all(rhs == 0)), [], [], identity

    stab_vecs = [pauli_to_xz(s) for s in stabilizers]
    A = np.stack(stab_vecs, axis=1)

    try:
        coeffs = gf2_solve(A, rhs)
    except ValueError:
        return False, [], [], None

    used_indices = [i for i, c in enumerate(coeffs) if c]
    used_stabilizers = [stabilizers[i] for i in used_indices]
    product = stabilizer_product(stabilizers, coeffs)

    return True, used_indices, used_stabilizers, product


def stabilizer_span(generators: list[stim.PauliString]) -> set[str]:
    """
    Explicitly enumerate the signed stabilizer group generated by generators.

    Warning:
        This is exponential in len(generators). Use only for small groups.

    Returns:
        Set of signed PauliString strings.
    """
    if len(generators) == 0:
        raise ValueError("Generator list cannot be empty.")

    span = set()
    n = len(generators)

    for bits in itertools.product([0, 1], repeat=n):
        p = stim.PauliString("I" * len(pauli_body(generators[0])))

        for bit, g in zip(bits, generators):
            if bit:
                p *= g

        span.add(str(p))

    return span


def pattern_key(p: stim.PauliString) -> tuple:
    """
    Hashable phase-free key for a Pauli string.

    Useful for comparing Pauli bodies modulo sign.
    """
    return tuple(pauli_to_xz(p))


# ---------------------------------------------------------------------
# Symplectic commutation / sign correction
# ---------------------------------------------------------------------

def symp_inner(v: np.ndarray, w: np.ndarray, n: int) -> int:
    """
    Compute the binary symplectic inner product.

    For:
        v = (x1 | z1)
        w = (x2 | z2)

    returns:

        x1 · z2 + z1 · x2 mod 2

    Meaning:
        0 => corresponding Paulis commute
        1 => corresponding Paulis anticommute
    """
    x1, z1 = v[:n], v[n:]
    x2, z2 = w[:n], w[n:]

    return int((np.dot(x1, z2) + np.dot(z1, x2)) % 2)


def find_pauli_sign_correction(
    target_gens: list[stim.PauliString],
    current_gens: list[stim.PauliString],
) -> stim.PauliString:
    """
    Find a Pauli P that fixes sign mismatches between two generator lists.

    Assumption:
        target_gens[i] and current_gens[i] have the same I/X/Y/Z body.

    Goal:
        Find P such that for every i,

            P current_gens[i] P† = target_gens[i]

    Since Pauli conjugation only changes signs, this solves the linear system:

        px · z_i + pz · x_i = flip_i   mod 2

    where:
        P = (px | pz)
        current_gens[i] = (x_i | z_i)
        flip_i = 1 iff target_gens[i] and current_gens[i] have opposite signs.

    Returns:
        A PauliString P.

    Raises:
        ValueError:
            If bodies do not match or no Pauli correction exists.
    """
    if len(target_gens) != len(current_gens):
        raise ValueError("target_gens and current_gens must have the same length.")

    if len(target_gens) == 0:
        raise ValueError("Generator lists cannot be empty.")

    n = len(pauli_body(target_gens[0]))

    A = []
    b = []

    for target, current in zip(target_gens, current_gens):
        if len(pauli_body(target)) != n or len(pauli_body(current)) != n:
            raise ValueError("All generators must have the same number of qubits.")

        if pauli_body(target) != pauli_body(current):
            raise ValueError(
                f"Bodies differ:\n"
                f"  target = {target}\n"
                f"  current = {current}"
            )

        v = pauli_to_xz(current)
        x, z = v[:n], v[n:]

        # Unknown correction P = (px | pz).
        # Anticommutation with current generator:
        #   px·z + pz·x
        A.append(np.concatenate([z, x]))

        flip = 0 if pauli_sign(target) == pauli_sign(current) else 1
        b.append(flip)

    A = np.array(A, dtype=np.uint8)
    b = np.array(b, dtype=np.uint8)

    sol = gf2_solve(A, b)

    return xz_to_pauli(sol)


def find_automorphism_pauli_correction(
    auto_gens: list[stim.PauliString],
    orig_gens: list[stim.PauliString],
) -> stim.PauliString:
    """
    Find the Pauli sign correction for an automorphism output.

    Inputs:
        orig_gens:
            Original stabilizer generators [g_1, ..., g_m].

        auto_gens:
            Images of those generators under an automorphism circuit U:

                auto_gens[i] = U g_i U†

    Important:
        auto_gens may not equal orig_gens one-by-one. Instead, they may be
        a different generating basis for the same stabilizer group.

    This function does two things:

    1. Span check:
        Verifies every auto generator lies in span(orig_gens), modulo phase.

    2. Rebase:
        For each original generator g_i, write it as a product of auto_gens:

            g_i ~ product_j auto_gens[j]^c_ij

        ignoring phase.

        This gives a rebased current generator current_i with the same body
        as g_i, but possibly the wrong sign.

    3. Sign correction:
        Finds a Pauli P such that:

            P current_i P† = g_i

        for all i.

    Returns:
        Pauli correction P.

    Raises:
        ValueError:
            If the automorphism output does not span the original stabilizer
            group, or if no Pauli sign correction exists.
    """
    if len(auto_gens) != len(orig_gens):
        raise ValueError("auto_gens and orig_gens must have the same length.")

    if len(auto_gens) == 0:
        raise ValueError("Generator lists cannot be empty.")

    n = len(pauli_body(orig_gens[0]))

    for g in orig_gens + auto_gens:
        if len(pauli_body(g)) != n:
            raise ValueError("All generators must have the same number of qubits.")

    # Check every auto generator is in the original stabilizer span.
    for g in auto_gens:
        decompose_in_stabilizer_basis(g, orig_gens)

    # Rebase original generators in terms of auto generators.
    H_auto = np.array([pauli_to_xz(g) for g in auto_gens], dtype=np.uint8)

    rebased_current = []

    for target in orig_gens:
        coeffs = gf2_solve(H_auto.T, pauli_to_xz(target))
        prod = stabilizer_product(auto_gens, coeffs)
        rebased_current.append(prod)

    return find_pauli_sign_correction(orig_gens, rebased_current)


def apply_pauli_correction_to_circuit(
    circuit: stim.Circuit,
    P: stim.PauliString,
) -> stim.Circuit:
    """
    Return a new circuit equal to circuit followed by Pauli correction P.

    If auto_gens were computed as:

        auto_gens[i] = U g_i U†

    then the corrected circuit U' = P U satisfies:

        U' g_i U'† = P U g_i U† P†
    """
    corrected = circuit.copy()
    corrected += pauli_to_circuit(P)
    return corrected


# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------

def print_combined_stabilizers(
    generators: list[stim.PauliString],
    message: str = "",
    num_blocks: int = 2,
) -> None:
    """
    Pretty-print stabilizers grouped by block rows.

    Assumes generators are ordered like:

        g1_block1, g1_block2, ..., g1_blockN,
        g2_block1, g2_block2, ..., g2_blockN,
        ...

    Example for num_blocks=3:

        row 1:
            g1_b1   g1_b2   g1_b3

        row 2:
            g2_b1   g2_b2   g2_b3
    """

    assert len(generators) % num_blocks == 0, (
        "Number of generators must be divisible by num_blocks"
    )

    print("\n" + 60 * "=")
    if message:
        print(message)
    print()

    num_rows = len(generators) // num_blocks

    for row in range(num_rows):

        row_strings = []

        for block in range(num_blocks):
            idx = row * num_blocks + block

            s = str(generators[idx]).replace("_", "I")
            row_strings.append(s)

        print("   ".join(row_strings))

    print(60 * "=" + "\n")



def print_logical_evolution(logicals_dict, circuit):
    """
    Prints table of logical operators before/after conjugation.
    
    logicals_dict: dict[str, stim.PauliString]
        e.g. {
            "XI": logical_XI,
            "ZI": logical_ZI,
            ...
        }
    circuit: stim.Circuit
    """

    def fmt(p):
        return str(p).replace("_", "I")

    results = {}

    for name, op in logicals_dict.items():
        after = conjugate_stabilizer_by_circuit(op, circuit)
        results[name] = (op, after)

    print("=" * 60)
    print(" Gate        |        before                |        after")
    print("=" * 60)

    for name, (before, after) in results.items():
        print(
            f"     {name}      |     {fmt(before):<25}|  {fmt(after):<25}"
        )

    print("=" * 60)


def identify_logical_action_with_sign(after_op, logical_basis, stabilizers):
    matches = []

    for name, basis_op in logical_basis.items():
        equiv, _, _, prod = equivalent_up_to_stabilizers(
            after_op,
            basis_op,
            stabilizers,
        )

        if equiv:
            reconstructed = basis_op * prod
            sign = "+" if pauli_sign(after_op) == pauli_sign(reconstructed) else "-"
            matches.append(sign + name)

    return matches


def get_logical_action_map(logicals_dict, circuit, stabilizers, verbose=False):
    """
    Returns mapping of logical operators under a circuit.

    Args:
        logicals_dict: dict[str, stim.PauliString]
            e.g. {"XI": ..., "ZI": ..., ...}

        circuit: stim.Circuit

        stabilizers: list[stim.PauliString]

        verbose: bool
            If True, also prints table.

    Returns:
        dict[str, list[str]]
            Mapping:
                "XI" -> ["-YI"]
                "ZI" -> ["ZI"]
                ...
    """
    mapping = {}

    if verbose:
        print("=" * 75)
        print(" Gate        |        before        |        after maps to")
        print("=" * 75)

    for name, before in logicals_dict.items():
        after = conjugate_stabilizer_by_circuit(before, circuit)
        matches = identify_logical_action_with_sign(after, logicals_dict, stabilizers)

        if len(matches) == 0:
            mapping[name] = []
            action = "not in listed logical basis"
        else:
            mapping[name] = matches
            action = ", ".join(matches)

        if verbose:
            print(f"{name:^12} | {str(before).replace('_','I'):<22} | {action}")

    if verbose:
        print("=" * 75)

    return mapping


def signed_logical_to_pauli(s):
    """
    Convert strings like '-YI', '+IX', 'ZI' into stim.PauliString.
    """
    sign = -1 if s.startswith("-") else +1
    body = s[1:] if s.startswith(("+", "-")) else s

    p = stim.PauliString(body)

    if sign == -1:
        p *= -1

    return p


def logical_map_to_tableau_circuit(mapping):
    """
    mapping should give images of canonical logical generators.
    Values may be strings like '-YI', '+IZ', or lists like ['-YI'].
    """

    def get_single(key):
        value = mapping[key]

        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Expected exactly one match for {key}, got {value}")
            value = value[0]

        return signed_logical_to_pauli(value)

    xs = [
        get_single("XI"),
        get_single("IX"),
    ]

    zs = [
        get_single("ZI"),
        get_single("IZ"),
    ]

    tableau = stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)
    return tableau.to_circuit()


def autoqec_to_stim_circuit(gates):
    """
    Convert AutoQEC-style gate list (1-indexed) into a Stim circuit (0-indexed).

    Supports:
        - Single qubit gates: 'X','Y','Z','H','S'
        - Two qubit gates: 'SWAP','CZ','CNOT'
        - Custom: 'GammaXZY', 'GammaXYZ'
    """

    circuit = stim.Circuit()

    for gate, qs in gates:

        # normalize to tuple
        if isinstance(qs, int):
            qs = (qs,)

        # convert to 0-index
        qs0 = [q - 1 for q in qs]

        # ---- custom gammas ----
        if gate == "GammaXZY":
            q = qs0[0]
            circuit.append("H", [q])
            circuit.append("S", [q])

        elif gate == "GammaXYZ":
            q = qs0[0]
            circuit.append("S", [q])
            circuit.append("H", [q])

        elif gate == "Xsqrt":
            q = qs0[0]
            circuit.append("SQRT_X", [q])

        # ---- standard 1-qubit gates ----
        elif gate in ["X", "Y", "Z", "H", "S"]:
            circuit.append(gate, qs0)

        # ---- 2-qubit gates ----
        elif gate in ["SWAP", "CZ", "CNOT"]:
            circuit.append(gate, qs0)


        else:
            raise ValueError(f"Unknown gate: {gate}")

    return circuit



def find_nontrivial_autoqec_circuits(generators, n, k, d, fileroot="./", verbose=False):
    """
    Find non-identity AutoQEC/Magma automorphism circuits.

    Args:
        n, k, d:
            Code parameters.
        generators:
            List of stabilizer strings. Signs are allowed and ignored.
            Example:
                ['-ZZZXIIIIII', '-IXZZZIIIII', ...]
        fileroot:
            Where Magma command/output files are stored.
        verbose:
            If True, prints circuits.

    Returns:
        nontrivial_circuits:
            List of tuples:
                (logical_action, circ_desc, U_logical)
    """

    # ------------------------------------------------------------
    # 1. Strip signs from generators
    # ------------------------------------------------------------
    clean_generators = []

    for g in generators:
        g = str(g)

        if g.startswith("+") or g.startswith("-"):
            g = g[1:]

        g = g.replace("_", "I")
        clean_generators.append(g)

    # ------------------------------------------------------------
    # 2. Build symplectic check matrix
    # ------------------------------------------------------------
    H_symp = stabs_to_H_symp(clean_generators)

    # ------------------------------------------------------------
    # 3. File names
    # ------------------------------------------------------------
    command_file_name = f"{fileroot}/magma_commands_n{n}k{k}d{d}.txt"
    output_file_name  = f"{fileroot}/magma_output_n{n}k{k}d{d}.txt"

    # clear old output
    with open(output_file_name, "w") as output_file:
        pass

    # ------------------------------------------------------------
    # 4. Run Magma automorphism search
    # ------------------------------------------------------------
    magma_obj = qec_code_auts_from_magma(n, k, d, H_symp)
    magma_obj.run(fileroot=fileroot, save_magma_commands=True)

    run_magma_online_from_file(command_file_name)

    code_auts_dict = parse_magma_output(output_file_name, magma_obj)

    auts = code_auts_dict["auts"]

    if verbose:
        print("number of automorphism generators =", len(auts))

    # ------------------------------------------------------------
    # 5. Convert automorphisms to circuits
    # ------------------------------------------------------------
    nontrivial_circuits = []

    for num, aut in enumerate(auts):
        phys_act = circ_from_aut(H_symp, aut)
        phys_circ, _ = phys_act.circ()

        log_act = logical_circ_and_pauli_correct(H_symp, phys_circ)
        circ = log_act.run()

        logical_action, circ_desc = circ
        U_logical = log_act.U_logical_act()

        # --------------------------------------------------------
        # 6. Skip identity circuits
        # --------------------------------------------------------
        is_trivial = (
            len(circ_desc) == 0
            or logical_action == []
            or str(circ_desc).strip() in ["[]", ""]
        )

        if not is_trivial:
            nontrivial_circuits.append(circ_desc)

            if verbose:
                print(f"Circuit ({num})", circ_desc)
                print("     Logical Action:", logical_action)

    return nontrivial_circuits


def find_nontrivial_autoqec_circuits_embedded(generators, n, k, d, fileroot="./", verbose=False):
    n_og = n
    # ------------------------------------------------------------
    # 1. Strip signs from generators
    # ------------------------------------------------------------
    clean_generators = []

    for g in generators:
        g = str(g)

        if g.startswith("+") or g.startswith("-"):
            g = g[1:]

        g = g.replace("_", "I")
        clean_generators.append(g)

    # ------------------------------------------------------------
    # 2. Build symplectic check matrix
    # ------------------------------------------------------------
    H_symp = stabs_to_H_symp(clean_generators)

    H_symp_embedded = qec_embed_code(H_symp, embedding = 'two_code_blocks').embed_mat()
    n = H_symp_embedded.shape[1]//2
    print(H_symp_embedded.shape)

    embed_codespace_dict = {1: (1,6),
                            2: (2,7),
                            3: (3,8),
                            4: (4,9),
                            5: (5,10)}
    
    G, LX, LZ, D = compute_standard_form(H_symp_embedded)

    # ------------------------------------------------------------
    # 3. File names
    # ------------------------------------------------------------
    command_file_name = f"{fileroot}/magma_commands_n{n}k{k}d{d}.txt"
    output_file_name  = f"{fileroot}/magma_output_n{n}k{k}d{d}.txt"

    # clear old output
    with open(output_file_name, "w") as output_file:
        pass

    # ------------------------------------------------------------
    # 4. Run Magma automorphism search
    # ------------------------------------------------------------
    magma_obj = qec_code_auts_from_magma(n, k, d, H_symp_embedded)
    magma_obj.run(fileroot=fileroot, save_magma_commands=True)

    run_magma_online_from_file(command_file_name)

    code_auts_dict = parse_magma_output(output_file_name, magma_obj)

    auts = code_auts_dict["auts"]

    if verbose:
        print("number of automorphism generators =", len(auts))

    # ------------------------------------------------------------
    # 5. Convert automorphisms to circuits
    # ------------------------------------------------------------

    auts = code_auts_dict['auts']
    circuits = []
    for num, aut in enumerate(auts):
        phys_act = circ_from_aut(H_symp_embedded,aut)        
        phys_circ, _ = phys_act.circ()
        log_act, circ = logical_circ_and_pauli_correct(H_symp_embedded,phys_circ).run()
        circuits.append((log_act,circ))

    nontrivial_circuits = []

    for num, (_, phys_circ) in enumerate(circuits):
        normal_circ = embed_circ_to_normal_circ(
            phys_circ, n_og, embed_codespace_dict
        )

        try:
            log_act = logical_circ_and_pauli_correct(H_symp, normal_circ).run()[0]
        except Exception as e:
            if verbose:
                print(f"Skipping circuit {num} due to error:", e)
            continue

        is_trivial = (
            len(normal_circ) == 0
            or log_act == []
            or str(normal_circ).strip() in ["[]", ""]
        )

        if not is_trivial:
            nontrivial_circuits.append((normal_circ, log_act))

            if verbose:
                print(f"Circuit ({num})", normal_circ)
                print("     Logical Action:", log_act)

    return nontrivial_circuits

def corrected_auto_from_generators(autoqec_circ_desc, generators):
    """
    Given an AutoQEC circuit description and the stabilizers of the current code,
    build the Stim circuit, find Pauli correction, and return corrected circuit.
    """

    auto_circuit = autoqec_to_stim_circuit(autoqec_circ_desc)
    auto_circuit.append("I", [i for i in range(10)])

    generators_after_auto = [
        conjugate_stabilizer_by_circuit(s, auto_circuit)
        for s in generators
    ]

    pauli_correction = find_automorphism_pauli_correction(
        auto_gens=generators_after_auto,
        orig_gens=generators,
    )

    corrected_auto_circuit = auto_circuit.copy()
    corrected_auto_circuit += pauli_to_circuit(pauli_correction)

    generators_after_corrected = [
        conjugate_stabilizer_by_circuit(s, corrected_auto_circuit)
        for s in generators
    ]

    return corrected_auto_circuit, generators_after_corrected, pauli_correction



def search_nested_pieceable_autos(
    combined_generators,
    logical_basis,
    prologues,
    epilogues,
    n=10,
    k=2,
    d=3,
    fileroot="./",
    verbose=False,
    display_nodes=True,
):
    results = []
    depth = len(prologues)
    assert len(epilogues) == depth

    def evaluate_node(level, current_gens, current_circuit, path):
        """
        Treat current node as a temporary leaf:
            current_circuit + epilogue_level ... epilogue_1

        If level = 2, that means we already applied:
            prologue1 auto1 corr1 prologue2 auto2 corr2

        Then close with:
            epilogue2 epilogue1
        """
        final_gens = current_gens
        final_circuit = current_circuit.copy()

        # close only the pieces already opened
        for epilogue in reversed(epilogues[:level]):
            final_gens = [
                conjugate_stabilizer_by_circuit(s, epilogue)
                for s in final_gens
            ]
            final_circuit += epilogue

        mapping = get_logical_action_map(
            logicals_dict=logical_basis,
            circuit=final_circuit,
            stabilizers=final_gens,
        )

        try:
            logical_circuit = logical_map_to_tableau_circuit(mapping)
        except Exception:
            logical_circuit = None

        result = {
            "level": level,
            "path": path,
            "full_circuit": final_circuit,
            "final_generators": final_gens,
            "mapping": mapping,
            "logical_circuit": logical_circuit,
        }

        results.append(result)

        if display_nodes:
            print("=" * 80)
            print("node level:", level)
            print("path:", [p["index"] for p in path])
            print("mapping:", mapping)

            display(final_circuit.diagram("timeline-svg"))

            if logical_circuit is not None:
                display(logical_circuit.diagram("timeline-svg"))

        return result

    def recurse(level, current_gens, current_circuit, path):
        if level == depth:
            return

        prologue = prologues[level]

        gens_after_prologue = [
            conjugate_stabilizer_by_circuit(s, prologue)
            for s in current_gens
        ]

        circuit_after_prologue = current_circuit + prologue

        autoqec_circuits = find_nontrivial_autoqec_circuits(
            gens_after_prologue,
            n=n,
            k=k,
            d=d,
            fileroot=fileroot,
            verbose=False,
        )

        if verbose:
            print(f"level {level}: num autos =", len(autoqec_circuits))

        for i, autoqec_desc in enumerate(autoqec_circuits):
            corrected_auto, gens_after_auto, pauli_corr = corrected_auto_from_generators(
                autoqec_desc,
                gens_after_prologue,
            )

            node_circuit = circuit_after_prologue + corrected_auto

            new_path = path + [{
                "level": level,
                "index": i,
                "auto": autoqec_desc,
                "pauli_corr": pauli_corr,
            }]

            # evaluate this node as a leaf
            evaluate_node(
                level=level + 1,
                current_gens=gens_after_auto,
                current_circuit=node_circuit,
                path=new_path,
            )

            # then recurse deeper
            recurse(
                level=level + 1,
                current_gens=gens_after_auto,
                current_circuit=node_circuit,
                path=new_path,
            )

    recurse(
        level=0,
        current_gens=combined_generators,
        current_circuit=stim.Circuit(),
        path=[],
    )

    return results

def remove_identity_gates(circuit):
    new_circuit = stim.Circuit()
    
    for inst in circuit:
        # Skip identity gates
        if inst.name == "I":
            continue
        
        # Otherwise keep the instruction
        new_circuit.append(inst.name, inst.targets_copy())
    
    return new_circuit
