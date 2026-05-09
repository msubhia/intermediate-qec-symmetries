"""
compare_circuits.py
-------------------
Helper module for converting stim.Circuit objects to Qiskit and comparing
them on depth, gate count, and two-qubit gate count after optimization.

All circuits are validated as Clifford before and after optimization.
Non-Clifford circuits raise a ValueError early so you get a clear error
instead of silently wrong results.

Usage:
    import stim
    from compare_circuits import compare_circuits

    circ_a = stim.Circuit(\"\"\"
        H 0
        CNOT 0 1
        CNOT 1 2
    \"\"\")

    circ_b = stim.Circuit(\"\"\"
        H 0 1
        CZ 0 1
        H 0
        CNOT 0 2
        S 1
    \"\"\")

    results = compare_circuits(circ_a, circ_b)

Stim gates supported:
    Single-qubit : I, H, X, Y, Z, S, T, S_DAG, T_DAG,
                   SQRT_X, SQRT_X_DAG, SQRT_Y, SQRT_Y_DAG, SQRT_Z, SQRT_Z_DAG
    Rotation     : RX, RY, RZ  (angle stored in stim gate's args)
    Two-qubit    : CNOT, CX, CY, CZ, SWAP, ISWAP, ISWAP_DAG, ECR,
                   XCX, XCY, XCZ, YCX, YCY, YCZ
    Three-qubit  : CCX, CSWAP

Stim-only instructions that have no unitary (DETECTOR, OBSERVABLE_INCLUDE,
TICK, M*, R*, etc.) are silently skipped during conversion.
"""

from __future__ import annotations

import stim
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Clifford

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

TWO_QUBIT_GATES = {
    "cx", "cy", "cz", "ecr", "swap", "iswap", "dcx",
}

# Strict Clifford basis for transpilation — no T, no arbitrary Rz
_CLIFFORD_BASIS_GATES = [
    "cx", "cz", "swap",
    "h", "s", "sdg", "x", "y", "z", "id",
    "sx", "sxdg",          # sqrt(X) also Clifford
]

# Stim gate name → function that applies it to a Qiskit QuantumCircuit.
_GATE_MAP: dict[str, callable] = {
    # ── single-qubit ──────────────────────────────────────────
    "I":          lambda qc, *q: qc.id(*q),
    "H":          lambda qc, *q: qc.h(*q),
    "X":          lambda qc, *q: qc.x(*q),
    "Y":          lambda qc, *q: qc.y(*q),
    "Z":          lambda qc, *q: qc.z(*q),
    "S":          lambda qc, *q: qc.s(*q),
    "T":          lambda qc, *q: qc.t(*q),
    "S_DAG":      lambda qc, *q: qc.sdg(*q),
    "T_DAG":      lambda qc, *q: qc.tdg(*q),
    "SQRT_X":     lambda qc, *q: qc.sx(*q),
    "SQRT_X_DAG": lambda qc, *q: qc.sxdg(*q),
    "SQRT_Y":     lambda qc, *q: qc.ry( 1.5707963267948966, *q),
    "SQRT_Y_DAG": lambda qc, *q: qc.ry(-1.5707963267948966, *q),
    "SQRT_Z":     lambda qc, *q: qc.s(*q),
    "SQRT_Z_DAG": lambda qc, *q: qc.sdg(*q),
    # ── rotation ─────────────────────────────────────────────
    "RX":         lambda qc, angle, *q: qc.rx(angle, *q),
    "RY":         lambda qc, angle, *q: qc.ry(angle, *q),
    "RZ":         lambda qc, angle, *q: qc.rz(angle, *q),
    # ── two-qubit ─────────────────────────────────────────────
    "CNOT":       lambda qc, *q: qc.cx(*q),
    "CX":         lambda qc, *q: qc.cx(*q),
    "CY":         lambda qc, *q: qc.cy(*q),
    "CZ":         lambda qc, *q: qc.cz(*q),
    "SWAP":       lambda qc, *q: qc.swap(*q),
    "ISWAP":      lambda qc, *q: qc.iswap(*q),
    "ISWAP_DAG":  lambda qc, *q: qc.dcx(*q),
    "ECR":        lambda qc, *q: qc.ecr(*q),
    "XCX":        lambda qc, *q: qc.cx(*q),
    "XCY":        lambda qc, *q: qc.cy(*q),
    "XCZ":        lambda qc, *q: qc.cx(*q),
    "YCX":        lambda qc, *q: qc.cx(*q),
    "YCY":        lambda qc, *q: qc.cy(*q),
    "YCZ":        lambda qc, *q: qc.cz(*q),
    # ── three-qubit ───────────────────────────────────────────
    "CCX":        lambda qc, *q: qc.ccx(*q),
    "CSWAP":      lambda qc, *q: qc.cswap(*q),
}

_GATE_ARITY: dict[str, int] = {
    **{g: 1 for g in ("I","H","X","Y","Z","S","T","S_DAG","T_DAG",
                       "SQRT_X","SQRT_X_DAG","SQRT_Y","SQRT_Y_DAG",
                       "SQRT_Z","SQRT_Z_DAG","RX","RY","RZ")},
    **{g: 2 for g in ("CNOT","CX","CY","CZ","SWAP","ISWAP","ISWAP_DAG",
                       "ECR","XCX","XCY","XCZ","YCX","YCY","YCZ")},
    **{g: 3 for g in ("CCX","CSWAP")},
}

_SKIP = {
    "DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS",
    "SHIFT_COORDS", "ELSE_CORRELATED_ERROR", "CORRELATED_ERROR",
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E",
    "M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ",
    "MM", "MPP",
    "R", "RX", "RY", "RZ",
}


# ─────────────────────────────────────────────────────────────
# Clifford validation
# ─────────────────────────────────────────────────────────────

def assert_clifford(qc: QuantumCircuit, label: str = "") -> Clifford:
    """
    Attempt to interpret qc as a Clifford. Raises ValueError with a
    clear message (including which gates are non-Clifford) if it fails.

    Returns the Clifford object on success.
    """
    # Check for non-Clifford gates before trying to construct the tableau.
    # T and T_DAG are the most common offenders; arbitrary Rz/U angles are another.
    NON_CLIFFORD_GATES = {"t", "tdg", "ccx", "cswap"}
    ANGLE_GATES = {"rx", "ry", "rz", "u", "u1", "u2", "u3", "p"}
    import math

    bad_gates = []
    for instruction in qc.data:
        name = instruction.operation.name
        if name in NON_CLIFFORD_GATES:
            bad_gates.append(name)
        elif name in ANGLE_GATES:
            # Clifford-compatible angles are multiples of pi/2
            for param in instruction.operation.params:
                try:
                    val = float(param)
                    # Check if val / (pi/2) is close to an integer
                    ratio = val / (math.pi / 2)
                    if abs(ratio - round(ratio)) > 1e-6:
                        bad_gates.append(f"{name}({val:.4f}) [angle not multiple of π/2]")
                except (TypeError, ValueError):
                    bad_gates.append(f"{name}(symbolic param — cannot verify)")

    if bad_gates:
        prefix = f"[{label}] " if label else ""
        raise ValueError(
            f"{prefix}Circuit is NOT Clifford. Offending gates:\n"
            + "\n".join(f"  - {g}" for g in bad_gates)
        )

    try:
        clifford = Clifford(qc)
    except Exception as e:
        prefix = f"[{label}] " if label else ""
        raise ValueError(
            f"{prefix}Clifford() constructor failed (circuit may have "
            f"non-Clifford structure not caught by gate scan):\n  {e}"
        ) from e

    return clifford


# ─────────────────────────────────────────────────────────────
# stim → Qiskit
# ─────────────────────────────────────────────────────────────

def stim_to_qiskit(circuit: stim.Circuit, name: str = "circuit") -> QuantumCircuit:
    """
    Convert a stim.Circuit to a Qiskit QuantumCircuit.

    Raises:
        ValueError: If an unsupported gate is encountered.
    """
    n_qubits = circuit.num_qubits
    qc = QuantumCircuit(n_qubits, name=name)

    for instruction in circuit:
        gate = instruction.name
        if gate in _SKIP:
            continue
        if gate not in _GATE_MAP:
            raise ValueError(
                f"Unsupported Stim gate '{gate}'. "
                f"Supported: {sorted(_GATE_MAP)}"
            )

        arity   = _GATE_ARITY[gate]
        targets = [t.value for t in instruction.targets_copy()]
        args    = list(instruction.gate_args_copy())

        for i in range(0, len(targets), arity):
            qubits = targets[i : i + arity]
            if args:
                _GATE_MAP[gate](qc, *args, *qubits)
            else:
                _GATE_MAP[gate](qc, *qubits)

    return qc


# ─────────────────────────────────────────────────────────────
# Optimize (Clifford-safe)
# ─────────────────────────────────────────────────────────────

def optimize_circuit(
    circuit: QuantumCircuit,
    backend=None,
    optimization_level: int = 3,
) -> QuantumCircuit:
    """
    Transpile a Qiskit circuit using a strict Clifford basis gate set
    and Clifford-aware synthesis, so no non-Clifford gates are introduced.
    """
    if backend is None:
        backend = AerSimulator()

    return transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        basis_gates=_CLIFFORD_BASIS_GATES,
        unitary_synthesis_method="clifford",
    )


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def collect_metrics(original: QuantumCircuit, optimized: QuantumCircuit) -> dict:
    orig_ops = original.count_ops()
    opt_ops  = optimized.count_ops()

    return {
        "label":           original.name,
        "orig_depth":      original.depth(),
        "opt_depth":       optimized.depth(),
        "orig_gate_count": sum(orig_ops.values()),
        "opt_gate_count":  sum(opt_ops.values()),
        "orig_2q":         sum(v for k, v in orig_ops.items() if k in TWO_QUBIT_GATES),
        "opt_2q":          sum(v for k, v in opt_ops.items()  if k in TWO_QUBIT_GATES),
        "orig_ops":        dict(orig_ops),
        "opt_ops":         dict(opt_ops),
    }


def _winner_label(val_a: int, val_b: int, label_a: str, label_b: str) -> str:
    if val_a < val_b:
        return label_a
    if val_b < val_a:
        return label_b
    return "tie"


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def compare_circuits(
    circuit_a: stim.Circuit,
    circuit_b: stim.Circuit,
    name_a: str = "Circuit A",
    name_b: str = "Circuit B",
    optimization_level: int = 3,
    backend=None,
) -> dict:
    """
    Convert, validate as Clifford, optimize, and compare two stim.Circuit objects.

    Both circuits are checked for Clifford structure before AND after
    optimization. A ValueError is raised immediately if either fails.

    Returns:
        {
            "circuit_a": { label, orig_depth, opt_depth,
                           orig_gate_count, opt_gate_count,
                           orig_2q, opt_2q, orig_ops, opt_ops },
            "circuit_b": { ...same... },
            "winners":   { depth, gate_count, two_q_gates },
            "optimized_circuits": { "a": QuantumCircuit, "b": QuantumCircuit },
            "clifford":  { "a": Clifford, "b": Clifford },   # pre-optimization
        }
    """
    if backend is None:
        backend = AerSimulator()

    # Convert
    qc_a = stim_to_qiskit(circuit_a, name=name_a)
    qc_b = stim_to_qiskit(circuit_b, name=name_b)

    # ── Validate Clifford BEFORE optimization ──────────────────
    print(f"Validating '{name_a}' is Clifford (pre-optimization)...")
    clif_a = assert_clifford(qc_a, label=name_a)
    print(f"  ✓ {name_a} is Clifford")

    print(f"Validating '{name_b}' is Clifford (pre-optimization)...")
    clif_b = assert_clifford(qc_b, label=name_b)
    print(f"  ✓ {name_b} is Clifford")

    # ── Optimize ───────────────────────────────────────────────
    opt_a = optimize_circuit(qc_a, backend=backend, optimization_level=optimization_level)
    opt_b = optimize_circuit(qc_b, backend=backend, optimization_level=optimization_level)

    # ── Validate Clifford AFTER optimization ───────────────────
    print(f"Validating '{name_a}' is Clifford (post-optimization)...")
    assert_clifford(opt_a, label=f"{name_a} [optimized]")
    print(f"  ✓ {name_a} (optimized) is Clifford")

    print(f"Validating '{name_b}' is Clifford (post-optimization)...")
    assert_clifford(opt_b, label=f"{name_b} [optimized]")
    print(f"  ✓ {name_b} (optimized) is Clifford")

    # ── Metrics & winners ──────────────────────────────────────
    m_a = collect_metrics(qc_a, opt_a)
    m_b = collect_metrics(qc_b, opt_b)

    winners = {
        "depth":       _winner_label(m_a["opt_depth"],      m_b["opt_depth"],      name_a, name_b),
        "gate_count":  _winner_label(m_a["opt_gate_count"], m_b["opt_gate_count"], name_a, name_b),
        "two_q_gates": _winner_label(m_a["opt_2q"],         m_b["opt_2q"],         name_a, name_b),
    }

    return {
        "circuit_a":          m_a,
        "circuit_b":          m_b,
        "winners":            winners,
        "optimized_circuits": {"a": opt_a, "b": opt_b},
        "clifford":           {"a": clif_a, "b": clif_b},
    }


# ─────────────────────────────────────────────────────────────
# Helper: print results as human-readable tables
# ─────────────────────────────────────────────────────────────

def print_results(r: dict) -> None:
    """
    Print the output of compare_circuits() as formatted tables.

    Args:
        r: The dict returned by compare_circuits().
    """
    ma = r["circuit_a"]
    mb = r["circuit_b"]
    w  = r["winners"]

    label_a = ma["label"]
    label_b = mb["label"]

    # Column widths
    C0 = 20
    C1 = max(len(label_a), 18)
    C2 = max(len(label_b), 18)

    hdiv = "=" * (C0 + C1 + C2 + 10)

    def header(title):
        print(f"\n{hdiv}")
        print(f"  {title}")
        print(hdiv)

    def col_headers():
        print(f"  {'Metric':<{C0}}  {label_a:^{C1}}  {label_b:^{C2}}")
        print(f"  {'':─<{C0}}  {'':─<{C1}}  {'':─<{C2}}")

    def row(label, before_a, after_a, before_b, after_b):
        d_a = f"({after_a - before_a:+d})" if after_a != before_a else ""
        d_b = f"({after_b - before_b:+d})" if after_b != before_b else ""
        cell_a = f"{before_a} -> {after_a} {d_a}"
        cell_b = f"{before_b} -> {after_b} {d_b}"
        print(f"  {label:<{C0}}  {cell_a:^{C1}}  {cell_b:^{C2}}")

    # Table 1: before / after
    header("BEFORE -> AFTER OPTIMIZATION")
    col_headers()
    row("Depth",         ma["orig_depth"],      ma["opt_depth"],
                         mb["orig_depth"],      mb["opt_depth"])
    row("Total gates",   ma["orig_gate_count"], ma["opt_gate_count"],
                         mb["orig_gate_count"], mb["opt_gate_count"])
    row("2-qubit gates", ma["orig_2q"],         ma["opt_2q"],
                         mb["orig_2q"],         mb["opt_2q"])

    # Table 2: winners
    header("WINNERS  (optimized, lower is better)")
    print(f"  {'Metric':<{C0}}  {'Winner'}")
    print(f"  {'':─<{C0}}  {'':─<{C1}}")
    for metric, victor in w.items():
        trophy = "[win]" if victor != "tie" else "[tie]"
        print(f"  {metric:<{C0}}  {trophy}  {victor}")

    # Table 3: gate breakdown
    header("GATE BREAKDOWN  (optimized)")
    all_gates = sorted(set(ma["opt_ops"]) | set(mb["opt_ops"]))
    print(f"  {'Gate':<{C0}}  {label_a:^{C1}}  {label_b:^{C2}}")
    print(f"  {'':─<{C0}}  {'':─<{C1}}  {'':─<{C2}}")
    for g in all_gates:
        va = ma["opt_ops"].get(g, 0)
        vb = mb["opt_ops"].get(g, 0)
        print(f"  {g:<{C0}}  {va:^{C1}}  {vb:^{C2}}")

    print(f"\n{hdiv}\n")
