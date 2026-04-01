"""
evaluator.py  —  Black-Box Fitness Evaluator for Cache-Replacement Heuristics

Accepts a dynamically-generated Python heuristic string, applies it to the
training CSV, and returns a Final Fitness Score that combines predictive
performance (F1) with a complexity penalty for over-reliance on hardcoded
constants (to guard against brittle, overfit heuristics).
"""

import ast
import textwrap
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ── Defaults (can be overridden at call time) ──────────────────────────────
DEFAULT_CSV  = "datasets/perlbench_train_70.csv"
TARGET_COL   = "decision"   # actual column written by analyze_data.py (0/1)

# Heuristics with ≤ CONSTANT_THRESHOLD numeric literals get no penalty.
# Each constant beyond that reduces the penalty factor by 0.10.
CONSTANT_THRESHOLD = 5
PENALTY_STEP       = 0.10   # 10 % reduction per extra constant


# ── AST helpers ────────────────────────────────────────────────────────────

def count_numeric_constants(code_str: str) -> int:
    """
    Walk the AST of *code_str* and count every numeric literal (int/float).
    Returns -1 on SyntaxError.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return -1

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            count += 1
    return count


def complexity_penalty(n_constants: int) -> float:
    """
    Penalty factor in (0, 1].
    f(k) = 1 / (1 + PENALTY_STEP * max(0, k - CONSTANT_THRESHOLD))
    """
    excess = max(0, n_constants - CONSTANT_THRESHOLD)
    return 1.0 / (1.0 + PENALTY_STEP * excess)


# ── Code normalisation ─────────────────────────────────────────────────────

def _normalise_code(heuristic_code: str) -> str:
    """
    Accept either a bare function body or a full 'def heuristic(row):' block.
    Always returns a complete, properly-indented function definition.
    """
    heuristic_code = textwrap.dedent(heuristic_code).strip()

    if heuristic_code.startswith("def "):
        return heuristic_code

    # Wrap bare body lines inside def heuristic(row):
    indented = textwrap.indent(heuristic_code, "    ")
    return f"def heuristic(row):\n{indented}"


# ── Main evaluation entry point ────────────────────────────────────────────

def evaluate_heuristic(
    heuristic_code: str,
    csv_path: str = DEFAULT_CSV,
    target_col: str = TARGET_COL,
) -> dict:
    """
    Evaluate a heuristic expressed as a Python string.

    The heuristic must (when wrapped in ``def heuristic(row):``) accept a
    pandas Series ``row`` and return a **numeric score**.  Rows where the
    score is **> 0** are predicted cache-friendly (label = 1); otherwise
    cache-averse (label = 0).

    Returns
    -------
    dict with keys:
      accuracy, f1, n_constants, complexity_penalty, final_fitness,
      n_samples, errors        — on success
      error                    — on failure (other keys still present but 0)
    """

    # ── 1. Normalise & count constants ────────────────────────────────────
    full_code  = _normalise_code(heuristic_code)
    n_consts   = count_numeric_constants(full_code)

    base_result = dict(
        accuracy=0.0,
        f1=0.0,
        n_constants=n_consts,
        complexity_penalty=complexity_penalty(max(n_consts, 0)),
        final_fitness=0.0,
        n_samples=0,
        errors=0,
    )

    if n_consts == -1:
        base_result["error"] = "SyntaxError: could not parse heuristic code"
        base_result["complexity_penalty"] = 1.0
        return base_result

    # ── 2. Compile & exec ─────────────────────────────────────────────────
    try:
        code_obj    = compile(full_code, "<heuristic>", "exec")
        exec_ns     = {}
        exec(code_obj, exec_ns)
        heuristic_fn = exec_ns["heuristic"]
    except Exception as exc:
        base_result["error"] = f"Execution error: {exc}"
        return base_result

    # ── 3. Load data ──────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot load CSV '{csv_path}': {exc}") from exc

    # Detect Git LFS pointer files (un-pulled LFS objects look like a tiny
    # one-column CSV whose first column name starts with "version https://")
    first_col = df.columns[0] if len(df.columns) > 0 else ""
    if first_col.startswith("version https://git-lfs"):
        raise RuntimeError(
            f"'{csv_path}' is a Git LFS pointer, not the actual data file.\n"
            "Pull the real file with:\n"
            "    sudo apt-get install git-lfs\n"
            "    git lfs install\n"
            "    git lfs pull --include='datasets/perlbench_train_70.csv'"
        )

    # Auto-detect target column: try the requested name, then common aliases,
    # then fall back to the last numeric column.
    if target_col not in df.columns:
        candidates = ["label", "decision", "cached", "target"]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            # last resort: last column (how most labelling scripts write it)
            found = df.columns[-1]
        print(
            f"  [INFO] Target column '{target_col}' not found; "
            f"using '{found}' instead. Available: {list(df.columns)}"
        )
        target_col = found

    y_true  = df[target_col].values
    scores  = np.empty(len(df), dtype=float)
    errors  = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            scores[idx] = float(heuristic_fn(row))
        except Exception:
            scores[idx] = 0.0
            errors      += 1

    # ── 4. Threshold scores → binary predictions ──────────────────────────
    y_pred = (scores > 0).astype(int)

    # ── 5. Performance metrics ────────────────────────────────────────────
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))

    # ── 6. Complexity penalty & Final Fitness ─────────────────────────────
    pen           = complexity_penalty(n_consts)
    final_fitness = f1 * pen          # penalise F1 if too many constants

    return dict(
        accuracy          = round(acc,           4),
        f1                = round(f1,            4),
        n_constants       = n_consts,
        complexity_penalty= round(pen,           4),
        final_fitness     = round(final_fitness, 4),
        n_samples         = len(df),
        errors            = errors,
    )


# ── Quick self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = "return float(row['hit'])"
    result = evaluate_heuristic(demo)
    print("Demo result:", result)
