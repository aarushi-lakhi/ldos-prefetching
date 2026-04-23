"""
evaluator.py  —  Black-Box Fitness Evaluator for Cache-Replacement Heuristics

Accepts a dynamically-generated Python heuristic string, applies it to the
training CSV, and returns a Final Fitness Score that combines predictive
performance (Balanced Accuracy) with a complexity penalty for over-reliance
on hardcoded constants (to guard against brittle, overfit heuristics).

Metrics reported (all skew-robust):
  balanced_accuracy  — average of per-class recall; immune to majority-class trap
  precision          — TP / (TP + FP) for class 1
  recall             — TP / (TP + FN) for class 1
  f1                 — harmonic mean of precision & recall for class 1
  acc_class0         — accuracy among cache-averse  rows (true label = 0)
  acc_class1         — accuracy among cache-friendly rows (true label = 1)
  final_fitness      — balanced_accuracy × complexity_penalty
"""

import ast
import re
import textwrap
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ── Defaults (can be overridden at call time) ──────────────────────────────
DEFAULT_CSV  = "datasets/perlbench_train_70.csv"
TARGET_COL   = "decision"   # actual column written by analyze_data.py (0/1)

# Glider (Shi et al. 2019) offline accuracy for the mcf trace — our comparison target.
GLIDER_MCF_BASELINE = 0.8114

# Heuristics with ≤ CONSTANT_THRESHOLD numeric literals get no penalty.
# Each constant beyond that reduces the penalty factor by 0.10.
CONSTANT_THRESHOLD = 10
PENALTY_STEP       = 0.10   # 10 % reduction per extra constant

# ── Banned features (leakage / inadmissible) ──────────────────────────────
# 'hit'        — directly encodes whether the line is already useful; trivially
#                leaks the answer and produces unrealisable policies.
# 'victim_addr'— the eviction candidate's address; not available to a general
#                replacement policy at decision time.
BANNED_FEATURES = ["hit", "victim_addr"]

# ── Dual-class gate threshold ─────────────────────────────────────────────
# If EITHER per-class accuracy falls below this, final_fitness is forced to 0.
# This prevents the LLM from gaming balanced-accuracy with all-one-class output.
DUAL_CLASS_MIN = 0.05


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


# ── Banned-feature detection ───────────────────────────────────────────────

def find_banned_features(code_str: str) -> list[str]:
    """
    Scan *code_str* for any access to a banned feature via row['name'] or
    row["name"].  Returns the list of banned names actually found.
    """
    found = []
    for feat in BANNED_FEATURES:
        pattern = rf"row\s*\[\s*['\"]({re.escape(feat)})['\"]"
        if re.search(pattern, code_str):
            found.append(feat)
    return found


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

# Stratified sample size for evaluation.  7 M-row traces take hours with
# iterrows(); 50 k rows finishes in ~5 s and gives stable metric estimates.
EVAL_SAMPLE_ROWS = 50_000


def evaluate_heuristic(
    heuristic_code: str,
    csv_path: str = DEFAULT_CSV,
    target_col: str = TARGET_COL,
    max_rows: int = EVAL_SAMPLE_ROWS,
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
      balanced_accuracy, precision, recall, f1,
      acc_class0, acc_class1,
      n_constants, complexity_penalty, final_fitness,
      n_samples, errors        — on success
      error                    — on failure (other keys still present but 0)
    """

    # ── 1. Normalise & count constants ────────────────────────────────────
    full_code  = _normalise_code(heuristic_code)
    n_consts   = count_numeric_constants(full_code)

    base_result = dict(
        balanced_accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        acc_class0=0.0,
        acc_class1=0.0,
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

    # ── 1b. Banned-feature check ──────────────────────────────────────────
    banned_used = find_banned_features(full_code)
    if banned_used:
        base_result["error"] = (
            f"BANNED FEATURES USED: {banned_used}. "
            "These features are inadmissible — they leak information that a "
            "real policy cannot access at decision time. "
            "Remove all references to: " + ", ".join(f"row['{f}']" for f in banned_used)
        )
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
            f"    git lfs pull --include='{csv_path}'"
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

    # ── 3b. Stratified sample (fast evaluation on large datasets) ────────────
    n_original = len(df)
    if max_rows and n_original > max_rows:
        # Sample proportionally from each class so the class ratio is preserved.
        # Done per-class manually to avoid FutureWarning from groupby.apply.
        parts = []
        for cls_val, grp in df.groupby(target_col):
            n_sample = max(1, round(max_rows * len(grp) / n_original))
            parts.append(grp.sample(n=n_sample, random_state=42))
        df = pd.concat(parts).reset_index(drop=True)
        print(
            f"  [INFO] Stratified sample: {len(df):,} rows "
            f"(from {n_original:,} total; class ratio preserved)"
        )

    y_true  = df[target_col].values

    # ── HARD DELETE CHEAT COLUMNS ─────────────────────────────────────────
    # Strip the label and every inadmissible feature so the heuristic
    # physically cannot access them, regardless of what the LLM wrote.
    _COLS_TO_DROP = ["hit", "victim_addr", "decision", "label", "cached", "target"]
    _existing_drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df_safe = df.drop(columns=_existing_drop)

    scores  = np.empty(len(df_safe), dtype=float)
    errors  = 0

    for idx, (_, row) in enumerate(df_safe.iterrows()):
        try:
            scores[idx] = float(heuristic_fn(row))
        except Exception:
            scores[idx] = 0.0
            errors      += 1

    # ── 4. Threshold scores → binary predictions ──────────────────────────
    y_pred = (scores > 0).astype(int)

    # ── 5. Performance metrics (skew-robust) ─────────────────────────────
    raw_acc   = float(accuracy_score(y_true, y_pred))
    bal_acc   = float(balanced_accuracy_score(y_true, y_pred))
    prec      = float(precision_score(y_true, y_pred, zero_division=0))
    rec       = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))

    print(f"  Raw Accuracy (Glider Metric): {raw_acc:.4f}")
    _delta = (raw_acc - GLIDER_MCF_BASELINE) * 100
    if _delta >= 0:
        print(f"  [GLIDER COMPARISON] Beat Baseline by {_delta:.2f}%")
    else:
        print(f"  [GLIDER COMPARISON] Missed Baseline by {abs(_delta):.2f}%")

    # Per-class accuracy (= per-class recall)
    mask0 = y_true == 0
    mask1 = y_true == 1
    acc_class0 = float((y_pred[mask0] == 0).mean()) if mask0.any() else 0.0
    acc_class1 = float((y_pred[mask1] == 1).mean()) if mask1.any() else 0.0

    # ── 6. Complexity penalty & Final Fitness ─────────────────────────────
    pen = complexity_penalty(n_consts)

    # Hard dual-class gate: a heuristic that predicts only one class is
    # worthless regardless of balanced accuracy.  Force fitness to zero so
    # the LLM receives an unambiguous signal to change strategy.
    if acc_class0 < DUAL_CLASS_MIN or acc_class1 < DUAL_CLASS_MIN:
        final_fitness = 0.0
    else:
        final_fitness = bal_acc * pen

    return dict(
        raw_accuracy      = round(raw_acc,       4),
        balanced_accuracy = round(bal_acc,       4),
        precision         = round(prec,          4),
        recall            = round(rec,           4),
        f1                = round(f1,            4),
        acc_class0        = round(acc_class0,    4),
        acc_class1        = round(acc_class1,    4),
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
