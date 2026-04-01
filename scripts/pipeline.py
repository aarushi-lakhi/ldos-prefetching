"""
pipeline.py  —  Level-1 Black-Box LLM Optimisation Pipeline
             for Cache Replacement Heuristics

Runs an iterative loop where an LLM (via AWS Bedrock) proposes Python
heuristic bodies, the evaluator scores them, and the score + complexity
penalty are fed back as graded feedback for the next generation.

Diversity tracking prevents the loop from stagnating on minor numeric tweaks:
if the last DIVERSITY_WINDOW heuristics are too similar (measured by
code-fingerprint clustering), the LLM is explicitly asked to explore a
fundamentally different strategy.

Usage
-----
    # from project root:
    python scripts/pipeline.py --iterations 15 --model claude-haiku4.5

    python scripts/pipeline.py --help
"""

import sys
import os
import re
import json
import argparse
import hashlib
import textwrap
from datetime import datetime

# ── Make sibling scripts importable when run from any directory ────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from LLMWrapper import get_wrapper          # noqa: E402
from evaluator  import evaluate_heuristic   # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Feature reference (derived from ChampSim datacollector.h + add_labels.py)
# ══════════════════════════════════════════════════════════════════════════
FEATURE_DOC = """\
Available features in `row` (pandas Series):
  row['triggering_cpu']  – CPU core that triggered the access (int, usually 0)
  row['set']             – Cache set index (int, 0 … num_sets-1)
  row['way']             – Cache way currently holding this line (int)
  row['full_addr']       – Full 64-bit memory address (int)
  row['ip']              – Instruction Pointer / Program Counter (int)
  row['victim_addr']     – Address of the line that would be evicted (int)
  row['type']            – Access type: 0=LOAD 1=RFO 2=PREFETCH 3=WRITE 4=TRANSLATION
  row['hit']             – 1 if this access was a cache hit, 0 if miss
  row['timestamp']       – Microsecond wall-clock timestamp of the access (int)

Label semantics (do NOT use in the heuristic):
  decision = 1  → Belady OPT says KEEP this line (cache-friendly)  [~97 % of rows]
  decision = 0  → Belady OPT says EVICT this line (cache-averse)   [~3 % of rows]

IMPORTANT — class imbalance: ~97 % of rows are label 1.
Simply returning a positive constant scores F1 ≈ 0.98 but is USELESS in practice
because it never predicts evictions.  A heuristic that correctly identifies even
some of the ~3 % eviction cases while keeping most cache-friendly hits will
produce a more ROBUST and GENERALIZABLE policy.  Aim to predict BOTH classes.
"""

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert computer architect designing cache replacement heuristics.

    TASK
    ----
    Write the BODY of this Python function (omit the `def` line):

        def heuristic(row):
            # your code here
            return <numeric_score>

    {feature_doc}

    CONTRACT
    --------
    • Return a single numeric score.
    • POSITIVE score  →  predict cache-friendly  (label = 1, keep the line)
    • ZERO / NEGATIVE →  predict cache-averse    (label = 0, evict the line)
    • Access features via row['feature_name'].
    • No imports, no print statements.
    • Minimise hardcoded magic numbers — each extra constant beyond 5
      reduces your Final Fitness score by 10 %.  Prefer relational logic.

    OUTPUT FORMAT
    -------------
    Return ONLY the Python function body — no markdown, no explanation.
    """)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _strip_markdown(text: str) -> str:
    """Remove fenced code-block markers that LLMs often add."""
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _fingerprint(code: str) -> str:
    """Short MD5 fingerprint of normalised code (ignores whitespace changes)."""
    normalised = re.sub(r"\s+", " ", code.strip())
    return hashlib.md5(normalised.encode()).hexdigest()[:10]


def _is_low_diversity(history: list, window: int) -> bool:
    """
    True when every heuristic in the last `window` entries shares a common
    set of features (i.e. same variables referenced), suggesting the LLM is
    only tweaking constants rather than exploring new logic.
    """
    if len(history) < window:
        return False

    recent = history[-window:]

    # Build the set of row['...'] feature names accessed in each heuristic
    def _features_used(code: str) -> frozenset:
        return frozenset(re.findall(r"row\['(\w+)'\]", code))

    feature_sets = [_features_used(h["code"]) for h in recent]

    # Low diversity: all heuristics touch exactly the same features
    return len(set(map(frozenset, feature_sets))) == 1


def _build_prompt(
    feedback: str,
    prev_code: str | None,
    diversity_warning: bool,
    iteration: int,
) -> str:
    parts = [SYSTEM_PROMPT.format(feature_doc=FEATURE_DOC)]

    if iteration > 1 and feedback:
        parts.append(f"\n--- FEEDBACK FROM PREVIOUS ITERATION ---\n{feedback}\n")

    if prev_code:
        parts.append(
            f"\n--- PREVIOUS HEURISTIC (for reference only) ---\n"
            f"```python\n{prev_code}\n```\n"
        )

    if diversity_warning:
        parts.append(
            "\n⚠️  DIVERSITY ALERT: Your last several heuristics used the same "
            "features and appear to be minor variations of each other.  "
            "Please explore a FUNDAMENTALLY DIFFERENT strategy.  Ideas:\n"
            "  • Bitwise address-page analysis (full_addr >> N)\n"
            "  • Combining hit/miss history with instruction type\n"
            "  • Set-occupancy reasoning via `set` and `way`\n"
            "  • Victim-address delta vs current address\n"
            "  • Timestamp-based recency proxy\n"
        )

    parts.append(
        "\nWrite an IMPROVED heuristic body now.  "
        "Return ONLY the function body (no `def` line, no markdown)."
    )
    return "\n".join(parts)


def _format_feedback(iteration: int, metrics: dict) -> str:
    """Convert an evaluation result dict into a natural-language feedback string."""
    lines = [
        f"Iteration {iteration} scores:",
        f"  Accuracy          : {metrics['accuracy']:.4f}",
        f"  F1 Score          : {metrics['f1']:.4f}",
        f"  Hardcoded constants: {metrics['n_constants']}  "
        f"(threshold = 5; penalty factor = {metrics['complexity_penalty']:.4f})",
        f"  ► Final Fitness   : {metrics['final_fitness']:.4f}  "
        f"  [= F1 × complexity_penalty]",
    ]

    if metrics["errors"] > 0:
        pct = 100 * metrics["errors"] / max(metrics["n_samples"], 1)
        lines.append(
            f"\n  ⚠ Your heuristic raised exceptions on {metrics['errors']} rows "
            f"({pct:.1f} %) — those were treated as score = 0."
        )

    # Qualitative hints
    if metrics["n_constants"] > CONSTANT_THRESHOLD:
        excess = metrics["n_constants"] - CONSTANT_THRESHOLD
        loss   = 1.0 - metrics["complexity_penalty"]
        lines.append(
            f"\n  Constants penalty: {excess} extra constant(s) cost you "
            f"{loss:.1%} of your fitness.  Use relational logic instead."
        )

    if metrics["f1"] < 0.35:
        lines.append(
            "\n  The F1 is very low.  The current approach may be predicting "
            "one class almost exclusively — try a different strategy entirely."
        )
    elif metrics["f1"] < 0.55:
        lines.append(
            "\n  F1 is modest.  Consider whether combining multiple features "
            "with additive weights or conditional branches could help."
        )
    else:
        lines.append(
            "\n  Good F1.  Focus on reducing constants or refining edge cases "
            "to push further."
        )

    return "\n".join(lines)


# numeric constant threshold (mirror from evaluator for messaging)
CONSTANT_THRESHOLD = 5


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline loop
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    model_name:       str  = "claude-haiku4.5",
    n_iterations:     int  = 10,
    csv_path:         str  = "datasets/perlbench_train_70.csv",
    target_col:       str  = "decision",
    diversity_window: int  = 3,
    results_path:     str  = "scripts/pipeline_results.json",
) -> dict:
    """
    Run the optimisation loop.

    Returns the best result dict found across all iterations.
    """
    print("=" * 60)
    print("  LLM Cache-Replacement Heuristic Optimiser")
    print("=" * 60)
    print(f"  Model      : {model_name}")
    print(f"  Iterations : {n_iterations}")
    print(f"  Dataset    : {csv_path}")
    print(f"  Target col : {target_col}")
    print("=" * 60, "\n")

    llm = get_wrapper(model_name)

    history: list[dict] = []
    best    = {"final_fitness": -1.0, "code": None, "metrics": None, "iteration": -1}
    feedback    = ""
    prev_code   = None

    for i in range(1, n_iterations + 1):
        print(f"── Iteration {i:>2}/{n_iterations} ──────────────────────────")

        low_div = _is_low_diversity(history, diversity_window)
        if low_div:
            print("  [diversity] Low-diversity signal — prompting for new strategy")

        prompt = _build_prompt(
            feedback         = feedback,
            prev_code        = prev_code,
            diversity_warning= low_div,
            iteration        = i,
        )

        # ── LLM call ──────────────────────────────────────────────────────
        try:
            raw   = llm.send_pdf(prompt, None)
            code  = _strip_markdown(raw)
        except Exception as exc:
            print(f"  [ERROR] LLM call failed: {exc}")
            feedback = f"Iteration {i}: LLM call failed ({exc}). Please try again."
            history.append({"iteration": i, "code": "", "metrics": None, "hash": ""})
            continue

        fp = _fingerprint(code)
        print(f"  Code fingerprint : {fp}")

        # ── Evaluate ──────────────────────────────────────────────────────
        metrics = evaluate_heuristic(code, csv_path=csv_path, target_col=target_col)

        if "error" in metrics:
            print(f"  [ERROR] Evaluator: {metrics['error']}")
            feedback = (
                f"Iteration {i}: Your heuristic could not be evaluated.\n"
                f"Error: {metrics['error']}\n"
                "Please write syntactically valid Python that returns a number."
            )
        else:
            print(f"  Accuracy         : {metrics['accuracy']:.4f}")
            print(f"  F1 Score         : {metrics['f1']:.4f}")
            print(f"  Constants (#)    : {metrics['n_constants']}")
            print(f"  Complexity penalty: {metrics['complexity_penalty']:.4f}")
            print(f"  ► Final Fitness  : {metrics['final_fitness']:.4f}")

            if metrics["final_fitness"] > best["final_fitness"]:
                best = {
                    "final_fitness": metrics["final_fitness"],
                    "code"         : code,
                    "metrics"      : metrics,
                    "iteration"    : i,
                }
                print(f"  ★ NEW BEST  →  Fitness = {metrics['final_fitness']:.4f}")

            feedback  = _format_feedback(i, metrics)
            prev_code = code

        history.append({
            "iteration": i,
            "code"     : code,
            "metrics"  : metrics,
            "hash"     : fp,
        })

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OPTIMISATION COMPLETE")
    print("=" * 60)

    if best["code"] is not None:
        m = best["metrics"]
        print(f"  Best iteration : {best['iteration']}")
        print(f"  Final Fitness  : {best['final_fitness']:.4f}")
        print(f"  Accuracy       : {m['accuracy']:.4f}")
        print(f"  F1 Score       : {m['f1']:.4f}")
        print(f"  Constants      : {m['n_constants']}")
        print(f"\nBest heuristic body:\n")
        print(textwrap.indent(best["code"], "    "))
    else:
        print("  No valid heuristic was produced.")

    # ── Persist results ───────────────────────────────────────────────────
    try:
        payload = {
            "timestamp"       : datetime.now().isoformat(),
            "model"           : model_name,
            "iterations_run"  : n_iterations,
            "dataset"         : csv_path,
            "target_col"      : target_col,
            "best"            : best,
            "history"         : [
                {k: v for k, v in h.items() if k != "code"}   # omit bulky code
                for h in history
            ],
        }
        with open(results_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  Results saved to: {results_path}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save results: {exc}")

    return best


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-driven optimisation pipeline for cache replacement heuristics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", "-m",
        default="claude-haiku4.5",
        help="LLM model key (see LLMWrapper.ALL_LLM_MODELS for valid names)",
    )
    p.add_argument(
        "--iterations", "-n",
        type=int, default=10,
        help="Number of optimisation iterations",
    )
    p.add_argument(
        "--csv",
        default="datasets/perlbench_train_70.csv",
        help="Path to the training CSV (relative to project root)",
    )
    p.add_argument(
        "--target",
        default="decision",
        help="Name of the binary target column in the CSV",
    )
    p.add_argument(
        "--diversity-window",
        type=int, default=3,
        dest="diversity_window",
        help="Number of recent heuristics to inspect for diversity",
    )
    p.add_argument(
        "--results",
        default="scripts/pipeline_results.json",
        help="Where to write the JSON results file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        model_name       = args.model,
        n_iterations     = args.iterations,
        csv_path         = args.csv,
        target_col       = args.target,
        diversity_window = args.diversity_window,
        results_path     = args.results,
    )
