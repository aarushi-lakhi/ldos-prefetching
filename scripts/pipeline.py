"""
pipeline.py  —  Level-2 Data Agent LLM Optimisation Pipeline
             for Cache Replacement Heuristics

Each iteration runs a four-step agentic loop:

  Step 1 — Exploration Prompt:
    Ask the LLM to write a pandas analysis script that examines the training
    CSV for correlations, distributions, and patterns relevant to prediction.
    The LLM returns code inside <explore_data> XML tags.

  Step 2 — Sandbox Execution:
    Extract and run that script via subprocess (60 s timeout).
    Capture stdout (insights) and stderr (errors) separately.

  Step 3 — Generation Prompt:
    Feed the captured stdout back to the LLM alongside feature docs and prior
    feedback. The LLM returns its heuristic inside <heuristic> XML tags.

  Step 4 — Evaluation:
    Extract the heuristic, run it through evaluator.py, and log
    Balanced Accuracy + Raw Accuracy (Glider metric) for every iteration.

Diversity tracking prevents stagnation: if the last DIVERSITY_WINDOW
heuristics reference the same features, the LLM is asked for a new strategy.

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
import subprocess
import tempfile
from datetime import datetime

import pandas as pd

# ── Make sibling scripts importable when run from any directory ────────────
_SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
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
  row['type']            – Access type: 0=LOAD 1=RFO 2=PREFETCH 3=WRITE 4=TRANSLATION
  row['timestamp']       – Microsecond wall-clock timestamp of the access (int)

BANNED FEATURES — accessing these causes IMMEDIATE DISQUALIFICATION (fitness = 0):
  row['hit']             — BANNED: leaks whether the line is already cached
  row['victim_addr']     — BANNED: not available to a real replacement policy

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

    DOMAIN KNOWLEDGE (from prior statistical analysis)
    ---------------------------------------------------
    R² feature importance ranking for cache-reuse prediction on this workload:
      1. ip        (instruction pointer)  — highest correlation with reuse
      2. full_addr (memory address)       — second highest correlation
      3. type      (access type)          — moderate signal
      4. set / way                        — low but useful structural signal
      5. triggering_cpu / timestamp       — minimal signal
    Focus primarily on the RELATIONSHIP between ip and full_addr (e.g., page
    alignment, address-delta, or bitwise overlap).  Heuristics built around
    ip and full_addr generalise far better than those based on type alone.

    CONTRACT
    --------
    • Return a single numeric score.
    • POSITIVE score  →  predict cache-friendly  (label = 1, keep the line)
    • ZERO / NEGATIVE →  predict cache-averse    (label = 0, evict the line)
    • Access features via row['feature_name'].
    • NEVER use row['hit'] or row['victim_addr'] — they are BANNED and will
      cause your submission to be IMMEDIATELY REJECTED with fitness = 0.
    • Your heuristic MUST predict BOTH classes (evict AND keep).
      Predicting only one class gives fitness = 0, regardless of accuracy.
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


def _extract_tag(text: str, tag: str) -> str | None:
    """Return the content between <tag>…</tag>, stripped, or None if absent."""
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Exploration prompt ─────────────────────────────────────────────────────

EXPLORE_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an AI architect. Before writing a cache replacement heuristic,
    you must analyze the data.

    The dataset is at '{csv_path}' and has these columns:
      {headers_str}

    The label is 'decision' (1 = cache-friendly / keep, ~97 % of rows;
    0 = cache-averse / evict, ~3 % of rows).

    Write a Python script using pandas to uncover patterns that predict 'decision'.
    Suggested analyses:
      • Value distributions of ip, full_addr, type, set, way per class (0 vs 1)
      • Page-alignment match: (ip >> 12) == (full_addr >> 12) per class
      • How access type (0=LOAD,1=RFO,2=PREFETCH,3=WRITE,4=TRANSLATION) correlates with eviction
      • Bitwise low-order bits of ip and full_addr for each class

    IMPORTANT: The CSV is ~500 MB. Always use nrows=50000 to stay fast.
    Keep your script concise (under 40 lines) and print a clear summary.

    Output your code EXACTLY inside <explore_data> XML tags. Do not write the heuristic yet.

    <explore_data>
    import pandas as pd
    # your exploration code here
    </explore_data>
    """)


def _build_explore_prompt(headers: list[str], csv_path: str) -> str:
    return EXPLORE_PROMPT_TEMPLATE.format(
        csv_path    = csv_path,
        headers_str = ", ".join(headers),
    )


def _run_exploration(code: str, timeout: int = 60) -> tuple[str, str]:
    """Write *code* to temp_explore.py, run it from project root, return (stdout, stderr)."""
    explore_path = os.path.join(_SCRIPTS_DIR, "temp_explore.py")
    with open(explore_path, "w") as fh:
        fh.write(code)
    try:
        proc = subprocess.run(
            [sys.executable, explore_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_PROJECT_ROOT,
        )
        return proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return "", f"Exploration script timed out after {timeout}s — use nrows=50000 to limit data."
    except Exception as exc:
        return "", f"Failed to run exploration script: {exc}"


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


SEED_HEURISTIC = textwrap.dedent("""\
    def heuristic(row):
        # Features available: ip, full_addr, type, set, way
        # Goal: Return a positive score to keep the line, negative to evict.
        score = 0.0

        # Basic intuition: Loads (type 0) and RFOs (type 1) are often reused.
        if row['type'] == 0 or row['type'] == 1:
            score += 1.0
        else:
            score -= 1.0

        # Check if the memory address is on the same page as the instruction pointer.
        # This indicates strong spatial locality (code and data are near each other).
        ip_page   = row['ip']       >> 12
        addr_page = row['full_addr'] >> 12
        if ip_page == addr_page:
            score += 2.0

        return score
    """)


def _build_prompt(
    feedback: str,
    prev_code: str | None,
    diversity_warning: bool,
    iteration: int,
    exploration_stdout: str = "",
    exploration_stderr: str = "",
) -> str:
    parts = [SYSTEM_PROMPT.format(feature_doc=FEATURE_DOC)]

    # ── Exploration insights (this iteration's data analysis) ──────────────
    if exploration_stdout.strip():
        trunc = exploration_stdout[:3000]
        if len(exploration_stdout) > 3000:
            trunc += "\n... [output truncated]"
        parts.append(
            "\n--- DATA EXPLORATION INSIGHTS ---\n"
            "Your exploration script produced the following statistical output.\n"
            "Use these patterns to design your heuristic:\n\n"
            f"{trunc}\n"
        )
    elif exploration_stderr.strip():
        parts.append(
            "\n--- DATA EXPLORATION NOTE ---\n"
            "The exploration script encountered an error:\n"
            f"{exploration_stderr[:500]}\n"
            "Proceed using domain knowledge and the feature descriptions above.\n"
        )

    if feedback:
        parts.append(f"\n--- FEEDBACK FROM PREVIOUS ITERATION ---\n{feedback}\n")

    if prev_code:
        parts.append(
            "\n--- PREVIOUS HEURISTIC (for reference only) ---\n"
            f"```python\n{prev_code}\n```\n"
        )

    if diversity_warning:
        parts.append(
            "\n⚠️  DIVERSITY ALERT: Your last several heuristics used the same "
            "features and appear to be minor variations of each other.  "
            "Please explore a FUNDAMENTALLY DIFFERENT strategy.  Ideas:\n"
            "  • Page-number relationship: (row['ip'] >> 12) vs (row['full_addr'] >> 12)\n"
            "  • Address-delta reasoning: row['full_addr'] % (row['set'] + 1)\n"
            "  • Set-occupancy proxy via row['set'] and row['way'] together\n"
            "  • Bitwise overlap between ip and full_addr low-order bits\n"
            "  • Timestamp-based recency: row['timestamp'] % some_expression\n"
        )

    parts.append(
        "\nBased on the exploration insights above, write your improved heuristic "
        "body inside <heuristic> XML tags (no `def` line, no markdown):\n"
        "<heuristic>\n"
        "# your function body here\n"
        "</heuristic>"
    )
    return "\n".join(parts)


def _format_feedback(iteration: int, metrics: dict) -> str:
    """Convert an evaluation result dict into a natural-language feedback string."""
    lines = [
        f"Iteration {iteration} scores:",
        f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}"
        f"  [= avg(Cache-Averse accuracy, Cache-Friendly accuracy)]",
        f"  Precision          : {metrics['precision']:.4f}",
        f"  Recall             : {metrics['recall']:.4f}",
        f"  F1 Score           : {metrics['f1']:.4f}",
        f"  Cache-Averse  Accuracy (class 0): {metrics['acc_class0']:.4f}",
        f"  Cache-Friendly Accuracy (class 1): {metrics['acc_class1']:.4f}",
        f"  Hardcoded constants: {metrics['n_constants']}  "
        f"(threshold = 5; penalty factor = {metrics['complexity_penalty']:.4f})",
        f"  ► Final Fitness    : {metrics['final_fitness']:.4f}  "
        f"  [= Balanced Accuracy × complexity_penalty]",
    ]

    if metrics["errors"] > 0:
        pct = 100 * metrics["errors"] / max(metrics["n_samples"], 1)
        lines.append(
            f"\n  ⚠ Your heuristic raised exceptions on {metrics['errors']} rows "
            f"({pct:.1f} %) — those were treated as score = 0."
        )

    # Qualitative hints
    n_consts = metrics["n_constants"]
    if n_consts > CONSTANT_THRESHOLD:
        excess = n_consts - CONSTANT_THRESHOLD
        loss   = 1.0 - metrics["complexity_penalty"]
        lines.append(
            f"\n  CONSTANTS PENALTY: You used {n_consts} hardcoded numeric constants "
            f"({excess} over the limit of {CONSTANT_THRESHOLD}).  "
            f"This cost you {loss:.1%} of your fitness score.\n"
            f"  You MUST replace magic numbers with relational logic.  Examples:\n"
            f"    BAD : if row['type'] == 2 and row['way'] > 4\n"
            f"    GOOD: ip_page = row['ip'] >> 12; addr_page = row['full_addr'] >> 12\n"
            f"          if ip_page == addr_page:  # no magic numbers — pure relation\n"
            f"  If you reduce to fewer than {CONSTANT_THRESHOLD} constants, "
            f"your complexity penalty disappears and your score will improve significantly."
        )
    elif n_consts > 0:
        lines.append(
            f"\n  Constants used: {n_consts} (within the limit of {CONSTANT_THRESHOLD} — no penalty)."
        )

    # ── Majority/minority trap detection ─────────────────────────────────
    acc0 = metrics["acc_class0"]
    acc1 = metrics["acc_class1"]
    trap_evict_all = acc0 > 0.95 and acc1 < 0.05   # predicted all-0
    trap_keep_all  = acc1 > 0.95 and acc0 < 0.05   # predicted all-1

    if trap_evict_all or trap_keep_all:
        which = "EVICT EVERYTHING (all class 0)" if trap_evict_all else "KEEP EVERYTHING (all class 1)"
        lines.append(
            f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            f"\n  MAJORITY-CLASS TRAP DETECTED — Final Fitness = 0.0"
            f"\n  Your heuristic predicted {which}."
            f"\n  This is completely useless — a real cache policy must"
            f"\n  decide WHICH lines to evict, not blindly evict or keep all."
            f"\n  FINAL FITNESS IS ZERO UNTIL YOU PREDICT BOTH CLASSES."
            f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            f"\n"
            f"\n  You MUST write logic that sometimes returns a POSITIVE score"
            f"\n  (predict keep) AND sometimes returns a NEGATIVE/ZERO score"
            f"\n  (predict evict).  Start with a conditional branch, e.g.:"
            f"\n    if row['type'] == 2:   # PREFETCH — likely less useful"
            f"\n        return -1.0"
            f"\n    return row['ip'] % 3 - 1  # mix of positive and negative"
        )
    elif acc0 < 0.10:
        lines.append(
            "\n  WARNING: Cache-Averse accuracy is very low — you are rarely"
            " predicting evictions.  Add more conditions that return negative scores."
        )
    elif metrics["balanced_accuracy"] < 0.55:
        lines.append(
            "\n  Balanced Accuracy is below 0.55.  Try combining type, ip, set, "
            "full_addr, or timestamp with conditional logic to separate the classes."
        )
    else:
        lines.append(
            "\n  Good Balanced Accuracy.  Focus on reducing constants or refining "
            "edge cases to push both class accuracies higher."
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
    csv_path:         str  = "datasets/mcf_train_70.csv",
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

    # ── Read CSV headers once (fast: nrows=0) ─────────────────────────────
    try:
        csv_headers = list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception as exc:
        print(f"  [WARN] Could not read CSV headers: {exc}")
        csv_headers = ["triggering_cpu", "set", "way", "full_addr", "ip", "type", "timestamp", "decision"]

    history: list[dict] = []
    best    = {"final_fitness": -1.0, "code": None, "metrics": None, "iteration": -1}
    feedback    = ""
    prev_code   = None

    for i in range(1, n_iterations + 1):
        print(f"── Iteration {i:>2}/{n_iterations} ──────────────────────────")

        low_div = _is_low_diversity(history, diversity_window)
        if low_div:
            print("  [diversity] Low-diversity signal — prompting for new strategy")

        # ── STEP 1: Exploration Prompt ────────────────────────────────────
        print("  [Step 1/4] Requesting data exploration script…")
        explore_prompt = _build_explore_prompt(csv_headers, csv_path)
        try:
            explore_raw = llm.send_pdf(explore_prompt, None)
        except Exception as exc:
            print(f"  [ERROR] Exploration LLM call failed: {exc}")
            explore_raw = ""

        explore_code = _extract_tag(explore_raw, "explore_data")
        if not explore_code:
            print("  [WARN] No <explore_data> tag found — skipping sandbox")
            explore_stdout, explore_stderr = "", "LLM did not produce an <explore_data> block."
        else:
            # ── STEP 2: Sandbox Execution ─────────────────────────────────
            print("  [Step 2/4] Running exploration script in sandbox…")
            explore_stdout, explore_stderr = _run_exploration(explore_code)
            if explore_stderr:
                print(f"  [explore stderr] {explore_stderr[:200]}")
            preview = explore_stdout[:300].replace("\n", " ")
            print(f"  [explore stdout] {preview}{'…' if len(explore_stdout) > 300 else ''}")

        # ── STEP 3: Generation Prompt ─────────────────────────────────────
        print("  [Step 3/4] Requesting heuristic from exploration insights…")
        gen_prompt = _build_prompt(
            feedback           = feedback,
            prev_code          = prev_code,
            diversity_warning  = low_div,
            iteration          = i,
            exploration_stdout = explore_stdout,
            exploration_stderr = explore_stderr,
        )
        try:
            gen_raw = llm.send_pdf(gen_prompt, None)
            code    = _extract_tag(gen_raw, "heuristic") or _strip_markdown(gen_raw)
        except Exception as exc:
            print(f"  [ERROR] Generation LLM call failed: {exc}")
            feedback = f"Iteration {i}: LLM call failed ({exc}). Please try again."
            history.append({"iteration": i, "code": "", "metrics": None, "hash": "",
                            "explore_stdout": explore_stdout[:500]})
            continue

        fp = _fingerprint(code)
        print(f"  Code fingerprint : {fp}")

        # ── STEP 4: Evaluate ──────────────────────────────────────────────
        print("  [Step 4/4] Evaluating heuristic…")
        metrics = evaluate_heuristic(code, csv_path=csv_path, target_col=target_col)

        if "error" in metrics:
            print(f"  [ERROR] Evaluator: {metrics['error']}")
            if "BANNED FEATURES" in metrics["error"]:
                feedback = (
                    f"Iteration {i}: SUBMISSION REJECTED — banned features detected.\n"
                    f"{metrics['error']}\n\n"
                    "You are FORBIDDEN from using row['hit'] or row['victim_addr'].\n"
                    "These are inadmissible — a deployed policy cannot access them.\n"
                    "Rewrite your heuristic using ONLY: "
                    "triggering_cpu, set, way, full_addr, ip, type, timestamp."
                )
            else:
                feedback = (
                    f"Iteration {i}: Your heuristic could not be evaluated.\n"
                    f"Error: {metrics['error']}\n"
                    "Please write syntactically valid Python that returns a number."
                )
        else:
            print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
            print(f"  Precision         : {metrics['precision']:.4f}")
            print(f"  Recall            : {metrics['recall']:.4f}")
            print(f"  F1 Score          : {metrics['f1']:.4f}")
            print(f"  Cache-Averse  Acc : {metrics['acc_class0']:.4f}  (class 0)")
            print(f"  Cache-Friendly Acc: {metrics['acc_class1']:.4f}  (class 1)")
            print(f"  Constants (#)     : {metrics['n_constants']}")
            print(f"  Complexity penalty: {metrics['complexity_penalty']:.4f}")
            print(f"  ► Final Fitness   : {metrics['final_fitness']:.4f}")

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
            "iteration"      : i,
            "code"           : code,
            "metrics"        : metrics,
            "hash"           : fp,
            "explore_stdout" : explore_stdout[:500],
        })

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OPTIMISATION COMPLETE")
    print("=" * 60)

    if best["code"] is not None:
        m = best["metrics"]
        print(f"  Best iteration    : {best['iteration']}")
        print(f"  Final Fitness     : {best['final_fitness']:.4f}")
        print(f"  Balanced Accuracy : {m['balanced_accuracy']:.4f}")
        print(f"  Precision         : {m['precision']:.4f}")
        print(f"  Recall            : {m['recall']:.4f}")
        print(f"  F1 Score          : {m['f1']:.4f}")
        print(f"  Cache-Averse  Acc : {m['acc_class0']:.4f}  (class 0)")
        print(f"  Cache-Friendly Acc: {m['acc_class1']:.4f}  (class 1)")
        print(f"  Constants         : {m['n_constants']}")
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
        default="datasets/mcf_train_70.csv",
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
