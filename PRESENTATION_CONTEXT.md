# Project Context for Presentation

## What This Project Is

This is a research project that uses LLMs as **autonomous agents to discover cache replacement heuristics** for CPU memory caches. The goal is to automatically engineer a Python function (a "heuristic") that predicts whether a cache line should be kept or evicted — and to beat **Glider**, a published ML-based cache replacement policy (Shi et al. 2019, raw accuracy: **81.14%** on the mcf benchmark).

The broader project (`Joint Learning the Cache and Prefetcher`) is about joint ML optimization of OS components. The scripts in the `scripts/` folder represent the LLM-driven heuristic discovery pipeline, which is a self-contained subproject within that.

---

## The Dataset

- **Source**: SPEC CPU 2006 benchmarks (perlbench, gcc, mcf), collected via the ChampSim simulator
- **What it captures**: Each row is a cache access event. The label (`decision`) encodes what Belady's OPT algorithm (the theoretical optimal policy) would do.
- **Class distribution**: ~80% label 1 (keep/cache-friendly), ~20% label 0 (evict/cache-averse)
- **Available features**:
  - `ip` — instruction pointer (strongest signal for reuse prediction)
  - `full_addr` — full 64-bit memory address (second strongest)
  - `type` — access type: 0=LOAD, 1=RFO, 2=PREFETCH, 3=WRITE, 4=TRANSLATION
  - `set`, `way` — cache structural info
  - `triggering_cpu`, `timestamp` — minimal signal
- **Banned features** (inadmissible leakage, enforced by evaluator):
  - `hit` — leaks whether the line is already cached
  - `victim_addr` — not available at real decision time

---

## The Evaluation System (`evaluator.py`)

Each heuristic is a Python function `heuristic(row) → numeric_score`:
- **Positive score** → predict keep (label 1)
- **Zero/negative score** → predict evict (label 0)

**Metrics computed:**
| Metric | Description |
|--------|-------------|
| Raw Accuracy | Standard accuracy; the Glider comparison metric |
| Balanced Accuracy | Average of per-class recall; immune to majority-class trap |
| Final Fitness | `balanced_accuracy × complexity_penalty` |
| Complexity Penalty | Penalizes >10 hardcoded numeric constants (10% off per extra constant) |
| acc_class0 / acc_class1 | Per-class accuracy (evict / keep) |

**Key design decisions in the evaluator:**
- Stratified 50k-row sample for speed (~5s vs hours on full 7M-row dataset)
- Dual-class gate: if either class accuracy < 5%, Final Fitness = 0 (prevents "always keep/evict" cheating)
- Banned feature check via regex + hard column deletion before heuristic runs
- Complexity penalty discourages magic-number-heavy solutions that overfit

---

## The Pipeline — Three Levels of Evolution

### Level 0: Baseline (Manual / Static)
Simple manually written heuristics, evaluated once. The seed heuristic used as a starting point:
```python
def heuristic(row):
    score = 0.0
    if row['type'] == 0 or row['type'] == 1:  # LOADs and RFOs are often reused
        score += 1.0
    else:
        score -= 1.0
    ip_page   = row['ip']        >> 12
    addr_page = row['full_addr'] >> 12
    if ip_page == addr_page:    # spatial locality
        score += 2.0
    return score
```

---

### Level 1: S-Search (Stochastic Search / Basic LLM Feedback Loop)
**What it does:** A single LLM (Claude Haiku 4.5) iteratively writes heuristics. After each evaluation, the LLM receives structured feedback (scores, per-class accuracy, warnings about majority-class traps) and tries to improve. This is called **S-Search (Stochastic Search)** because the LLM samples a different heuristic on every call — results vary across runs.

**Structure (per iteration):**
1. Build prompt with feature docs + feedback from previous iteration
2. LLM writes heuristic inside `<heuristic>` XML tags
3. Evaluator scores it
4. Format feedback → next iteration

**Key prompt engineering:**
- Explicit feature importance ranking (ip > full_addr > type > set/way > timestamp)
- Banning of `hit` and `victim_addr` with IMMEDIATE DISQUALIFICATION language
- Diversity tracking: if last 3 heuristics use identical features → send diversity warning
- Penalties explained in feedback with concrete examples of good vs bad logic

**Issue discovered:** Early version was "cheating" (passing inadmissible features). Fixed in commit `d76f3aa`.

**Best observed result (single run, iteration 8 of 50):**
- Raw Accuracy: **82.50%** (+1.36% over Glider)
- Balanced Accuracy: **86.89%**

> **Stochastic caveat:** Because the pipeline is a random search, results vary between runs. The figure above comes from one particularly successful run and should be treated as an approximate upper bound until more trials are collected.

**Best heuristic from that run:**
```python
ip = row['ip']
full_addr = row['full_addr']
access_type = row['type']
way = row['way']

if access_type == 2:       # PREFETCH — speculative, often not reused
    return -1
if access_type == 4:       # TRANSLATION — minimal reuse
    return -0.5

ip_page   = ip        >> 12
addr_page = full_addr >> 12
if ip_page == addr_page:   # same page → strong spatial locality
    return 3

addr_delta = abs(full_addr - ip)
if addr_delta < 256:       # very close addresses → tight coupling
    return 2

xor_val = ip ^ full_addr
hamming = bin(xor_val).count('1')
if hamming < 10:    return 1.5   # strong bit alignment
if hamming < 20:    return 0.8   # moderate alignment

if access_type == 1:    return 0.2
elif access_type == 0:  return -0.2
return -0.8
```

---

### Level 2: Data Agent (Current)
**What it changed:** Before writing any heuristic, the pipeline runs a **fixed deterministic data exploration** on the training CSV. The statistical insights are injected into every LLM generation prompt, so the LLM reasons from real data patterns rather than pure domain knowledge.

**Why deterministic exploration instead of LLM-written scripts:** An earlier design had the LLM write its own pandas analysis script, which was then sandboxed and run. This failed on every iteration because the `ip`/`full_addr` columns store 64-bit addresses in a dtype that pandas couldn't bitshift as a Series in this environment. Replacing that with a fixed pre-written exploration solved the problem entirely.

**2-step loop per iteration (after a one-time pre-run exploration):**
```
Pre-run (once before the loop):
  Fixed Python script reads 50k rows and computes:
  - Class distribution and per-type eviction rates
  - Page alignment rate per class: (ip>>12) == (full_addr>>12)
  - Address delta |ip - full_addr| statistics per class
  - Way index distribution per class
  - XOR Hamming weight popcount(ip ^ full_addr) per class

Step 1 — Generation Prompt:
  LLM sees exploration insights + feature docs + prior feedback
  → writes heuristic inside <heuristic> XML tags.

Step 2 — Evaluate:
  evaluator.py scores the heuristic. Metrics logged + best tracked.
```

**Why this matters:** The LLM can ground its logic in actual data statistics (e.g., that on the mcf trace, virtually all evictions are type=LOAD while RFO/TRANSLATION are almost always kept, and that Hamming weight of `ip ^ full_addr` separates the classes strongly) rather than guessing from documentation alone. Like Level 1, results still vary across runs because the LLM generation step is stochastic.

**Best observed result (single run, iteration 44 of 50):**
- Raw Accuracy: **86.61%** (+5.47% over Glider)
- Balanced Accuracy: **89.49%**
- Cache-Averse Acc: 97.92% | Cache-Friendly Acc: 81.06%

> **Upper bound caveat:** This is the best single result observed so far — one particularly successful run. Because the pipeline is stochastic, this should be treated as an approximate upper bound until more runs are collected to establish a reliable mean.

**Best heuristic found (iteration 42, highest Final Fitness = 0.8596):**
```python
access_type = row['type']

# RFO (1) and TRANSLATION (4) are almost never evicted → strong keep signal
if access_type == 1 or access_type == 4:
    return 100

ip = row['ip']
addr = row['full_addr']
way_idx = row['way']

# Primary signal: XOR Hamming weight between ip and addr
# Evict median=26, Keep median=13 → strongest differentiator from exploration data
xor_val = ip ^ addr
hamming = bin(xor_val).count('1')

# Secondary signal: Way index (Evict mean=1.48, Keep mean=4.58 from exploration data)
way_signal = way_idx

# Hamming score: normalize around observed median split (26 vs 13)
hamming_score = 26 - hamming

# Way score: threshold naturally between means (1.48 and 4.58)
way_score = (way_signal - 3) * 3

# Combine signals
score = hamming_score + way_score

return score
```

**What this heuristic shows:** The LLM directly used the exploration statistics — it cited the exact Hamming weight medians (26 for evict, 13 for keep) and way index means (1.48 vs 4.58) from the pre-run data analysis. This is the Level 2 mechanism working as intended.

---

## Results

| Pipeline Level | Raw Accuracy | vs. Glider (81.14%) | Notes |
|----------------|-------------|---------------------|-------|
| Glider baseline (Shi et al. 2019) | 81.14% | — | Published result |
| Level 1 (S-Search) | 82.50% | +1.36% | Best observed; stochastic — varies per run |
| **Level 2 (Data Agent)** | **86.61%** | **+5.47%** | Best observed; stochastic — treat as upper bound |

> **On stochastic variability:** Both pipelines are random searches — the LLM samples different heuristics each call. A single run's best result is not a reliable estimate of true performance. Multiple runs are needed to establish a mean and confidence interval. The figures above are the best results observed so far across all runs.

**Level 1 best run details:**
- Model: Claude Haiku 4.5 (via AWS Bedrock)
- Iterations: 50 | Best iteration: #8
- Raw Accuracy: 82.50% | Balanced Accuracy: 86.89%

**Level 2 best run details:**
- Model: Claude Haiku 4.5 (via AWS Bedrock)
- Iterations: 50 | Best iteration by raw accuracy: #44 | Best by Final Fitness: #42
- Raw Accuracy: 86.61% (iter 44) | Balanced Accuracy: 89.49% (iter 44)
- Final Fitness: 0.8596 (iter 42) — uses only 6 hardcoded constants (no penalty)

---

## Tech Stack

- **LLM**: Claude Haiku 4.5 via AWS Bedrock (`BedrockWrapper`)
- **LLM Abstraction**: `LLMWrapper.py` — supports Claude, Gemini, GPT, Llama, Nova, DeepSeek, Qwen via a unified interface
- **Dataset simulator**: ChampSim (open source CPU simulator)
- **Data labeling**: Belady's OPT algorithm (`add_labels.py`)
- **Evaluation**: Custom `evaluator.py` with sklearn metrics
- **Language**: Python (pandas, numpy, sklearn)

---

## Key Insights / Story Arc for Presentation

1. **The problem**: Cache replacement is hard. Belady's OPT is optimal but requires future knowledge. Real policies (LRU, etc.) are heuristic. ML approaches like Glider exist but require custom training.

2. **Our approach**: Use LLMs to *automatically discover* heuristics as Python code — no model training, just prompt engineering + evaluation.

3. **Level 1 result**: S-Search already beats Glider in its best observed run (82.50% vs 81.14%). But because it's stochastic, it doesn't beat Glider *reliably* — some runs land below the baseline.

4. **Level 2 goal**: Inject real data statistics into every prompt so the LLM makes *informed* decisions rather than guessing. The fixed deterministic exploration now provides page-alignment rates, per-type eviction rates, address deltas, and Hamming weight distributions before each heuristic is generated.

5. **Open question**: Does grounded data context make the LLM's search more reliable (higher floor across runs), or does raw stochastic luck dominate? More trials needed.
