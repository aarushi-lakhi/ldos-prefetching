def preprocess_future_indices(accesses):
    """
    For every position i, return the index of the *next* reference to
    the same address (∞ if it never re-appears).  Same as your original.
    """
    next_access = {}
    last_occurrence = {}
    for idx in reversed(range(len(accesses))):
        addr = accesses[idx]
        next_access[idx] = last_occurrence.get(addr, float("inf"))
        last_occurrence[addr] = idx
    return next_access


def get_beladys_with_doa_labels(accesses, cache_size):
    """
    Returns a list `labels` such that labels[i] ∈ {0,1} and corresponds
    to the line *inserted* at access i.

        1 → line was reused ≥ 1 time before eviction (friendly)
        0 → line was evicted with 0 hits (dead-on-arrival)

    Accesses that *hit* an existing line are not insertions; their
    labels entry remains None.
    """
    future_idx = preprocess_future_indices(accesses)

    # addr → (next_use, reused_flag, insert_pos)
    cache = {}
    labels = [None] * len(accesses)

    for i, addr in enumerate(accesses):
        if i % 1_000_000 == 0:
            print(f"Processing access {i:,} / {len(accesses):,}")

        nxt = future_idx[i]

        # ---------- HIT ----------
        if addr in cache:
            # mark this line as reused
            cache[addr] = (nxt, True, cache[addr][2])
            continue

        # ---------- MISS ----------
        if len(cache) == cache_size:
            # choose Belady victim: farthest next use
            victim = max(cache, key=lambda a: cache[a][0])
            v_next, v_reused, v_ins = cache.pop(victim)
            labels[v_ins] = "Cached" if v_reused else "Not Cached"   # DoA label

        # insert incoming line
        cache[addr] = (nxt, False, i)

    # ---------- LINES LEFT AT END ----------
    for _, (__, reused, ins) in cache.items():
        labels[ins] = "Cached" if reused else "Not Cached"

    return labels


if __name__ == "__main__":
    access_seq = [5, 1, 3, 5, 2, 4, 1, 5, 5, 1, 3, 5, 2, 3, 4, 5]
    cache_sz   = 3
    doa_labels = get_beladys_with_doa_labels(access_seq, cache_sz)

    print("Insertion-point labels:")
    for idx, lbl in enumerate(doa_labels):
        if lbl is not None:   # only positions that inserted a line
            print(f"idx {idx:2d}: label={lbl}")
