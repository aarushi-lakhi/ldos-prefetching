import heapq
from typing import Iterable, List

import numpy as np


def preprocess_future_indices(accesses: np.ndarray) -> np.ndarray:
    """
    Vector-friendly version of your original routine.
    Returns an int array `next_use[i]` = index of the *next* access
    to the same address, or len(accesses) if none.
    Memory cost 4-8 bytes per access (int32 or int64).
    """
    n = accesses.size
    next_use = np.empty(n, dtype=np.int64)          # int32 works if n < 2 B
    next_use.fill(n)

    last_occurrence = {}
    for idx in range(n - 1, -1, -1):                # reverse scan
        addr = int(accesses[idx])                   # ndarray scalar → py int
        next_use[idx] = last_occurrence.get(addr, n)
        last_occurrence[addr] = idx
    return next_use


def get_beladys_with_doa_labels(
    accesses: Iterable[int],
    cache_size: int,
    progress_every: int | None = 1_000_000,
) -> List[str]:
    """
    Same public contract as before, but ~10-100x faster on large traces.
    """
    # ---- Load once into a compact ndarray (fast & memory-efficient) ----
    accesses = np.fromiter(accesses, dtype=np.int64, count=-1)
    n = accesses.size
    next_use = preprocess_future_indices(accesses)

    # labels: -1 = not an insertion, 0 = Not Cached (DoA), 1 = Cached (reused)
    labels = np.full(n, -1, dtype=np.int8)

    # cache bookkeeping -------------------------------------------------
    # addr → (next_use_idx, reused_flag, insertion_pos)
    cache: dict[int, tuple[int, bool, int]] = {}

    # max-heap of (-next_use, addr) for O(log C) evictions
    heap: list[tuple[int, int]] = []                # (-next_use, addr)

    for i in range(n):
        if progress_every and i % progress_every == 0:
            print(f"processed {i:,}/{n:,} accesses")

        addr = int(accesses[i])
        nu   = int(next_use[i])

        # -------------------- HIT --------------------
        if addr in cache:
            _, _, ins = cache[addr]
            cache[addr] = (nu, True, ins)           # mark as reused
            heapq.heappush(heap, (-nu, addr))       # push new key
            continue

        # -------------------- MISS -------------------
        if len(cache) == cache_size:
            # Evict the line with the farthest next use.
            while heap:
                neg_nu, victim = heapq.heappop(heap)
                if victim in cache and cache[victim][0] == -neg_nu:
                    break                           # found the real current max
            v_nu, v_reused, v_ins = cache.pop(victim)
            labels[v_ins] = 1 if v_reused else 0    # record DoA / friendly

        cache[addr] = (nu, False, i)                # insert newcomer
        heapq.heappush(heap, (-nu, addr))

    # ------------------ DRAIN CACHE ------------------
    for (_, reused, ins) in cache.values():
        labels[ins] = 1 if reused else 0

    # Convert to required string output ---------------
    out = ["Cached"  if x == 1 else
           "Not Cached" if x == 0 else None
           for x in labels]
    return out


# ----------------------------------------------------------------------
# Small self-test identical to the original example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    access_seq = [5, 1, 3, 5, 2, 4, 1, 5, 5, 1, 3, 5, 2, 3, 4, 5]
    cache_sz   = 3
    doa_labels = get_beladys_with_doa_labels(access_seq, cache_sz)

    print("\nInsertion-point labels:")
    for idx, lbl in enumerate(doa_labels):
        if lbl is not None:
            print(f"idx {idx:2d}: label={lbl}")
