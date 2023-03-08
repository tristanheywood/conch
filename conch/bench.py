import math
from functools import partial
from itertools import product

import numpy as np
from triton.testing import do_bench
from triton.compiler import OutOfResources


def grid_search(func,
                warmup=100,
                rep=100,
                min_val_prod=0,
                do_print=False,
                **meta_param_from_to):
    """
    Example usage:
    ```
    grid_search(partial(f, W, x), BLOCK_SIZE_N = (16, 1024), BLOCK_SIZE_M = (16, 1024))
    ```
    """

    mp_base_ranges = {
        name: (int(math.log2(start)), int(math.log2(stop)))
        for name, (start, stop) in meta_param_from_to.items()
    }

    assert all(2**start_base == start and 2**stop_base == stop for (
        start_base,
        stop_base), (start, stop) in zip(mp_base_ranges.values(
        ), meta_param_from_to.values())), "start and stop must be powers of 2"

    mp_ranges = {
        name: [2**i for i in range(start_base, stop_base + 1)]
        for name, (start_base, stop_base) in mp_base_ranges.items()
    }

    name_val_pairs = [[(name, val) for val in vals]
                      for name, vals in mp_ranges.items()]

    results = []
    max_val_prod = 1e10

    for mp_vals in product(*name_val_pairs):
        mp_vals = dict(mp_vals)
        val_prod = math.prod(mp_vals.values())

        if val_prod >= max_val_prod or val_prod < min_val_prod:
            continue

        try:
            median_us = do_bench(
                partial(func, **mp_vals), warmup=warmup, rep=rep)[0] * 1000
        except OutOfResources as e:
            if do_print:
                print(e.message)
            max_val_prod = val_prod
            continue

        if do_print:
            print(*[f"{k.lower()}={v}" for k, v in mp_vals.items()],
                  f": {median_us:.2f} us")

        results.append((mp_vals, median_us))

    return sorted(results, key=lambda elt: elt[1])


def results_to_df(results, top_n: int = 10):
    """
    Convert grid search results into a Pandas dataframe for displaying in Jupyter.
    """
    import pandas as pd
    return pd.DataFrame({
        **metparams, "Time (us)": rt
    } for metparams, rt in results).head(top_n)
