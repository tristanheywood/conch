from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import math
from functools import partial
from itertools import product
from typing import Callable, Dict, List
from warnings import warn

import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from triton.testing import do_bench
from triton.compiler import OutOfResources

_GPU_WARMUP_TENSOR = None
_GPU_WARMUP_RUNTIME = None


def _gpu_warmup_fn(t: torch.Tensor):
    return t @ t @ t @ t


def ensure_warm_gpu():
    global _GPU_WARMUP_TENSOR, _GPU_WARMUP_RUNTIME

    warmup_fn = lambda: _gpu_warmup_fn(_GPU_WARMUP_TENSOR)
    gpu_is_warm = lambda rtime: abs(rtime - _GPU_WARMUP_RUNTIME
                                    ) / _GPU_WARMUP_RUNTIME < 0.05

    if _GPU_WARMUP_TENSOR is None:
        _GPU_WARMUP_TENSOR = torch.empty((128, 128), device="cuda")
        _GPU_WARMUP_RUNTIME = np.median(running_times(warmup_fn, 4096)[-500:])
    else:
        rtime = running_times(warmup_fn, 1).item()

        # If the run-time of `warmup_fn` is within 5% of time taken after warmup, then
        # we assume the GPU is already warm.
        if gpu_is_warm(rtime):
            return

        for _ in range(8):
            rtime = np.median(running_times(warmup_fn, 512))

            if gpu_is_warm(rtime):
                return

        warn(
            "Timed-out attempting to warm-up GPU. Benchmark results may be inaccurate."
        )


def auto_bench(fn, min_rep=10):
    """
    Similar to `triton.testing.do_bench`, but does not require `warmup` and `rep` to be
    manually specified. The GPU will be automatically warmed-up if required, and `fn`
    will be benchmarked until the run-time measurements stabilize.
    """
    ensure_warm_gpu()

    bench_start = datetime.now()
    base = 0
    rtimes = None
    while (datetime.now() - bench_start).total_seconds() * 1000 < min_rep:
        rtimes = running_times(fn, 2**base)
        base += 1

    # TODO: Add some statistical tests (IQR, variance etc) to determine if the run-times
    # are stable.

    return np.median(rtimes)


def running_times(fn: Callable, n_repeats: int, units="ms") -> np.ndarray:
    """
    Execute `fn` `n_repeats` times and return the running times.
    """

    # Execute once to make sure the function is jitted and everything is initialized.
    fn()
    torch.cuda.synchronize()

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    start_events = [
        torch.cuda.Event(enable_timing=True) for i in range(n_repeats)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True) for i in range(n_repeats)
    ]

    for i in range(n_repeats):
        cache.zero_()
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    scalar = 1
    if units == "us":
        scalar = 1000
    elif units != "ms":
        raise ValueError("units must be 'ms' or 'us'")

    return np.array(
        [s.elapsed_time(e) * scalar for s, e in zip(start_events, end_events)])


@dataclass
class RunningTimeGMM:
    """
    A Gaussian Mixture Model fitted to the running times of a function.
    """
    gmm: GaussianMixture
    component_means: List[float]
    component_stds: List[float]
    rtimes_per_mean: Dict[float, int]
    iqr_per_mean: Dict[float, int]


def fit_gaussian_mixture(rtimes: List[float]) -> RunningTimeGMM:
    """
    When the same GPU function is executed thousands of times, the running times often
    to experience multiple regimes, such that a plot of the running times looks like
    a downward staircase.
    """
    if len(rtimes) < 64:
        raise ValueError("Not enough running times to analyse.")

    rtimes = np.array(rtimes)

    # Remove outliers - the 10 fastest and 10 slowest runs. Removing by percentiles might
    # remove an entire regime.
    rtimes = np.sort(rtimes)[10:-10]

    ## Fix a Gaussian Mixture model to the running times. We aim to fit one Gaussian to
    # each regime.
    prev_gm = None
    prev_score = None
    new_gm = GaussianMixture(n_components=1).fit(rtimes[:, None])
    new_score = new_gm.score(rtimes[:, None])

    # Keep adding more components while adding a new component improves the likelihood
    # by more than 10%.
    while prev_gm is None or (new_score - prev_score) / prev_score > 0.01:
        prev_gm = new_gm
        prev_score = new_score
        new_gm = GaussianMixture(n_components=prev_gm.n_components + 1).fit(
            rtimes[:, None])
        new_score = new_gm.score(rtimes[:, None])
    gm = prev_gm

    # Determine the most probable mixture component for each time measurement.
    # Shape: (n_times, n_components).
    component_probs = gm.predict_proba(rtimes[:, None])
    # Shape: (n_times,).
    rtime_components = np.argmax(component_probs, axis=1)

    # Sort the means of the mixture components by decreasing mean.
    means = gm.means_.flatten()
    mean_order = np.argsort(means)[::-1]
    means = means[mean_order]
    rtime_components = mean_order[rtime_components]
    rtimes_per_component = np.unique(rtime_components, return_counts=True)[1]

    return RunningTimeGMM(
        gmm=gm,
        component_means=means,
        component_stds=np.sqrt(gm.covariances_.flatten()[mean_order]),
        rtimes_per_mean={
            mean: count
            for mean, count in zip(means, rtimes_per_component)
        },
        iqr_per_mean={
            mean: np.subtract(
                *np.percentile(rtimes[rtime_components == i], [75, 25]))
            for i, mean in enumerate(means)
        })


class MetaParamGrid:
    """Enables iteration over a grid of meta parameters - as required for a grid search."""
    def __init__(self,
                 verbose=False,
                 min_val_prod: int = 0,
                 max_val_prod: int = 1e10,
                 **meta_param_from_to):
        self.verbose = verbose
        self.min_val_prod = min_val_prod
        self.max_val_prod = max_val_prod

        mp_base_ranges = {
            name: (int(math.log2(start)), int(math.log2(stop)))
            for name, (start, stop) in meta_param_from_to.items()
        }

        assert all(2**start_base == start and 2**stop_base == stop
                   for (start_base, stop_base), (start, stop) in zip(
                       mp_base_ranges.values(), meta_param_from_to.values())
                   ), "start and stop must be powers of 2"

        mp_ranges = {
            name: [2**i for i in range(start_base, stop_base + 1)]
            for name, (start_base, stop_base) in mp_base_ranges.items()
        }

        self.name_val_pairs = [[(name, val) for val in vals]
                               for name, vals in mp_ranges.items()]

    def clipped_next(self):
        """Return the next set of meta parameter values in the grid, skipping any values
        whose product is not within [self.min_val_prod, self.max_val_prod]."""

        mp_vals = next(self.grid_gen)
        val_prod = math.prod(mp_vals.values())

        if val_prod > self.min_val_prod and val_prod < self.max_val_prod:
            return mp_vals
        else:
            return self.clipped_next()

    def __next__(self):
        mp_vals = self.clipped_next()

        @contextmanager
        def safely_get_next():
            try:
                yield mp_vals
            except OutOfResources as e:
                if self.verbose:
                    print(e.message)
                self.max_val_prod = math.prod(mp_vals.values())

        return safely_get_next

    def __iter__(self):
        self.grid_gen = (dict(mp_vals)
                         for mp_vals in product(*self.name_val_pairs))
        return self

    @contextmanager
    def safe_next(self):
        """Context manager that yields the next set of values, while also catching
        Triton's OOM exception and using it to set the new `max_val_prod` for the grid.
        
        Example usage:

        with grid.safe_next() as mp_vals:
            rtime = dispatch_kernel(**mp_vals)
        """

        try:
            mp_vals = self.clipped_next()
            yield mp_vals
        except OutOfResources as e:
            if self.verbose:
                print(e.message)
            self.max_val_prod = math.prod(mp_vals.values())


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


def auto_grid_search(fn, grid: MetaParamGrid, do_print=False):

    results = []

    for get_next in grid:
        with get_next() as mp_vals:
            rtime = auto_bench(partial(fn, **mp_vals))

        rtime = 1000 * rtime  # ms -> us.

        if do_print:
            print(*[f"{k.lower()}={v}" for k, v in mp_vals.items()],
                  f": {rtime:.2f} us")

        results.append((mp_vals, rtime))

    return sorted(results, key=lambda elt: elt[1])


def results_to_df(results, top_n: int = 10):
    """
    Convert grid search results into a Pandas dataframe for displaying in Jupyter.
    """
    import pandas as pd
    return pd.DataFrame({
        **metparams, "Time (us)": rt
    } for metparams, rt in results).head(top_n)
