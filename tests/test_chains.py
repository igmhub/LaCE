import numpy as np

from lace.utils.chains import purge_chains


def _synthetic_chain():
    """Converged walkers plus one drifting and one low (stuck) walker."""
    rng = np.random.default_rng(1)
    n_steps, n_walkers = 203, 7  # unequal axes, n_steps not divisible by nsplit
    chain = -50.0 + rng.normal(scale=1.0, size=(n_steps, n_walkers))
    chain[:, 2] = -200.0 + rng.normal(scale=1.0, size=n_steps)  # stuck low
    chain[:, 5] += np.linspace(-60.0, 0.0, n_steps)  # still burning in
    return chain


def test_purge_chains_median_threshold():
    keep = purge_chains(_synthetic_chain(), nsplit=4, abs_diff=15)
    assert keep.dtype == bool
    assert keep.shape == (7,)
    np.testing.assert_array_equal(np.flatnonzero(~keep), [2, 5])


def test_purge_chains_fixed_threshold():
    chain = _synthetic_chain()
    # the fixed threshold is below the stuck walker: only drift is rejected
    keep = purge_chains(chain, nsplit=5, abs_diff=5, minval=-1000)
    np.testing.assert_array_equal(np.flatnonzero(~keep), [5])
    # a threshold above the stuck walker also rejects it
    keep = purge_chains(chain, nsplit=5, abs_diff=5, minval=-100)
    np.testing.assert_array_equal(np.flatnonzero(~keep), [2, 5])
