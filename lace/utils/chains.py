"""Diagnostics shared by ensemble-sampler post-processing."""

import numpy as np


def purge_chains(ln_prob_chains, nsplit=4, abs_diff=15.0, minval=None):
    """Flag walkers with stable and sufficiently high log-probability chunks.

    Each walker history is split into ``nsplit`` consecutive chunks along the
    step axis and the mean log probability of every chunk is computed. A
    walker is kept only if (i) every chunk mean deviates from the walker's
    mean of chunk means by strictly less than ``abs_diff`` and (ii) every chunk
    mean is strictly above ``minval``.

    Parameters
    ----------
    ln_prob_chains : array_like, shape (n_steps, n_walkers)
        Log-probability values with sampling steps on axis zero and walkers on
        axis one.
    nsplit : int, default: 4
        Number of consecutive step chunks used for the stability check.
    abs_diff : float, default: 15.0
        Maximum allowed absolute deviation of a chunk mean from the walker's
        mean of chunk means.
    minval : float, optional
        Strict lower threshold applied to every chunk mean. If omitted, use
        ``np.median(ln_prob_chains) - abs_diff``, i.e. a threshold relative to
        the median over all steps and walkers.

    Returns
    -------
    ndarray of bool, shape (n_walkers,)
        ``True`` for walkers satisfying both criteria.

    Notes
    -----
    Use :func:`numpy.flatnonzero` on the returned mask (and its negation) to
    obtain the indices of kept and rejected walkers.
    """
    ln_prob_chains = np.asarray(ln_prob_chains)
    if minval is None:
        minval = np.median(ln_prob_chains) - abs_diff

    # chunk means, shape (n_walkers, nsplit)
    chunks = np.array_split(ln_prob_chains, nsplit, axis=0)
    chunk_means = np.array([chunk.mean(axis=0) for chunk in chunks]).T
    walker_means = chunk_means.mean(axis=1)

    # step-dependence: chunk means do not drift from the walker mean
    stable = (np.abs(chunk_means - walker_means[:, np.newaxis]) < abs_diff).all(
        axis=1
    )
    # level: every chunk mean is above the target minimum
    high_enough = (chunk_means > minval).all(axis=1)

    return stable & high_enough
