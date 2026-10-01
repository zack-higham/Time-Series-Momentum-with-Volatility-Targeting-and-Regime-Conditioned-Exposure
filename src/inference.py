"""Statistical inference: Sharpe-ratio differences and spanning regressions.

Two strategy variants built from the same signals are highly correlated, so
comparing their Sharpe ratios as if they were independent overstates the
uncertainty in some cases and understates it in others. The test here
resamples the PAIR of monthly return series jointly with the stationary
bootstrap of Politis and Romano (1994): random-length blocks (geometric, mean
length 6 months) preserve both the cross-correlation between the two series
and their autocorrelation and volatility clustering. The p-value is
two-sided, from the bootstrap distribution of the difference re-centred on
zero (the null of equal Sharpe ratios).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import BOOTSTRAP_MEAN_BLOCK_MONTHS, BOOTSTRAP_REPS, BOOTSTRAP_SEED, newey_west_lags


def sharpe(x: np.ndarray) -> float:
    return x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(12)


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of positions 0..n-1: blocks start at random points, each
    step continues the block with probability 1 - 1/mean_block, wrapping
    around the end of the sample."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < 1.0 / mean_block
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if new_block[i] else (idx[i - 1] + 1) % n
    return idx


def sharpe_difference_test(a: pd.Series, b: pd.Series, reps: int = BOOTSTRAP_REPS,
                           mean_block: float = BOOTSTRAP_MEAN_BLOCK_MONTHS,
                           seed: int = BOOTSTRAP_SEED) -> dict:
    """Test H0: Sharpe(a) = Sharpe(b) on aligned monthly returns."""
    pair = pd.concat([a, b], axis=1).dropna().values
    n = len(pair)
    observed = sharpe(pair[:, 0]) - sharpe(pair[:, 1])
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        sample = pair[stationary_bootstrap_indices(n, mean_block, rng)]
        s = sharpe(sample)
        diffs[r] = s[0] - s[1]
    p_value = np.mean(np.abs(diffs - observed) >= abs(observed))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"sharpe_a": sharpe(pair[:, 0]), "sharpe_b": sharpe(pair[:, 1]),
            "diff": observed, "ci_low": lo, "ci_high": hi, "p_value": p_value,
            "corr": np.corrcoef(pair.T)[0, 1], "months": n}


def spanning_regression(y: pd.Series, factors: pd.DataFrame) -> dict:
    """y = alpha + beta' factors + e, monthly; Newey-West t-statistics.

    Alpha is reported annualised (x12). If the factors are tradable excess
    returns, alpha is the return y earns beyond what a static, constant-beta
    combination of them would have earned.
    """
    df = pd.concat([y.rename("y"), factors], axis=1).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df[factors.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(df))})
    out = {"alpha_ann_pct": fit.params["const"] * 1200, "alpha_t": fit.tvalues["const"],
           "r2": fit.rsquared, "months": int(fit.nobs)}
    for f in factors.columns:
        out[f"beta_{f}"] = fit.params[f]
        out[f"t_{f}"] = fit.tvalues[f]
    return out


def deflated_sharpe_ratio(monthly: pd.Series, trial_sharpes_m: np.ndarray,
                          n_trials: int | None = None) -> dict:
    """Bailey and Lopez de Prado (2014) deflated Sharpe ratio.

    Probability that the true Sharpe ratio exceeds SR0, the maximum Sharpe
    expected from N independent trials with zero true Sharpe and the observed
    cross-trial variance of Sharpe estimates. All Sharpe ratios are monthly
    (non-annualised); skewness and kurtosis correct for non-normal returns.
    n_trials overrides the count (e.g. an effective number of independent
    trials) while keeping the observed cross-trial dispersion.
    """
    from scipy.stats import norm
    m = monthly.dropna()
    t = len(m)
    sr = m.mean() / m.std()
    g3, g4 = m.skew(), m.kurt() + 3
    n = len(trial_sharpes_m) if n_trials is None else n_trials
    euler = 0.5772156649
    sd_trials = np.std(trial_sharpes_m, ddof=1)
    sr0 = sd_trials * ((1 - euler) * norm.ppf(1 - 1 / n) + euler * norm.ppf(1 - 1 / (n * np.e)))
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    z = (sr - sr0) * np.sqrt(t - 1) / denom
    psr0 = norm.cdf(sr * np.sqrt(t - 1) / denom)   # probabilistic Sharpe vs 0, no deflation
    return {"n_trials": n, "sr_ann": sr * np.sqrt(12), "sr0_ann": sr0 * np.sqrt(12),
            "trial_sr_sd_ann": sd_trials * np.sqrt(12), "psr_vs_0": psr0, "dsr": norm.cdf(z)}
