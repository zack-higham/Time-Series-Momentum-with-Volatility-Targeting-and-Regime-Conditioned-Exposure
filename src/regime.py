"""Walk-forward two-state Markov-switching regime model.

Observation series (weekly, weeks ending Friday):
  R1 "trend state":  the cross-asset trend payoff. Each day d, every instrument
      with a 12-month signal contributes s_i * (r_{i,d} - rf_d) / sigma_i, using
      the signal and EWMA volatility from the latest month-end STRICTLY before d
      (the position actually held on day d). The payoff is the average over
      instruments: the daily return of an equal-risk 12m TSMOM portfolio with
      unit annual volatility per position. Weekly payoff = sum over the week.
      Before April 2008 it uses whichever instruments already have signals
      (burn-in), so the number of instruments rises from 4 to 10.
  R2 "market turbulence": weekly SPY excess return.

Model (Hamilton 1989): y_w = mu_{S_w} + e_w, e_w ~ N(0, sigma^2_{S_w}), with
S_w a two-state Markov chain. Fitted by maximum likelihood (EM) with random
restarts. At each month-end t the model is refitted on all weeks ending on or
before t, and the FILTERED probability P(S_w = j | y_1..y_w) of the last such
week is the regime signal for month t+1. Filtered probabilities use only past
and current observations. Smoothed probabilities, P(S_w = j | y_1..y_T), use
the whole sample and are never used for anything presented as a result.

State labels are arbitrary in estimation (state 0 in one fit can be state 1 in
the next), so they are re-identified at every refit: R1 "trending" is the
state with the higher mean payoff; R2 "turbulent" is the state with the
higher variance.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import HMM_FREQ, HMM_MIN_TRAIN_WEEKS, HMM_SEARCH_REPS, HMM_SEED, PRIMARY_LOOKBACK
from returns import daily_excess_returns
from signals import tsmom_signal
from volatility import ewma_vol

SCALE = 100.0   # fit in percent units: better-conditioned likelihood surface
# Diagnostic, added after the first Stage 5 run (NOT pre-registered): a fit in
# which either state has stay probability below 0.5 (expected duration under
# two weeks) is an outlier-absorbing solution, not a regime. Such refits are
# flagged; the primary V3 still uses them as pre-registered, and a labelled
# diagnostic variant replaces them with no overlay.
MIN_STAY_PROB = 0.5


def daily_trend_payoff(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-asset trend payoff and the number of instruments in it."""
    dex = daily_excess_returns(prices)
    vol = ewma_vol(dex)
    sig = tsmom_signal(prices, PRIMARY_LOOKBACK)
    month_ends = sig.index
    # Signal and vol fixed at each month-end, carried forward to the days of
    # the next month only: reindex to days, forward-fill, then shift one day
    # so the month-end's own return is earned by the previous position.
    held_sig = sig.reindex(dex.index).ffill().shift(1)
    held_vol = vol.loc[month_ends].reindex(dex.index).ffill().shift(1)
    contrib = held_sig * dex / held_vol
    n = contrib.notna().sum(axis=1)
    payoff = contrib.mean(axis=1)            # mean over available instruments
    out = pd.DataFrame({"payoff": payoff, "n_instruments": n})
    return out[out["n_instruments"] > 0]


def weekly_trend_payoff(prices: pd.DataFrame) -> pd.DataFrame:
    daily = daily_trend_payoff(prices)
    weekly = daily["payoff"].resample(HMM_FREQ).sum(min_count=1)
    n = daily["n_instruments"].resample(HMM_FREQ).min()
    return pd.DataFrame({"y": weekly, "n_instruments": n}).dropna()


def weekly_spy_excess(prices: pd.DataFrame) -> pd.DataFrame:
    x = daily_excess_returns(prices)["SPY"].dropna()
    weekly = (1 + x).resample(HMM_FREQ).prod(min_count=1) - 1
    return weekly.dropna().to_frame("y")


def fit_markov(y: np.ndarray, seed: int = HMM_SEED):
    """Two-state switching mean and variance; best of HMM_SEARCH_REPS restarts."""
    np.random.seed(seed)   # statsmodels draws restart values from numpy's global RNG
    model = sm.tsa.MarkovRegression(y * SCALE, k_regimes=2, trend="c", switching_variance=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(search_reps=HMM_SEARCH_REPS, disp=False)
    params = dict(zip(model.param_names, res.params))
    return model, res, params


def target_state(params: dict, rule: str) -> int:
    """Index of the state of interest: 'mean' -> higher mean, 'variance' -> higher variance."""
    key = "const" if rule == "mean" else "sigma2"
    return int(params[f"{key}[1]"] > params[f"{key}[0]"])


def walk_forward(weekly: pd.DataFrame, dates: pd.DatetimeIndex, rule: str,
                 min_weeks: int = HMM_MIN_TRAIN_WEEKS) -> pd.DataFrame:
    """Refit at each date on weeks ending on or before it; return the filtered
    probability of the target state for the last of those weeks, plus the
    fitted parameters (state-ordered: 'hi' is the target state)."""
    rows = []
    for t in dates:
        hist = weekly.loc[:t, "y"]
        if len(hist) < min_weeks:
            rows.append({"date": t, "p": np.nan, "n_weeks": len(hist)})
            continue
        _, res, params = fit_markov(hist.values)
        j = target_state(params, rule)
        k = 1 - j
        filtered = np.asarray(res.filtered_marginal_probabilities)
        rows.append({
            "date": t, "p": filtered[-1, j], "n_weeks": len(hist),
            "last_week": hist.index[-1],
            "mu_hi": params[f"const[{j}]"] / SCALE, "mu_lo": params[f"const[{k}]"] / SCALE,
            "sd_hi": np.sqrt(params[f"sigma2[{j}]"]) / SCALE,
            "sd_lo": np.sqrt(params[f"sigma2[{k}]"]) / SCALE,
            "stay_hi": params[f"p[{j}->{j}]"] if f"p[{j}->{j}]" in params
            else 1 - params[f"p[{j}->{k}]"],
            "stay_lo": params[f"p[{k}->{k}]"] if f"p[{k}->{k}]" in params
            else 1 - params[f"p[{k}->{j}]"],
            "llf": res.llf,
        })
        rows[-1]["degenerate"] = min(rows[-1]["stay_hi"], rows[-1]["stay_lo"]) < MIN_STAY_PROB
    return pd.DataFrame(rows).set_index("date")


def full_sample_probabilities(weekly: pd.DataFrame, rule: str) -> pd.DataFrame:
    """ONE fit on the whole sample: filtered and smoothed probabilities of the
    target state. Uses future data by construction; for the lookahead
    illustration only."""
    _, res, params = fit_markov(weekly["y"].values)
    j = target_state(params, rule)
    return pd.DataFrame({
        "filtered_fullparams": np.asarray(res.filtered_marginal_probabilities)[:, j],
        "smoothed": np.asarray(res.smoothed_marginal_probabilities)[:, j],
    }, index=weekly.index)


def exposure_multiplier(p: pd.Series, floor: float) -> pd.Series:
    """m = floor + p, in [floor, floor + 1]."""
    return floor + p
