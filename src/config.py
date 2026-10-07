"""Research design, fixed before any strategy result was computed.

Every parameter below was chosen from the literature or from data availability,
and committed to the repository before the first backtest was run. Results are
reported for these settings; alternatives appear only as labelled robustness
checks, never as replacements chosen after seeing the numbers.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
PAPER_DIR = ROOT / "paper"

# --- Universe ----------------------------------------------------------------
# Ten liquid US-listed ETFs spanning four asset classes. FX uses three
# differentiated exposures (broad dollar, yen as safe haven, Australian dollar
# as a risk-on/commodity currency) rather than UUP plus FXE, which would be
# close to the same trade twice (EUR is ~58% of the dollar index).
INSTRUMENTS = {
    "SPY": "Equity",       # S&P 500
    "EFA": "Equity",       # MSCI EAFE (developed ex-US)
    "EEM": "Equity",       # MSCI Emerging Markets
    "IEF": "Rates",        # 7-10y US Treasuries
    "TLT": "Rates",        # 20y+ US Treasuries
    "GLD": "Commodity",    # physical gold
    "DBC": "Commodity",    # diversified commodity futures basket
    "UUP": "FX",           # long USD vs DXY basket (futures-based)
    "FXY": "FX",           # long JPY vs USD
    "FXA": "FX",           # long AUD vs USD
}
ASSET_CLASSES = ("Equity", "Rates", "Commodity", "FX")

# --- Sample ------------------------------------------------------------------
DOWNLOAD_START = "2002-01-01"   # earliest history, used for warm-up and HMM burn-in
# UUP starts 2007-03-01, so its first 12-month signal is at end-March 2008.
# First month in which all ten instruments hold a position:
MAIN_START = "2008-04"
SAMPLE_END = "2026-09-30"       # last complete month: September 2026
# Two equal halves of the 222-month main sample, for a stability check.
SUBPERIOD_SPLIT = "2017-07"     # second half starts here

# --- Risk-free rate ----------------------------------------------------------
# Yahoo ^IRX (13-week T-bill, annualised %). Converted to a per-period return as
# rate/100 * calendar_days/360, using the rate known at the start of the period.
# Cross-checked against the Ken French daily RF series over their overlap.
RF_TICKER = "^IRX"

# --- Signal ------------------------------------------------------------------
LOOKBACKS_MONTHS = (1, 3, 6, 12)  # each tested independently
PRIMARY_LOOKBACK = 12             # Moskowitz, Ooi and Pedersen (2012) headline
# Pre-specified alternative: equal-weight blend of the four lookback signals.
# Signal = sign of the trailing cumulative EXCESS return (over the T-bill).
PREDICTIVE_REG_MAX_LAG = 24       # months, pooled predictive regressions

# --- Volatility estimation and targeting -------------------------------------
EWMA_COM_DAYS = 60          # centre of mass, as in MOP (2012)
ANNUALISATION_DAYS = 252
VOL_MIN_OBS = 120           # daily observations before an estimate is used
CORR_WINDOW_DAYS = 252      # trailing window for the correlation matrix
CORR_MIN_OBS = 126
PORTFOLIO_VOL_TARGET = 0.10  # annualised, ex ante
# Risk allocation: equal ex-ante risk per ASSET CLASS (25% each), split equally
# across the instruments within a class. Within-class correlations are high
# (equity ~1.3 and rates ~1.1 effective bets), so equal risk per instrument
# would hand a cluster of near-duplicates a multiple of its diversification
# value. Revised from per-instrument before any result was computed.
# Robustness: equal risk per instrument (MOP convention; AQR factor comparison).
RISK_ALLOCATION = "asset_class"
RISK_ALLOCATION_SENSITIVITY = ("instrument",)
GROSS_LEVERAGE_CAP = None    # primary: uncapped, leverage reported
LEVERAGE_CAP_SENSITIVITY = (2.0, 3.0)

# --- Rebalancing and execution ------------------------------------------------
# Signals and weights are computed from the close of the last trading day of
# month t and held through month t+1, with daily mark-to-market drift.
# Primary: trade at that same close (MOP convention).
# Robustness: trade at the next day's close (one-day execution lag).
EXECUTION_LAG_DAYS = 0
EXECUTION_LAG_SENSITIVITY = (1,)

# --- Costs -------------------------------------------------------------------
COST_BPS = 10.0                           # per unit of notional traded, one way
COST_SENSITIVITY_BPS = (0, 2, 5, 10, 20, 50)
BORROW_FEE_SENSITIVITY_BPS = (0, 50, 100)  # per year on short notional
# Added 2026-10-01 after Stage 4, before any of its results were computed:
# leverage above 1x NAV must be financed, and a broker charges more than the
# T-bill rate. Spread per year over rf on borrowed cash, where borrowed cash =
# max(long notional - NAV, 0) (short sale proceeds conservatively not used
# to fund longs).
FINANCING_SPREAD_SENSITIVITY_BPS = (0, 50, 100)
# Every robustness variant also reports its volatility bias statistic
# (std of monthly return / ex-ante monthly vol), so the effect of each
# estimator choice on risk-forecast accuracy is visible, not only on Sharpe.

# --- Regime layer -------------------------------------------------------------
# Primary (R1, "trend state"): 2-state Gaussian Markov-switching model with
# switching mean and variance, fitted to the weekly cross-asset trend payoff
# (average over instruments of the unit-risk 12-month TSMOM return).
# Alternative (R2, "market turbulence"): same model on weekly SPY excess returns.
# Fitting: expanding window, refitted every month-end, FILTERED probabilities
# only (never smoothed). Weeks before MAIN_START are used as burn-in.
HMM_FREQ = "W-FRI"
HMM_N_STATES = 2
HMM_MIN_TRAIN_WEEKS = 156
HMM_SEARCH_REPS = 20        # random restarts for EM, keep max likelihood
HMM_SEED = 2026
# Exposure multiplier for month t+1 from probability p_t at the end of month t:
#   R1: m = 0.5 + P(trending state)    R2: m = 0.5 + P(turbulent state)
# Applied after portfolio vol targeting (not re-targeted, or it would cancel).
REGIME_FLOOR = 0.5

# --- Evaluation ---------------------------------------------------------------
# Crisis windows are defined by rule, not chosen by eye: every SPY peak-to-trough
# drawdown (closing prices) deeper than this threshold within the main sample.
CRISIS_DRAWDOWN_THRESHOLD = 0.15
BOOTSTRAP_REPS = 5000
BOOTSTRAP_MEAN_BLOCK_MONTHS = 6   # stationary bootstrap (Politis and Romano 1994)
BOOTSTRAP_SEED = 2026


# --- Portfolio application: trend as a diversifier (added 2026-10-02) ---------
# Added after every result above had been seen, and fixed here before any
# blend number was computed. It does not change the trend strategy: V2 is used
# exactly as reported, so it is not a further trend configuration and does not
# enter the deflated Sharpe ratio's trial count.
#
# Primary blend: 80% in the 60/40 benchmark + 20% in V2. Secondary: 80% SPY +
# 20% V2. Justification: 10% to 20% is a common allocation range for managed
# futures in diversified portfolios, and 20% is the round number at the top of
# that range. It was not chosen by comparing alternatives, and no other weight
# is computed.
#
# Construction (daily, matching the engine): sleeves are rebalanced to the
# fixed weights at the close of each month's last trading day (the engine's
# rebalance dates) and drift with their returns within the month,
#     R_blend,d = a_{d-1} R_bench,d + (1 - a_{d-1}) R_V2,d,
#     a_d = a_{d-1} (1 + rf_d + R_bench,d) / (1 + rf_d + R_blend,d).
# Monthly excess returns compound the daily ones exactly as for every other
# series (backtest.monthly_excess), over the same 222 months. V2 is net of its
# 10 bps costs; the 60/40 and SPY benchmarks are gross, as in the rest of the
# paper. Rebalancing turnover is reported as the NAV moved between sleeves per
# year, and as an upper bound on instrument notional traded (sleeve trade x
# that sleeve's gross leverage); no cost is deducted from the blend.
# Drawdowns are measured on the daily series, like every other drawdown.
# Inference: Sharpe difference of each blend against its own benchmark with the
# paired stationary bootstrap above (same block length, reps and seed).
#
# Interpretation rule: the headline is the effect on maximum drawdown and on
# losses in the six rule-based crisis windows, the purpose of a diversifier.
# The Sharpe change is reported with its bootstrap p-value; if p > 0.05 it is
# described as not significant, and the blend is not described as "improving
# the portfolio" unless the evidence supports it.
BLEND_WEIGHT_TREND = 0.20
BLENDS = {"60/40+V2": "60/40", "SPY+V2": "SPY"}   # blend -> its benchmark
BLEND_TREND_SERIES = "V2"


# --- Exposure-normalised regime overlay (added 2026-10-07) --------------------
# Added after every result above had been seen, and fixed here before any of
# its numbers was computed. Motivation: V3 multiplies V2 by (0.5 + p_t), whose
# mean over the sample is 1.19, so V3 changes both the LEVEL and the TIMING of
# exposure, and V3 against V2 does not isolate timing.
#
# (a) "V3_norm", walk-forward normalised overlay (tradable). Weights at
#     rebalance t are V2's times
#         m_t = (0.5 + p_t) / (0.5 + pbar_t),
#     where p_t is the primary R1 walk-forward filtered probability exactly as
#     used by V3 (output/stage5_walkforward_R1.csv, degenerate refits included)
#     and pbar_t is the mean of p_s over the rebalance dates s from the first
#     live refit (2008-03-31, the first rebalance date) up to and including t:
#     an expanding mean, no future data. Nothing is refitted. Dividing by the
#     full-sample mean multiplier instead would use future information, the
#     lookahead error the paper warns about, and is not done.
# (b) "V2_x_const", constant-leverage diagnostic (NOT tradable). V2 weights
#     times c, the realised mean of V3's multiplier, mean over the 222
#     rebalance dates of (0.5 + p_t) (1.19 rounded; the unrounded value is
#     used). It uses full-sample information and is labelled as such. V3
#     against (b) shows whether V3 beats holding V2 at V3's average exposure.
# Both run through the same daily engine, 10 bps costs, 222 months.
#
# Report: one row each in the regime table (Table 13): mean, vol, Sharpe, MDD,
# mean gross leverage, Sharpe difference from V2 with the paired stationary
# bootstrap p-value (BOOTSTRAP_* settings above), alpha on V2 with NW t. The
# text also reports the bootstrap Sharpe difference of V3 against (b).
# (a) is added to the predictive table (Table 14): V2's return in month t+1
# regressed on m_t (slope in % per month per unit of multiplier, NW t), and
# V2's Sharpe ratio in months with m_t > 1 and m_t <= 1.
#
# Interpretation rule: the conclusion that the overlay adds no timing value is
# confirmed if (a)'s Sharpe difference from V2 is not significant at 5%
# (p >= 0.05) OR is negative. If it is significantly positive (difference > 0
# and p < 0.05), the timing component is reported as having value once
# leverage is held constant, and Section 4.7 and the Conclusion are revised.
#
# Multiple testing: (a) is a new trend configuration, so the trial count rises
# from 25 to 26. The deflated Sharpe table keeps its N = 25 rows unchanged and
# adds an N = 26 row in which (a)'s Sharpe ratio joins the trial set (cross-
# trial dispersion recomputed over the 26). (b) is a diagnostic benchmark, like
# the lookahead illustrations, and is not counted.
NORM_OVERLAY_FIRST_LIVE = "2008-03-31"

def newey_west_lags(n_obs: int) -> int:
    """Newey-West (1994) automatic lag rule, floor(4 * (T/100)^(2/9))."""
    return int(4 * (n_obs / 100) ** (2 / 9))
