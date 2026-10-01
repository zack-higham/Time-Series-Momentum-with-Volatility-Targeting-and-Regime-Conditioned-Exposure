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


def newey_west_lags(n_obs: int) -> int:
    """Newey-West (1994) automatic lag rule, floor(4 * (T/100)^(2/9))."""
    return int(4 * (n_obs / 100) ** (2 / 9))
