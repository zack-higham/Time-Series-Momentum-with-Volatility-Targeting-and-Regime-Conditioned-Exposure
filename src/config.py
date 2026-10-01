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


# =============================================================================
# PHASE 2: breadth and carry. Pre-registered 2026-10-01, after the phase 1
# results (Stages 3 to 8) were known and BEFORE any phase 2 data was saved or
# any phase 2 result was computed. Only inception dates, pre-sample trading
# volume and data availability were checked to set the rules below.
#
# Motivation (economic, not a response to a Sharpe ratio): phase 1 found that
# trend alone is a modest stand-alone strategy whose value lies in crisis
# convexity, and that timing it with a regime model does not work. The
# standard response is diversification rather than timing: more instruments
# (breadth) and a second premium with low correlation to trend (carry; Koijen,
# Moskowitz, Pedersen and Vrugt 2018). Every phase 1 result stays as reported.
# =============================================================================

# --- Expanded universe U2 ------------------------------------------------------
# Candidates: liquid US-listed ETFs giving distinct exposures within the four
# asset classes (US size/style and major single-country equity funds; US
# Treasury, TIPS, investment-grade and aggregate bond funds; single-commodity
# and commodity-sector funds; currency funds). A candidate enters U2 if:
#   1. its first trade is on or before 2007-03-30, so it has a 12-month
#      signal at 2008-03-31 and the main sample is unchanged;
#   2. its median daily dollar volume over 2007-03-01 to 2008-03-31 (before
#      the main sample) is at least $1m;
#   3. it is an unlevered long exposure (inverse and leveraged funds out);
#   4. it does not hold the same underlying as a fund already in U2;
#   5. it is not excluded for the phase 1 reason (SHY: ~1.5% volatility).
# Screening (first trade; median $m/day): HYG 2007-04-11, UNG 2007-04-18,
# EMB 2007-12-19, BWX 2007-10-11 fail rule 1; DBB 0.9 and DBE 0.8 fail rule 2;
# UDN fails rule 3 (inverse dollar); IAU fails rule 4 (gold, as GLD); SHY
# fails rule 5. FXS (Swedish krona) was a candidate in 2007 but has been
# delisted and has no data: a survivorship gap, disclosed.
# USO was hand-excluded from the phase 1 core; under the mechanical rule it
# is included.
PHASE2_CANDIDATES = {
    "Equity": ["IWM", "QQQ", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "FXI", "EWT", "EWY",
               "EWW", "EZU"],
    "Rates": ["TIP", "LQD", "AGG", "SHY", "HYG", "EMB", "BWX"],
    "Commodity": ["SLV", "USO", "DBA", "DBB", "DBE", "GSG", "UNG", "IAU"],
    "FX": ["FXE", "FXB", "FXC", "FXF", "UDN", "FXS"],
}
PHASE2_FIRST_TRADE_BY = "2007-03-30"
PHASE2_MIN_DOLLAR_VOLUME_M = 1.0
PHASE2_LIQUIDITY_WINDOW = ("2007-03-01", "2008-03-31")
PHASE2_ADDITIONS = {
    "IWM": "Equity", "QQQ": "Equity", "EWJ": "Equity", "EWG": "Equity", "EWU": "Equity",
    "EWC": "Equity", "EWA": "Equity", "EWZ": "Equity", "FXI": "Equity", "EWT": "Equity",
    "EWY": "Equity", "EWW": "Equity", "EZU": "Equity",
    "TIP": "Rates", "LQD": "Rates", "AGG": "Rates",
    "SLV": "Commodity", "USO": "Commodity", "DBA": "Commodity", "GSG": "Commodity",
    "FXE": "FX", "FXB": "FX", "FXC": "FX", "FXF": "FX",
}
UNIVERSE_U2 = {**INSTRUMENTS, **PHASE2_ADDITIONS}   # 34 ETFs
# Risk allocation, signals, volatility estimator, vol target, costs and sample
# are exactly as in phase 1 (equal risk per asset class, so adding funds to a
# class spreads its budget rather than enlarging it).

# --- Carry ---------------------------------------------------------------------
# Carry is the return an instrument earns if prices do not change (KMPV 2018).
#  Equity and bond ETFs: trailing 12-month distributions per share divided by
#    the price, minus the T-bill yield (^IRX), both at month-end t. Uses only
#    distributions with ex-dates on or before t.
#  Currency funds (long foreign currency): foreign minus US 3-month interbank
#    rate (OECD, via FRED). Monthly averages, so the value for month t-1 is the
#    one used at month-end t (known by then). Euro: the German series
#    (IR3TIB01DEM156N; the euro-area series ends 2026-01). Sterling: the 3-month
#    series ends 2026-01, so from 2026-02 the monthly mean of SONIA (IUDSOIA) is
#    used, disclosed as a splice.
#  UUP (long USD against the DXY basket): US rate minus the DXY-weighted
#    foreign rate.
#  Commodities: no carry (futures curves are not freely available), so the
#    carry portfolio holds equities, rates and FX only.
CARRY_CLASSES = ("Equity", "Rates", "FX")
FX_RATE_SERIES = {
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01DEM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "CAD": "IR3TIB01CAM156N", "CHF": "IR3TIB01CHM156N",
    "AUD": "IR3TIB01AUM156N", "SEK": "IR3TIB01SEM156N",
}
GBP_SPLICE_SERIES = "IUDSOIA"           # SONIA, daily; monthly mean from 2026-02
CURRENCY_OF_FUND = {"FXE": "EUR", "FXY": "JPY", "FXB": "GBP", "FXC": "CAD", "FXF": "CHF",
                    "FXA": "AUD"}
DXY_WEIGHTS = {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091, "SEK": 0.042,
               "CHF": 0.036}
# Signal (primary): cross-sectional within each carry asset class, as KMPV:
#   s_i = (rank_i - (n+1)/2) / ((n-1)/2)  in [-1, 1], highest carry long.
# Positions then use the phase 1 machinery: inverse volatility with equal risk
# per carry class (1/3 each, split equally within class), scaled to 10%
# ex-ante portfolio volatility.
# Robustness: time-series carry, s_i = sign(carry_i).
CARRY_SIGNAL = "cross_sectional_rank"

# --- Combination -----------------------------------------------------------------
# Primary phase 2 result: COMBO = 0.5 x TREND_U2 + 0.5 x CARRY_U2 (each already
# scaled to 10% ex-ante volatility), rescaled to 10% ex-ante volatility. Fixed
# 50/50 risk weights: not optimised.
COMBO_RISK_WEIGHTS = {"trend": 0.5, "carry": 0.5}
# Portfolio value: 60/40 blended with COMBO at these fixed weights.
BLEND_WEIGHTS = (0.1, 0.2, 0.3)

# --- Historical validation (1990 to 2008), the same rules on proxy data -------------
# The ETF sample starts in 2008. To test the phase 2 rules on data that played
# no part in any design decision, the same trend, carry and combination rules
# are applied to a proxy universe built from free long-history sources:
#   US equity: Kenneth French daily market excess return.
#   US Treasuries, 5y and 10y: synthetic constant-maturity bond excess returns
#     from FRED yields (DGS5, DGS10), duration and convexity approximation.
#   Currencies vs USD: GBP, CAD, AUD, JPY, CHF from FRED daily spot rates plus
#     the short-rate differential (forward excess return under covered
#     interest parity). Rates: OECD 3-month interbank; Japan and Switzerland
#     use OECD call-money rates where the 3-month series starts too late.
# Carry: Treasuries = yield minus 3-month bill (DTB3); currencies = rate
# differential; equity has no carry here (no free daily dividend yield).
# Classes: Equity, Rates, FX (no commodities). Holding months 1990-01 to
# 2008-03; earlier data is warm-up only. Costs 10 bps as in the main sample.
HIST_START = "1990-01"
HIST_END = "2008-03"
HIST_FX = {"GBP": ("DEXUSUK", True), "CAD": ("DEXCAUS", False), "AUD": ("DEXUSAL", True),
           "JPY": ("DEXJPUS", False), "CHF": ("DEXSZUS", False)}   # (series, quoted as USD per unit)
HIST_FX_RATES = {"USD": "IR3TIB01USM156N", "GBP": "IR3TIB01GBM156N", "CAD": "IR3TIB01CAM156N",
                 "AUD": "IR3TIB01AUM156N", "JPY": "IRSTCI01JPM156N", "CHF": "IRSTCI01CHM156N"}
HIST_BOND_YIELDS = {"UST5": ("DGS5", 5), "UST10": ("DGS10", 10)}
HIST_BILL = "DTB3"
