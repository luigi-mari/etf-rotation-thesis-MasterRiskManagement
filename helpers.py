"""
helpers.py — Reusable quant research utilities for THESIS.ipynb and FeatureRedundancy_v11.ipynb
================================================================================

Sections
--------
 1. Kalman filter (Latent Momentum)                    -> KalmanFilter1D
 2. Volume-profile Point of Control                   -> _compute_poc_array
 3. SMA derivative                                  -> compute_sma_derivative
 4. Raw feature engineering                         -> FEATURES_ALL, FEATURE_WINDOWS, 15x_feat_*,
                                                      build_raw_features, normalise_ts, zscore_ts
 5. Selection hysteresis                            -> apply_hysteresis_margin
 6. Portfolio metrics                               -> perf_stats, mdd_s, equity, sharpe_ann, perf_sub,
                                                      net_sharpe, perf_row, calc_metrics, print_perf_report, ir_vs
 7. IC / alpha / turnover diagnostics              -> weekly_ic, icir, spy_aligned_returns,
                                                      alpha_vs_benchmark, turnover_symmetric, turnover_one_way
 8. Statistical significance                        -> jobson_korkie_memmel, deflated_sr, _mp_denoise, cscv_pbo
 9. Selection funnel & construction methods         -> SelectionFunnel, soft_exposure, ConstructionMethods,
                                                      run_funnel_with_hysteresis
10. ML feature utilities & data loading           -> extract_features_at, apply_cs_rank, load_weekly_universe
11. ML benchmark helpers (XGBoost / LightGBM)     -> BufferPortfolio, build_xgb_panel/train_xgb/score_assets,
                                                      build_lgb_panel/panel_upto/purged_cv_score/random_search_lgb,
                                                      plot_learning_curve, tree_split_paths, RunStore

Both notebooks import from this module instead of re-defining the logic locally.
THESIS-specific tuning constants (XGB_PARAMS, SEARCH_SPACE, ...) stay in THESIS.ipynb;
the functions here take them as explicit parameters.
"""

import numpy as np
import pandas as pd
from numba import njit
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from typing import List, Optional, Tuple, Dict
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════
#  1. KALMAN FILTER  —  Latent Momentum Estimation
# ═══════════════════════════════════════════════════════════════

class KalmanFilter1D:
    """
    Constant-velocity Kalman Filter for price series.

    State vector: [position, velocity]
    The velocity component is the "Kalman momentum" — a denoised
    estimate of the latent trend in the price series.

    References
    ----------
    - Harvey (1989), "Forecasting, Structural Time Series Models
      and the Kalman Filter"
    - Used here for latent momentum estimation (kalman_dev feature)
    """

    def __init__(self, process_var: float = 1e-4, measurement_var: float = 1e-2):
        """
        Parameters
        ----------
        process_var : float
            Process noise variance (Q). Lower = smoother estimates.
            Controls how much the true state is assumed to change per step.
        measurement_var : float
            Measurement noise variance (R). Higher = trust observations less.
        """
        self.process_var = process_var
        self.measurement_var = measurement_var

    def filter(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Kalman filter on a 1D price series.

        Parameters
        ----------
        prices : np.ndarray
            Raw price series (e.g., daily close prices).

        Returns
        -------
        filtered_prices : np.ndarray
            Kalman-smoothed price estimates.
        kalman_momentum : np.ndarray
            Velocity component = denoised momentum signal.
        """
        n = len(prices)
        # State: [position, velocity]
        x = np.zeros((n, 2))
        P = np.zeros((n, 2, 2))

        # Transition matrix: constant velocity model
        F = np.array([[1.0, 1.0],
                       [0.0, 1.0]])
        # Observation matrix: we observe position only
        H = np.array([[1.0, 0.0]])
        # Process noise
        Q = np.array([[self.process_var, 0.0],
                       [0.0, self.process_var * 0.1]])
        # Measurement noise
        R = np.array([[self.measurement_var]])

        # Initialize with first observation
        x[0] = [prices[0], 0.0]
        P[0] = np.eye(2) * 1.0

        for t in range(1, n):
            # Predict
            x_pred = F @ x[t - 1]
            P_pred = F @ P[t - 1] @ F.T + Q

            # Update
            y = prices[t] - H @ x_pred          # innovation
            S = H @ P_pred @ H.T + R            # innovation covariance
            K = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain

            x[t] = x_pred + K.flatten() * y.item()
            P[t] = (np.eye(2) - K @ H) @ P_pred

        filtered_prices = x[:, 0]
        kalman_momentum = x[:, 1]

        return filtered_prices, kalman_momentum

    def compute_features(self, close: pd.Series,
                         vol_window: int = 21) -> pd.DataFrame:
        """
        Compute Kalman-based features for a single asset.

        Returns DataFrame with columns:
        - kalman_price: filtered price
        - kalman_mom: raw velocity (momentum)
        - kalman_dev: (Price - Kalman) / rolling_vol — normalized deviation
          Positive = price overextended above Kalman (overbought)
          Negative = price below Kalman (oversold / mean-reversion opportunity)
        """
        prices = close.values.astype(np.float64)
        mask = ~np.isnan(prices)
        if mask.sum() < 10:
            return pd.DataFrame(index=close.index)

        # Forward-fill NaNs for Kalman (it needs continuous series)
        prices_clean = pd.Series(prices, index=close.index).ffill().bfill().values

        filt_p, filt_v = self.filter(prices_clean)

        df = pd.DataFrame(index=close.index)
        df["kalman_price"] = filt_p
        df["kalman_mom"] = filt_v

        # σ_t = (P_t - K_t) / σ_rolling
        # Normalized deviation of price from Kalman estimate
        ret = close.pct_change()
        roll_vol = ret.rolling(vol_window, min_periods=10).std() * close
        roll_vol = roll_vol.replace(0, np.nan)
        df["kalman_dev"] = (close.values - filt_p) / (roll_vol + 1e-10)

        return df
# ═══════════════════════════════════════════════════════════════
#  2. VOLUME PROFILE  —  Point of Control (POC) via Numba
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_poc_array(high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, volume: np.ndarray,
                       window: int, n_bins: int = 50) -> np.ndarray:
    """
    Compute rolling Point of Control (POC) using Fixed Range Volume Profile.

    POC = price level with the highest accumulated volume over the lookback window.

    Parameters
    ----------
    high, low, close, volume : np.ndarray
        OHLCV data arrays (1D, same length).
    window : int
        Lookback window for volume profile computation.
    n_bins : int
        Number of price bins for the volume histogram.

    Returns
    -------
    poc : np.ndarray
        POC price level for each bar.
    """
    n = len(close)
    poc = np.full(n, np.nan)

    for i in range(window, n):
        start = i - window
        # Price range over the window
        h_max = high[start:i].max()
        l_min = low[start:i].min()
        if h_max <= l_min:
            poc[i] = close[i]
            continue

        bin_size = (h_max - l_min) / n_bins
        vol_bins = np.zeros(n_bins)

        for j in range(start, i):
            if bin_size == 0:
                continue
            # Distribute volume across bins that the bar's range covers
            lo_bin = int((low[j] - l_min) / bin_size)
            hi_bin = int((high[j] - l_min) / bin_size)
            lo_bin = max(0, min(lo_bin, n_bins - 1))
            hi_bin = max(0, min(hi_bin, n_bins - 1))
            n_covered = hi_bin - lo_bin + 1
            vol_per_bin = volume[j] / n_covered
            for b in range(lo_bin, hi_bin + 1):
                vol_bins[b] += vol_per_bin

        # POC = midpoint of the bin with max volume
        max_bin = 0
        max_vol = vol_bins[0]
        for b in range(1, n_bins):
            if vol_bins[b] > max_vol:
                max_vol = vol_bins[b]
                max_bin = b

        poc[i] = l_min + (max_bin + 0.5) * bin_size

    return poc


# ═══════════════════════════════════════════════════════════════
#  3. SMA DERIVATIVE  —  Trend Acceleration
# ═══════════════════════════════════════════════════════════════

def compute_sma_derivative(close: pd.Series, sma_window: int,
                           diff_window: int = 5) -> pd.Series:
    """
    Compute the slope (derivative) of a Simple Moving Average.

    d(SMA)/dt ≈ (SMA_t - SMA_{t-k}) / k, normalized by price.
    Positive = trend accelerating upward, Negative = decelerating.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    sma_window : int
        SMA window (e.g., 50 or 200).
    diff_window : int
        Finite difference window for the derivative.

    Returns
    -------
    pd.Series : Normalized SMA slope.
    """
    sma = close.rolling(sma_window, min_periods=sma_window).mean()
    slope = sma.diff(diff_window) / diff_window
    return slope / (close + 1e-10)


# ═══════════════════════════════════════════════════════════════
#  4. FEATURES — Raw Feature Computation (shared by FR + THESIS)
# ═══════════════════════════════════════════════════════════════
#
#  This section is the single source of truth for all 15 raw features
#  used by both notebooks. Each feature has a named `_feat_*` helper so it
#  can be unit-tested or imported independently. The `build_raw_features`
#  orchestrator assembles them into one DataFrame per ticker.
#
#  Window defaults below match FR's CONFIG exactly (VIX_ZS_WINDOW=252).
#  THESIS historically used VIX_ZS_WINDOW=260 — callers override via the
#  `vix_zs_window` keyword on build_raw_features (or pass a custom window to
#  _feat_vix_zscore directly). No other notebook has drifted.
#
#  Sign conventions are preserved verbatim from FR c8 / THESIS T-2:
#    • kalman_dev        : -1 × kf.kalman_dev           (positive = underperforming vs trend)
#    • poc_cross         : (POC_50 − POC_78) / close     (positive = price above mid-term POC)
#    • sma_deriv_neg     : -1 × SMA(12).diff(3)          (mean-reversion: high = decelerating trend)
#    • vol_4w            : 4w realised vol                (annualised, sqrt(52))
#    • mom_52w           : close[t-1]/close[t-53] − 1     (12-month momentum)
#    • rev_2w            : -1 × (close[t-1]/close[t-3] − 1) (Lehmann/Jegadeesh short-term reversal)
#    • calmar_neg_26w    : -1 × roll_ret/|max_dd|         (negated Calmar, clipped ±20)
#    • pearson_r_neg_26w : -1 × Pearson(close vs time)    (negated trend-linearity)
#    • sma50_zscore      : -1 × z-score of close vs SMA50 (positive = price BELOW SMA50)
#    • f_poc_xs          : -1 × z-score of dist from POC_16w (positive = below 16w POC)
#    • mom_26w           : close[t-1]/close[t-27] − 1     (6-month momentum, Jegadeesh-Titman)
#    • vol_26w           : 26w realised vol               (Moreira-Muir regime signal)
#    • volume_zscore     : z-score of weekly volume       (Lo-Wang abnormal volume), clipped ±3
#    • vix_zscore        : z-score of VIX vs own history   (Whaley fear regime), clipped ±3
#    • price_range_z     : z-score of (H−L)/close          (Garman-Klass intraweek range), clip ±3
#
#  NOTE: `vix_zscore` is a MACRO feature — identical value across all tickers on any given date.
#        CS IC for it is ~0 by construction; only TS IC carries signal. Kept in the panel so that
#        downstream ML models (XGB/LGB) can use it as a global regime input without a separate path.

FEATURES_ALL: List[str] = [
    "kalman_dev", "poc_cross", "sma_deriv_neg",
    "vol_4w",     "mom_52w",   "rev_2w",
    "sma50_zscore", "pearson_r_neg_26w", "calmar_neg_26w", "f_poc_xs",
    "mom_26w", "vol_26w", "volume_zscore", "vix_zscore", "price_range_z",
]

# FR's LIVE_FEATURES subset — the 6 features that survived prior IC/alpha-orthogonality tests.
FEATURES_LIVE: List[str] = [
    "kalman_dev", "poc_cross", "sma_deriv_neg",
    "vol_4w",     "mom_52w",   "rev_2w",
]

# Default rolling window constants (FR values; THESIS overrides vix_zs_window=260 at call site).
FEATURE_WINDOWS: Dict[str, int] = dict(
    sma_deriv_window  = 12,
    sma_deriv_diff    = 3,
    calmar_window     = 26,
    pearson_window    = 26,
    sma50_window      = 50,
    poc_dist_window   = 16,   # f_poc_xs POC distance window
    poc_zscore_wz     = 52,   # f_poc_xs rolling z-score window (wz)
    mom_26w_shift     = 27,   # close[t-1]/close[t-27] − 1
    vol_26w_window    = 26,
    volume_zs_window  = 52,
    vix_zs_window     = 252,  # FR default; THESIS passes 260 at call site
    price_range_wnd   = 52,
)

# ── Individual feature functions (one per name in FEATURES_ALL) ─────────────
#
#  Each helper takes the OHLCV DataFrame for ONE ticker and returns a pd.Series
#  aligned to df.index. Features that need an external series (vix_zscore → vix_weekly)
#  take it as an explicit parameter rather than reading from notebook scope.

def _feat_kalman_dev(df: pd.DataFrame,
                     process_var: float = 1e-4,
                     measurement_var: float = 1e-2) -> pd.Series:
    """
    kalman_dev — NEGATED Kalman filter deviation from smoothed close.

    Positive value = price below Kalman-smoothed trend (mean-reversion buy signal).
    Uses KalmanFilter1D(process_var=1e-4, measurement_var=1e-2) — same defaults as
    the notebooks have historically used; do not change without re-validating.
    """
    kf = KalmanFilter1D(process_var=process_var, measurement_var=measurement_var)
    kf_f = kf.compute_features(df["Close"])
    dev  = kf_f.get("kalman_dev", pd.Series(np.nan, index=df.index))
    return -1.0 * dev


def _feat_poc_cross(df: pd.DataFrame) -> pd.Series:
    """
    poc_cross — (POC_50w − POC_78w) / close.

    Positive = price above mid-term POC band (trend continuation).
    Negative = price below POC band (mean-reversion buy signal).
    Windows 50 and 78 are the values used in both notebooks; kept as literals here to
    match FR c8 / THESIS T-2 exactly.
    """
    def poc_arr(w: int) -> pd.Series:
        return pd.Series(
            _compute_poc_array(
                df["High"].values.astype(np.float64),
                df["Low"].values.astype(np.float64),
                df["Close"].values.astype(np.float64),
                df["Volume"].values.astype(np.float64), window=w),
            index=df.index)
    poc_50  = poc_arr(50)
    poc_78  = poc_arr(78)   # named 78 in code; semantically the 'poc_200' slot
    close   = df["Close"]
    return (poc_50 - poc_78) / (close + 1e-10)


def _feat_sma_deriv_neg(df: pd.DataFrame,
                        sma_window: int = 12, diff_window: int = 3) -> pd.Series:
    """
    sma_deriv_neg — NEGATED 12w SMA slope (diff=3).

    Raw SMA derivative showed strong but NEGATIVE TS IC (t=-4.43); negation reframes it as a
    mean-reversion signal consistent in sign with kalman_dev. Verified GENUINE vs kalman_dev
    via alpha-orthogonality test (t_alpha=+10.33).
    """
    return -1.0 * compute_sma_derivative(df["Close"],
                                         sma_window=sma_window,
                                         diff_window=diff_window)


def _feat_vol_4w(df: pd.DataFrame) -> pd.Series:
    """vol_4w — 4-week annualised realised vol (std of log-returns × sqrt(52))."""
    close   = df["Close"]
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(4, min_periods=2).std() * np.sqrt(52)


def _feat_mom_52w(df: pd.DataFrame) -> pd.Series:
    """mom_52w — 12-month momentum: close[t-1]/close[t-53] − 1 (skip last week to avoid short-term reversal)."""
    close = df["Close"]
    return close.shift(1) / close.shift(53) - 1.0


def _feat_rev_2w(df: pd.DataFrame) -> pd.Series:
    """rev_2w — NEGATED 2-week past return (Lehmann 1990, Jegadeesh 1990 short-term reversal)."""
    close = df["Close"]
    return -1.0 * (close.shift(1) / close.shift(3) - 1.0)


def _feat_calmar_neg_26w(df: pd.DataFrame, window: int = 26) -> pd.Series:
    """
    calmar_neg_26w — NEGATED rolling Calmar (return/|maxDD| over `window` weeks), clipped ±20.

    Raw positive Calmar (clean uptrend) predicted TS reversal (t=-3.38 on raw → negated).
    """
    close   = df["Close"]
    roll_ret = close / close.shift(window) - 1
    roll_max = close.rolling(window, min_periods=window // 2).max()
    max_dd   = (close / roll_max - 1).rolling(
        window, min_periods=window // 2).min()
    return (-1.0 * (roll_ret / (max_dd.abs() + 1e-4))).clip(-20, 20)


def _feat_pearson_r_neg_26w(df: pd.DataFrame, window: int = 26) -> pd.Series:
    """
    pearson_r_neg_26w — NEGATED Pearson R between close and a linear time index over `window` weeks.

    Raw positive R (clean uptrend linearity) predicted TS reversal (t=-3.12 on raw → negated).
    Values near 0 = sideways / noisy price action; large |R| = clean trend either way.
    """
    def _pearson_r(y: np.ndarray) -> float:
        n = len(y)
        if n < 3:
            return np.nan
        x  = np.arange(n, dtype=float)
        mx, my = x.mean(), y.mean()
        num = ((x - mx) * (y - my)).sum()
        den = np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
        return float(num / den) if den > 1e-10 else 0.0
    close = df["Close"]
    raw   = close.rolling(window, min_periods=window // 2).apply(_pearson_r, raw=True)
    return -1.0 * raw


def _feat_sma50_zscore(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    sma50_zscore — NEGATED z-score of close vs SMA`window`.

    Positive value = price BELOW its SMA (mean-reversion buy signal), consistent with kalman_dev sign.
    Complementary to kalman_dev: deterministic reference instead of adaptive Kalman estimate.
    """
    close   = df["Close"]
    sma     = close.rolling(window, min_periods=window // 2).mean()
    sma_std = close.rolling(window, min_periods=window // 2).std()
    return -1.0 * (close - sma) / (sma_std + 1e-8)


def _feat_f_poc_xs(df: pd.DataFrame,
                   poc_w: int = 16,
                   wz: int = 52) -> pd.Series:
    """
    f_poc_xs — NEGATED z-score of distance from POC_`poc_w` (default 16w), rolling `wz` window.

    dist   = (close − POC_16w) / POC_16w            (% above/below 16w point-of-control)
    dist_z = z-score of dist over trailing 52 weeks
    f_poc_xs = -1 × dist_z                          (positive = price BELOW 16w POC → buy signal)

    Validated in TentNWB: raw CS IC=-0.07, t=-5.0; negation yields positive IC.
    """
    def poc_arr(w: int) -> pd.Series:
        return pd.Series(
            _compute_poc_array(
                df["High"].values.astype(np.float64),
                df["Low"].values.astype(np.float64),
                df["Close"].values.astype(np.float64),
                df["Volume"].values.astype(np.float64), window=w),
            index=df.index)
    close   = df["Close"]
    poc_p   = poc_arr(poc_w)
    dist_s  = (close - poc_p) / (poc_p + 1e-10)
    dist_z  = ((dist_s - dist_s.rolling(wz, min_periods=wz // 2).mean())
               / dist_s.rolling(wz, min_periods=wz // 2).std().replace(0, np.nan))
    return -1.0 * dist_z


def _feat_mom_26w(df: pd.DataFrame) -> pd.Series:
    """mom_26w — 6-month momentum (Jegadeesh & Titman 1993): close[t-1]/close[t-27] − 1."""
    close = df["Close"]
    return close.shift(1) / close.shift(27) - 1.0


def _feat_vol_26w(df: pd.DataFrame, window: int = 26) -> pd.Series:
    """vol_26w — `window`-week annualised realised vol (Moreira & Muir 2017 regime signal)."""
    close   = df["Close"]
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window // 2).std() * np.sqrt(52)


def _feat_volume_zscore(df: pd.DataFrame, window: int = 52) -> pd.Series:
    """
    volume_zscore — z-score of weekly volume vs trailing `window` history (Lo & Wang 2000).

    Positive = abnormally high volume this week. Returns all-NaN if Volume column missing or sparse.
    Clipped ±3 to damp outliers.
    """
    if "Volume" in df.columns and df["Volume"].notna().sum() > 50:
        vol_series = df["Volume"].replace(0, np.nan).ffill()
        vol_mu     = vol_series.rolling(window, min_periods=window // 2).mean()
        vol_std    = vol_series.rolling(window, min_periods=window // 2).std()
        return ((vol_series - vol_mu) / (vol_std + 1e-8)).clip(-3, 3)
    return pd.Series(np.nan, index=df.index)


def _feat_vix_zscore(vix_weekly: Optional[pd.Series],
                     df_index: pd.Index,
                     window: int = 252) -> pd.Series:
    """
    vix_zscore — z-score of VIX vs its own trailing `window` history (Whaley 2009 macro fear regime).

    MACRO feature: identical value across all tickers on any given date.
      • CS IC ~ 0 by construction (no cross-sectional ranking power)
      • TS IC carries the signal (some assets respond more to VIX spikes than others)

    Parameters
    ----------
    vix_weekly : pd.Series, optional
        Weekly ^VIX close series. If None or empty, returns an all-NaN series aligned to df_index.
    window : int
        Rolling z-score window in weeks. FR default = 252 (5y); THESIS historically used 260 (~5y).
        Override at call site — do NOT change the default without re-validating both notebooks.
    """
    if vix_weekly is None or len(vix_weekly) == 0:
        return pd.Series(np.nan, index=df_index)
    vix_aligned = vix_weekly.reindex(df_index).ffill()
    vix_mu      = vix_aligned.rolling(window, min_periods=window // 2).mean()
    vix_std     = vix_aligned.rolling(window, min_periods=window // 2).std()
    return ((vix_aligned - vix_mu) / (vix_std + 1e-8)).clip(-3, 3)


def _feat_price_range_z(df: pd.DataFrame, window: int = 52) -> pd.Series:
    """
    price_range_z — z-score of intraweek range (High − Low)/Close vs trailing `window` history.

    Garman & Klass (1980) intraperiod dispersion measure. High value = wide intraweek range =
    elevated uncertainty or institutional disagreement. Different from vol_4w which uses close-to-close.
    """
    if "High" in df.columns and "Low" in df.columns:
        close  = df["Close"]
        pr     = (df["High"] - df["Low"]) / (close + 1e-10)
        pr_mu  = pr.rolling(window, min_periods=window // 2).mean()
        pr_std = pr.rolling(window, min_periods=window // 2).std()
        return ((pr - pr_mu) / (pr_std + 1e-8)).clip(-3, 3)
    return pd.Series(np.nan, index=df.index)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def build_raw_features(df_ohlcv: pd.DataFrame,
                       vix_weekly: Optional[pd.Series] = None,
                       feature_list: Optional[List[str]] = None,
                       *,
                       vix_zs_window: int = 252) -> pd.DataFrame:
    """
    Compute the full raw-feature panel for ONE ticker.

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        OHLCV weekly bars for a single ticker. Must have columns High/Low/Close/Volume.
        Returns an empty DataFrame if len(df) < 60 (not enough history).
    vix_weekly : pd.Series, optional
        Weekly ^VIX close series used by the `vix_zscore` feature. If None and 'vix_zscore'
        is in feature_list, that column will be all-NaN.
    feature_list : list[str], optional
        Which features to compute. Default = FEATURES_ALL (15). Pass a subset for lighter panels.
        Unknown names raise ValueError.
    vix_zs_window : int
        Rolling window for the vix_zscore z-score. FR default 252; THESIS passes 260.
        Only affects `vix_zscore` — all other features are independent of this parameter.

    Returns
    -------
    pd.DataFrame
        Index = df_ohlcv.index (weekly dates). Columns = the requested feature_list in order.
        Empty DataFrame if len(df) < 60.
    """
    if len(df_ohlcv) < 60:
        return pd.DataFrame()
    if feature_list is None:
        feature_list = FEATURES_ALL
    unknown = [f for f in feature_list if f not in FEATURES_ALL]
    if unknown:
        raise ValueError(f"build_raw_features: unknown feature(s): {unknown}. Valid: {FEATURES_ALL}")

    # Lazily compute only the requested features (cheap wins when feature_list is small).
    out = {}
    for name in feature_list:
        if name == "kalman_dev":
            out[name] = _feat_kalman_dev(df_ohlcv)
        elif name == "poc_cross":
            out[name] = _feat_poc_cross(df_ohlcv)
        elif name == "sma_deriv_neg":
            out[name] = _feat_sma_deriv_neg(df_ohlcv,
                                            sma_window=FEATURE_WINDOWS["sma_deriv_window"],
                                            diff_window=FEATURE_WINDOWS["sma_deriv_diff"])
        elif name == "vol_4w":
            out[name] = _feat_vol_4w(df_ohlcv)
        elif name == "mom_52w":
            out[name] = _feat_mom_52w(df_ohlcv)
        elif name == "rev_2w":
            out[name] = _feat_rev_2w(df_ohlcv)
        elif name == "calmar_neg_26w":
            out[name] = _feat_calmar_neg_26w(df_ohlcv, window=FEATURE_WINDOWS["calmar_window"])
        elif name == "pearson_r_neg_26w":
            out[name] = _feat_pearson_r_neg_26w(df_ohlcv, window=FEATURE_WINDOWS["pearson_window"])
        elif name == "sma50_zscore":
            out[name] = _feat_sma50_zscore(df_ohlcv, window=FEATURE_WINDOWS["sma50_window"])
        elif name == "f_poc_xs":
            out[name] = _feat_f_poc_xs(df_ohlcv,
                                       poc_w=FEATURE_WINDOWS["poc_dist_window"],
                                       wz=FEATURE_WINDOWS["poc_zscore_wz"])
        elif name == "mom_26w":
            out[name] = _feat_mom_26w(df_ohlcv)
        elif name == "vol_26w":
            out[name] = _feat_vol_26w(df_ohlcv, window=FEATURE_WINDOWS["vol_26w_window"])
        elif name == "volume_zscore":
            out[name] = _feat_volume_zscore(df_ohlcv, window=FEATURE_WINDOWS["volume_zs_window"])
        elif name == "vix_zscore":
            out[name] = _feat_vix_zscore(vix_weekly, df_ohlcv.index,
                                         window=vix_zs_window)
        elif name == "price_range_z":
            out[name] = _feat_price_range_z(df_ohlcv, window=FEATURE_WINDOWS["price_range_wnd"])
    return pd.DataFrame(out, index=df_ohlcv.index)


def normalise_ts(df_raw: pd.DataFrame, lookback: int = 52) -> pd.DataFrame:
    """
    Rolling z-score each column of df_raw over trailing `lookback` weeks.
    min_periods = lookback // 2; result clipped to ±3. Identical math to both notebooks'
    existing local normalise_ts definitions (FR c8, THESIS T-2).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw feature panel (one row per week). NaNs propagate naturally through the rolling ops.
    lookback : int
        Rolling window in weeks. Default 52 = NORM_LOOKBACK in both notebooks' CONFIG.

    Returns
    -------
    pd.DataFrame with same shape/index; each column z-scored and clipped ±3.
    """
    df_norm = pd.DataFrame(index=df_raw.index)
    for col in df_raw.columns:
        s   = df_raw[col]
        mu  = s.rolling(lookback, min_periods=lookback // 2).mean()
        sig = s.rolling(lookback, min_periods=lookback // 2).std()
        df_norm[col] = ((s - mu) / (sig + 1e-8)).clip(-3, 3)
    return df_norm


def zscore_ts(s: pd.Series, lookback: int = 52) -> pd.Series:
    """
    Single-series version of normalise_ts. THESIS T-2 uses this to build kalman_z,
    sma_deriv_neg_z and sma50_z dicts for Layer-1 conviction scoring.

    Parameters
    ----------
    s : pd.Series
        Raw feature series (one row per week).
    lookback : int
        Rolling window in weeks. Default 52 = NORM_LOOKBACK.

    Returns
    -------
    pd.Series — z-scored and clipped ±3, same index as input.
    """
    mu  = s.rolling(lookback, min_periods=lookback // 2).mean()
    sig = s.rolling(lookback, min_periods=lookback // 2).std()
    return ((s - mu) / (sig + 1e-8)).clip(-3, 3)


# ══════════════════════════════════════════════════════════════════════════════
# SELECTION HYSTERESIS — margin-based incumbent protection
# Used by THESIS funnel loops (T-5 main, T-6 scheme, run_funnel_with_hysteresis_exact)
# ══════════════════════════════════════════════════════════════════════════════

def apply_hysteresis_margin(scores_raw: dict, prev_tickers: list,
                            K1: int, margin: float) -> dict:
    """
    Apply margin-based hysteresis to a raw score dict.

    Parameters
    ----------
    scores_raw : dict[str, float]
        Raw L1 conviction scores for all candidate tickers at this rebalance.
    prev_tickers : list[str]
        Tickers held in the previous period (empty/None → no hysteresis).
    K1 : int
        Target portfolio size.
    margin : float
        Minimum score improvement required to displace an incumbent.
        Use 0.0 for "always top-K" (no protection).

    Returns
    -------
    dict[str, float]
        The selected ticker→score mapping after hysteresis is applied.
    """
    if not prev_tickers:
        # No incumbents → simple top-K1 selection
        top = sorted(scores_raw, key=scores_raw.get, reverse=True)[:K1]
        return {tkr: scores_raw[tkr] for tkr in top}

    incumbents  = {tkr: scores_raw[tkr] for tkr in prev_tickers if tkr in scores_raw}
    challengers = {tkr: sc for tkr, sc in scores_raw.items() if tkr not in incumbents}
    current     = dict(incumbents)

    if len(current) < K1 and challengers:
        # Fill empty slots from best challengers (no margin needed for new slots)
        sorted_challengers = sorted(challengers, key=challengers.get, reverse=True)
        for tkr in sorted_challengers:
            if len(current) >= K1:
                break
            current[tkr] = challengers[tkr]
    elif challengers and margin > 0:
        # Replace weakest incumbent only if best challenger exceeds it by margin
        weakest_tk = min(current, key=current.get)
        best_chal  = max(challengers, key=challengers.get)
        if challengers[best_chal] > current[weakest_tk] + margin:
            current.pop(weakest_tk)
            current[best_chal] = challengers[best_chal]

    return current
# ════════════════════════════════════════════════════════════════════════════
#  5. PORTFOLIO METRICS — shared quant helpers (THESIS centralisation)
#  NOTE: THESIS contained several *different* Sortino formulas. perf_stats below
#        reproduces each one EXACTLY via the `sortino_variant` flag so that no
#        result changes when call-sites are migrated. Do NOT merge variants.
# ════════════════════════════════════════════════════════════════════════════
import math as _math
from scipy.stats import spearmanr as _spearmanr, norm as _norm
from itertools import combinations as _combinations


def mdd_s(s: pd.Series) -> float:
    """Maximum drawdown of a weekly return series (min of c/c.cummax() - 1)."""
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    c = (1 + s).cumprod()
    return float((c / c.cummax() - 1.0).min())


def perf_stats(returns: pd.Series, risk_free_rate: float = 0.0,
               sortino_variant: str = "downside") -> dict:
    """
    Canonical weekly performance metrics.

    Parameters
    ----------
    returns : pd.Series of weekly simple returns (NaNs dropped internally).
    risk_free_rate : annualised risk-free rate (default 0.0), used by the
        `downside` Sortino variant and the Sharpe numerator.
    sortino_variant :
        "negonly"   — CAGR / std(returns[returns<0]) * sqrt(52)      [THESIS perf()]
        "downside"  — (ann_mean - rf) / downside_std, where
                      downside = min(0, r - rf/52), ddof=1            [calc_metrics / _stats]

    Returns a dict of NUMERIC values: CAGR, AnnVol, Sharpe, Sortino,
        MaxDD, HitRate.
    """
    s = returns.dropna()
    n_weeks = len(s)
    if n_weeks < 1:
        return {k: np.nan for k in ("CAGR", "AnnVol", "Sharpe", "Sortino", "MaxDD", "HitRate")}

    n_years   = n_weeks / 52.0
    cagr      = (1 + s).prod() ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_mean  = float(s.mean()) * 52.0
    vol       = float(s.std(ddof=1)) * np.sqrt(52)
    mdd       = mdd_s(s)
    hit       = float((s > 0).mean())
    sharpe    = (ann_mean - risk_free_rate) / (vol + 1e-10)

    if sortino_variant == "negonly":
        neg = s[s < 0]
        so  = cagr / (float(neg.std(ddof=1)) * np.sqrt(52) + 1e-10) if len(neg) > 1 else np.nan
    elif sortino_variant == "downside":
        target       = risk_free_rate / 52.0
        downside     = np.minimum(0, s - target)
        downside_std = float(np.sqrt(np.mean(downside ** 2))) * np.sqrt(52)
        so           = (ann_mean - risk_free_rate) / (downside_std + 1e-10)
    else:
        raise ValueError(f"Unknown sortino_variant: {sortino_variant!r}")

    return {"CAGR": float(cagr), "AnnVol": float(vol), "Sharpe": float(sharpe),
            "Sortino": float(so), "MaxDD": mdd, "HitRate": hit}




# ── Formatted per-strategy metric rows (moved verbatim from THESIS local defs) ──

def equity(s: pd.Series, v: float = 10_000) -> pd.Series:
    """Cumulative equity curve from a weekly return series (THESIS cell-18 `equity`)."""
    return v * (1 + s.fillna(0)).cumprod()


def sharpe_ann(s: pd.Series) -> float:
    """Annualised Sharpe of the gross weekly returns; NaN if fewer than 20 weeks."""
    s = s.dropna()
    if len(s) < 20:
        return np.nan
    return perf_stats(s)["Sharpe"]


def perf_sub(s: pd.Series, mask: "pd.Series[bool]") -> dict:
    """perf_stats restricted to the masked subset (regime / regime-shift analyses)."""
    sub = s[mask].dropna()
    if len(sub) < 5:
        return {}
    return perf_stats(sub)


def net_sharpe(s: pd.Series, annual_to: float, rt: "pd.Series") -> float:
    """Sharpe after subtracting the per-week cost `annual_to * weekly turnover / 52`."""
    s = s.fillna(0) - annual_to * rt / 52.0
    return sharpe_ann(s)


def perf_row(returns: pd.Series, benchmark: pd.Series) -> dict:
    """
    Formatted metrics row for one strategy (THESIS cell-18 `perf()`; the SPY
    benchmark is now an explicit parameter instead of a notebook global).
    """
    s = returns.dropna()
    if len(s) < 20:
        return {}
    m       = perf_stats(s, sortino_variant="negonly")
    cagr, vol, sr, so, mdd, hit = (m["CAGR"], m["AnnVol"], m["Sharpe"],
                                  m["Sortino"], m["MaxDD"], m["HitRate"])
    cal   = cagr / (abs(mdd) + 1e-10)
    alpha = alpha_vs_benchmark(s, benchmark)["alpha_ann"]
    return {"CAGR": f"{cagr:.2%}", "Vol": f"{vol:.2%}", "Sharpe": f"{sr:.3f}",
            "Sortino": f"{so:.3f}", "MaxDD": f"{mdd:.2%}", "Calmar": f"{cal:.2f}",
            "HitRate": f"{hit:.1%}", "Alpha_ann": f"{alpha:.2%}"}


def calc_metrics(s: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """Formatted metrics row (THESIS cell-36 `calc_metrics`)."""
    m = perf_stats(s, risk_free_rate=risk_free_rate, sortino_variant="downside")
    return {
        "CAGR": f"{m['CAGR']:.2%}",
        "Ann. Vol": f"{m['AnnVol']:.2%}",
        "Sharpe": f"{m['Sharpe']:.3f}",
        "Sortino": f"{m['Sortino']:.3f}",
        "Max DD": f"{m['MaxDD']:.2%}",
        "Hit Rate": f"{m['HitRate']:.1%}",
    }


def print_perf_report(s: pd.Series, label: str) -> None:
    """Print the full per-strategy stats block (THESIS cell-38 `_stats`)."""
    s       = s.dropna()
    m       = perf_stats(s, sortino_variant="downside")
    cagr, vol, sharpe, sortino, mdd, hit = (m["CAGR"], m["AnnVol"], m["Sharpe"],
                                            m["Sortino"], m["MaxDD"], m["HitRate"])
    skew   = float(s.skew())
    kurt   = float(s.kurt())
    print(f"  [{label}]")
    print(f"    Periodi OOS   : {len(s)}")
    print(f"    Range date    : {s.index[0].date()} -> {s.index[-1].date()}")
    print(f"    CAGR          : {cagr:+.2%}")
    print(f"    Ann. Vol      : {vol:.2%}")
    print(f"    Sharpe        : {sharpe:+.3f}")
    print(f"    Sortino       : {sortino:+.3f}")
    print(f"    Max DD        : {mdd:.2%}")
    print(f"    Hit Rate      : {hit:.1%}")
    print(f"    Skewness      : {skew:+.3f}")
    print(f"    Kurtosis      : {kurt:+.3f}")


def ir_vs(s: pd.Series, bench: pd.Series) -> dict:
    """Annualised alpha + information ratio vs a benchmark (THESIS cell-32 `ir_vs`)."""
    c = s.index.intersection(bench.index)
    pair = pd.concat([s.reindex(c), bench.reindex(c)], axis=1).dropna()
    _ab = alpha_vs_benchmark(pair.iloc[:, 0], pair.iloc[:, 1])
    return {"Alpha_ann": f"{_ab['alpha_ann']:.2%}", "IR": f"{_ab['ir']:.3f}"}
# ── 7. IC / ALPHA / TURNOVER diagnostics ───────────────────────────────
def weekly_ic(scores: dict, rets: pd.Series, min_n: int = 5):
    """
    Cross-sectional Spearman IC between a {ticker: score} mapping and the
    matching {ticker: return} series.
    Returns float IC, or np.nan if fewer than `min_n` common non-NaN pairs.
    """
    from scipy.stats import spearmanr as _sp
    common = [k for k in scores if k in rets.index]
    if len(common) < min_n:
        return np.nan
    sv  = pd.Series({k: scores[k]  for k in common})
    rv  = rets.reindex(common)
    df  = pd.concat([sv, rv], axis=1).dropna()
    if len(df) < min_n:
        return np.nan
    r, _ = _sp(df.iloc[:, 0], df.iloc[:, 1])
    return float(r) if not np.isnan(r) else np.nan


def icir(ic_list):
    """
    Information coefficient ratio: mean(IC) / (std(IC) + 1e-10), numpy std
    (ddof=0) — identical to the original THESIS inline ICIR sites.
    NaN-safe. Returns np.nan for an empty/NaN-only list.
    """
    arr = np.asarray([x for x in ic_list if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(arr) == 0:
        return np.nan
    m   = arr.mean()
    sd  = arr.std(ddof=0)
    return float(m / (sd + 1e-10))


def spy_aligned_returns(spy_close: pd.Series, index: pd.Index) -> pd.Series:
    """
    Weekly SPY simple returns aligned to `index` via ffill-reindex (THESIS pattern).
    Returns a clean pd.Series of weekly pct-changes on the given index.
    """
    prices = spy_close.reindex(index, method="ffill")
    return prices.pct_change().dropna()


def alpha_vs_benchmark(strategy: pd.Series, benchmark: pd.Series,
                       risk_free_rate: float = 0.0) -> dict:
    """
    Alpha vs a benchmark (e.g. SPY), annualised.
    Returns {alpha_ann, ir, pct_beat} where
        alpha_ann : mean(strategy - bench.reindex(idx).fillna(0)) * 52
        ir        : alpha_ann / std(alpha) * sqrt(52)
        pct_beat  : fraction of weeks where strategy > benchmark
    """
    s     = strategy.dropna()
    idx   = s.index
    bench = benchmark.reindex(idx).fillna(0.0)
    alpha = (s - bench).dropna()
    if len(alpha) < 2:
        return {"alpha_ann": np.nan, "ir": np.nan, "pct_beat": np.nan}
    alpha_ann = float(alpha.mean()) * 52.0
    vol_a     = float(alpha.std(ddof=1)) * np.sqrt(52)
    ir        = (alpha_ann - risk_free_rate) / (vol_a + 1e-10)
    pct_beat  = float((s > bench).mean())
    return {"alpha_ann": alpha_ann, "ir": float(ir), "pct_beat": pct_beat}


def turnover_symmetric(prev: list, curr: list) -> float:
    """
    Symmetric weekly turnover (THESIS T-7 / T-8):
        (len(added) + len(removed)) / 2 / max(len(curr), 1)
    """
    p, c = set(prev or []), set(curr or [])
    added   = c - p
    removed = p - c
    return float((len(added) + len(removed))) / 2.0 / max(len(c), 1)


def turnover_one_way(prev: list, curr: list) -> float:
    """
    One-way (additions-only) weekly turnover (THESIS T-9 regime cell / T-14i [5B]):
        len(curr - prev) / max(len(curr), 1)
    NOTE: intentionally different from `turnover_symmetric`; do NOT unify.
    """
    p, c = set(prev or []), set(curr or [])
    return float(len(c - p)) / max(len(c), 1)
#  6. STATISTICAL SIGNIFICANCE — JKM, Deflated SR, MP denoising, CSCV-PBO ───────────
def jobson_korkie_memmel(r1: pd.Series, r2: pd.Series,
                         annualization_factor: float = 52):
    """
    Memmel (2003) corrected Jobson-Korkie test for two overlapping Sharpe Ratios.
    Returns (z_stat, p_value).
    """
    df = pd.concat([r1, r2], axis=1).dropna()
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    T = len(x)
    if T < 2:
        return np.nan, np.nan
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    var_x, var_y   = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    std_x, std_y   = np.sqrt(var_x), np.sqrt(var_y)
    sr_x_p = mean_x / std_x
    sr_y_p = mean_y / std_y
    rho    = float(np.corrcoef(x, y)[0, 1])
    var_diff = (1.0 / T) * (2.0 * (1.0 - rho)
                           + 0.5 * (sr_x_p ** 2 + sr_y_p ** 2
                                   - 2.0 * sr_x_p * sr_y_p * (rho ** 2)))
    if var_diff <= 0:
        return 0.0, 1.0
    z_stat = (sr_x_p - sr_y_p) / np.sqrt(var_diff)
    p_value = float(2.0 * (1.0 - _norm.cdf(abs(z_stat))))
    return float(z_stat), p_value


def deflated_sr(sr_obs: float, n_obs: int, n_trials: int) -> float:
    """
    Deflated Sharpe Ratio (Lopez de Prado 2014).
    Returns P(SR > SR_star) as a probability in [0, 1].
    """
    euler_gamma = 0.5772156649
    sr_star = ((1 - euler_gamma) * _norm.ppf(1 - 1 / n_trials)
               + euler_gamma * _norm.ppf(1 - 1 / (n_trials * _math.e)))
    sr_star *= np.sqrt(1.0 / n_obs)
    sr_weekly = sr_obs / np.sqrt(52)
    z = (sr_weekly - sr_star) * np.sqrt(n_obs - 1)
    return float(_norm.cdf(z))


def _mp_denoise(cov: np.ndarray, q: float = 1.0) -> np.ndarray:
    """
    Marchenko-Pastur eigenvalue denoising of a covariance matrix.
    Returns the re-scaled denoised covariance array (same shape as input).
    """
    std   = np.sqrt(np.diag(cov))
    std   = np.where(std < 1e-10, 1e-10, std)
    corr  = np.clip(cov / np.outer(std, std), -1, 1)
    np.fill_diagonal(corr, 1.0)
    evals, evecs = np.linalg.eigh(corr)
    lmax         = (1.0 + 1.0 / max(q, 0.01)) ** 2
    noise_mask   = evals < lmax
    if noise_mask.any():
        evals_c = evals.copy()
        evals_c[noise_mask] = float(evals[noise_mask].mean())
    else:
        evals_c = evals
    corr_dn = evecs @ np.diag(evals_c) @ evecs.T
    np.fill_diagonal(corr_dn, 1.0)
    return (corr_dn * np.outer(std, std)).astype(float)


def cscv_pbo(returns_dict: dict, S: int = 10, min_block: int = 10):
    """
    Combinatorially Symmetric Cross-Validation for Probability of Backtest Overfitting.
    Bailey, Borwein, Lopez de Prado & Zhu (2017).

    Returns (pbo: float in [0,1], logits: np.ndarray, names: list[str]).
    """
    if S % 2 != 0:
        S += 1
    names = list(returns_dict.keys())
    df    = pd.DataFrame({n: returns_dict[n] for n in names}).dropna(how="any")
    T     = len(df)
    block_size = T // S
    if block_size < min_block:
        raise ValueError(
            f"Only {T} shared observations across {len(names)} strategies -- "
            f"S={S} gives block_size={block_size} < min_block={min_block}. Reduce S.")
    T_trim = block_size * S
    df     = df.iloc[-T_trim:]
    blocks = [df.iloc[i * block_size:(i + 1) * block_size] for i in range(S)]
    half   = S // 2
    is_index_sets = list(_combinations(range(S), half))
    logits = []
    for is_idx in is_index_sets:
        oos_idx = tuple(i for i in range(S) if i not in is_idx)
        is_df   = pd.concat([blocks[i] for i in is_idx])
        oos_df  = pd.concat([blocks[i] for i in oos_idx])
        is_sharpe  = is_df.mean()  / (is_df.std()  + 1e-12)
        oos_sharpe = oos_df.mean() / (oos_df.std() + 1e-12)
        best_is_strategy = is_sharpe.idxmax()
        oos_rank = oos_sharpe.rank(method="average")[best_is_strategy]
        rel_rank = np.clip(oos_rank / (len(names) + 1), 1e-6, 1 - 1e-6)
        logits.append(float(np.log(rel_rank / (1 - rel_rank))))
    logits_arr = np.asarray(logits, dtype=float)
    pbo = float((logits_arr < 0).mean())
    return pbo, logits_arr, names


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTION FUNNEL & CONSTRUCTION METHODS  (THESIS Section-1 core logic)
#  All state is passed via __init__ / explicit parameters — no notebook globals.
# ══════════════════════════════════════════════════════════════════════════════

def soft_exposure(vix_pct_val: float, vix_mid: float = 0.65,
                  vix_scale: float = 0.14, exp_min: float = 1.0,
                  exp_max: float = 1.0) -> float:
    """
    Sigmoid VIX soft-exposure scalar (THESIS cell-12 `soft_exposure`).
    Defaults match THESIS config; with EXP_MIN == EXP_MAX the result is always 1.0.
    Re-enable exposure scaling by passing different exp_min/exp_max at call sites.
    """
    sig = 1.0 / (1.0 + np.exp(-(vix_pct_val - vix_mid) / vix_scale))
    return float(np.clip(1.0 + (exp_min - 1.0) * sig, exp_min, exp_max))


class SelectionFunnel:
    """
    Two-layer rank funnel over z-scored features (original THESIS `layer2_funnel`,
    `layer1_conviction` and `get_selection`).

    Layer-2 (breadth):  composite score of the LAYER-2 features
                        (default poc_cross + mom_52w), keep top K2 tickers.
    Layer-1 (conviction): kalman_dev_z and sma_deriv_neg_z z-scores, keep top K1 out of K2.

    The `*_weights` variants take an arbitrary {feature: weight} dict so the same
    method serves every entry of THESIS WEIGHT_SCHEMES without separate code paths.
    """

    def __init__(self, feat_raw: Dict[str, pd.DataFrame],
                 kalman_z: Dict[str, pd.Series],
                 sma_deriv_neg_z: Dict[str, pd.Series],
                 dates_full: pd.Index,
                 assets: List[str],
                 w_poc: float = 1.0, w_mom: float = 1.0,
                 w_kalman: float = 1.0, w_sma_neg: float = -1.0,
                 k1: int = 3, k2: int = 5):
        self.feat_raw          = feat_raw
        self.kalman_z          = kalman_z
        self.sma_deriv_neg_z   = sma_deriv_neg_z
        self.dates_full        = dates_full
        self.assets            = list(assets)
        self.w_poc, self.w_mom    = w_poc, w_mom
        self.w_kalman, self.w_sma_neg = w_kalman, w_sma_neg
        self.k1, self.k2       = k1, k2

    # ── Layer 2 ────────────────────────────────────────────────────────────
    def layer2_funnel(self, t: int) -> list:
        """Top K2 tickers by the default LAYER-2 composite score (poc_cross + mom_52w)."""
        return self.layer2_funnel_weights(t, {"poc_cross": self.w_poc,
                                              "mom_52w": self.w_mom}, k2=self.k2)

    def layer2_funnel_weights(self, t: int, weights_l2: Dict[str, float],
                             k2: Optional[int] = None) -> list:
        """Top K2 tickers by a generic {feature: weight} LAYER-2 composite score.
        (original `layer2_funnel_scheme`: k2 is explicit per call)."""
        if k2 is None:
            k2 = self.k2
        dt_prev = self.dates_full[t - 1]
        scores = []
        for tkr in self.assets:
            if tkr not in self.feat_raw:
                continue
            try:
                row = self.feat_raw[tkr].loc[dt_prev, list(weights_l2)]
                if row.isna().any():
                    continue
                scores.append((sum(weights_l2[f] * float(row[f]) for f in weights_l2), tkr))
            except KeyError:
                continue
        scores.sort(key=lambda x: -x[0])
        return [tkr for s, tkr in scores[: k2]]

    def layer2_composite(self, t: int, weights_l2: Optional[Dict[str, float]] = None) -> dict:
        """LAYER-2 composite score of EVERY asset at dt_prev (diagnostics / meta-labels)."""
        w = weights_l2 if weights_l2 is not None else {"poc_cross": self.w_poc, "mom_52w": self.w_mom}
        out: Dict[str, float] = {}
        dt_prev = self.dates_full[t - 1]
        for tkr in self.assets:
            if tkr not in self.feat_raw:
                continue
            try:
                row = self.feat_raw[tkr].loc[dt_prev, list(w)]
                if row.isna().any():
                    continue
                out[tkr] = sum(w[f] * float(row[f]) for f in w)
            except KeyError:
                continue
        return out

    # ── Layer 1 ────────────────────────────────────────────────────────────
    def layer1_conviction(self, candidates: list, t: int,
                         k1: Optional[int] = None) -> dict:
        """Original `layer1_conviction`: LAYER-1 conviction score for `candidates`.
        k1=None (default) scores ALL candidates — the top-K1 slicing happens
        downstream via apply_hysteresis_margin, exactly as in THESIS."""
        return self.layer1_conviction_weights(
            candidates, t, k1,
            {"kalman_dev": self.w_kalman, "sma_deriv_neg": self.w_sma_neg})

    def layer1_conviction_weights(self, candidates: list, t: int, k1: Optional[int],
                                 weights_l1: Dict[str, float]) -> dict:
        """Generic {feature: weight} LAYER-1 score (original `layer1_conviction_scheme`).
        k1=None keeps all scored candidates; k1=int keeps the top k1."""
        dt_prev = self.dates_full[t - 1]
        convs = []
        for tkr in candidates:
            try:
                kv = self.kalman_z[tkr].loc[dt_prev]
                sv = self.sma_deriv_neg_z[tkr].loc[dt_prev]
                if np.isfinite(kv) and np.isfinite(sv):
                    convs.append((weights_l1["kalman_dev"] * kv + weights_l1["sma_deriv_neg"] * sv, tkr))
            except KeyError:
                continue
        convs.sort(key=lambda x: -x[0])
        top = convs if k1 is None else convs[: k1]
        return {tkr: float(c) for c, tkr in top}

    def get_selection(self, t: int) -> dict:
        """Full funnel at index t (original THESIS `get_selection`, verbatim logic)."""
        l2 = self.layer2_funnel(t)
        if not l2:
            return {}
        return self.layer1_conviction(l2, t)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION METHODS  (original THESIS cell-14 weighting schemes)
# ══════════════════════════════════════════════════════════════════════════════

class ConstructionMethods:
    """
    Weighting schemes applied to the selected K tickers. `assets`/`prices_mat` are
    passed in via __init__ so no notebook globals are needed (original THESIS used
    module-scope ASSETS, prices_mat, VOL_LOOKBACK, COV_LOOKBACK).
    """

    def __init__(self, assets: List[str], prices_mat: np.ndarray,
                 vol_lookback: int = 13, cov_lookback: int = 26):
        self.assets       = list(assets)
        self.prices_mat   = prices_mat
        self.vol_lookback = vol_lookback
        self.cov_lookback = cov_lookback

    def cov_matrix(self, tickers: list, t: int, lookback: Optional[int] = None) -> np.ndarray:
        """Realised weekly-return covariance matrix for given tickers, up to t-1."""
        if lookback is None:
            lookback = self.cov_lookback
        idxs = [self.assets.index(tkr) for tkr in tickers]
        p = self.prices_mat[max(0, t - lookback - 1):t, idxs]
        if p.shape[0] < 5:
            return np.eye(len(tickers)) * 1e-4
        rets = np.diff(np.log(p + 1e-10), axis=0)
        cov = np.cov(rets, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # Tiny ridge for numerical stability
        cov += np.eye(len(tickers)) * 1e-8
        return cov

    def weights_equal(self, tickers: list, t: int, scores: dict) -> np.ndarray:
        n = len(tickers)
        return np.ones(n) / n

    def weights_inv_vol(self, tickers: list, t: int, scores: dict,
                        lookback: Optional[int] = None) -> np.ndarray:
        if lookback is None:
            lookback = self.vol_lookback
        vols = []
        for tkr in tickers:
            gi = self.assets.index(tkr)
            p_vec = self.prices_mat[max(0, t - lookback):t, gi]
            lr = np.log(p_vec[1:] / (p_vec[:-1] + 1e-10))
            vols.append(lr.std() * np.sqrt(52) + 1e-6)
        inv = 1.0 / np.array(vols)
        return inv / inv.sum()

    def weights_score(self, tickers: list, t: int, scores: dict) -> np.ndarray:
        """v6's method: softmax(conviction score) x inverse-vol."""
        # Keep the z-scores with their original sign (the softmax handles negatives natively)
        raw = np.array([scores.get(tkr, 0.0) for tkr in tickers])
        exp_s = np.exp(raw - raw.max())
        soft = exp_s / exp_s.sum()
        inv_v = self.weights_inv_vol(tickers, t, scores)
        w = soft * inv_v
        return w / w.sum()

    def weights_erc(self, tickers: list, t: int, scores: dict,
                    max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
        """
        Equal Risk Contribution via iterative proportional adjustment.
        Each weight is adjusted proportional to (target_risk / actual_risk_contrib).
        """
        n = len(tickers)
        if n < 2:
            return np.ones(n) / max(n, 1)
        cov = self.cov_matrix(tickers, t)
        w = np.ones(n) / n
        for _ in range(max_iter):
            port_var = w @ cov @ w
            if port_var < 1e-14:
                break
            marginal_contrib = cov @ w
            risk_contrib = w * marginal_contrib
            target = port_var / n
            # Multiplicative update toward equal risk contribution
            ratio = np.sqrt(target / (risk_contrib + 1e-12))
            w_new = w * ratio
            w_new = np.clip(w_new, 1e-6, None)
            w_new = w_new / w_new.sum()
            if np.abs(w_new - w).max() < tol:
                w = w_new
                break
            w = w_new
        return w

    # ── HRP internals (Lopez de Prado 2016) ────────────────────────────────
    def _hrp_quasi_diag(self, link: np.ndarray) -> list:
        """Recursive function to get the order of leaves in the dendrogram."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _hrp_cluster_var(self, cov: np.ndarray, items: list) -> float:
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    def _hrp_recursive_bisect(self, cov: np.ndarray, sort_ix: list) -> np.ndarray:
        w = pd.Series(1.0, index=sort_ix)
        clusters = [sort_ix]
        while clusters:
            clusters = [c[start:end] for c in clusters
                        for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
                        if len(c) > 1]
            for i in range(0, len(clusters), 2):
                if i + 1 >= len(clusters):
                    continue
                c0, c1 = clusters[i], clusters[i + 1]
                v0 = self._hrp_cluster_var(cov, c0)
                v1 = self._hrp_cluster_var(cov, c1)
                alpha = 1 - v0 / (v0 + v1 + 1e-12)
                w[c0] *= alpha
                w[c1] *= (1 - alpha)
        return w.sort_index().values

    def weights_hrp(self, tickers: list, t: int, scores: dict, lookback: Optional[int] = None) -> np.ndarray:
        """
        Hierarchical Risk Parity (Lopez de Prado 2016).
        1. Correlation → distance matrix → hierarchical clustering
        2. Quasi-diagonalize (reorder assets so correlated ones are adjacent)
        3. Recursive bisection: split risk budget by inverse cluster variance
        """
        if lookback is None:
            lookback = self.cov_lookback
        n = len(tickers)
        if n < 3:
            return self.weights_inv_vol(tickers, t, scores)
        cov  = self.cov_matrix(tickers, t, lookback=lookback)
        std  = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)

        dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = self._hrp_quasi_diag(link)
        w = self._hrp_recursive_bisect(cov, sort_ix)
        return w / w.sum()


def run_funnel_with_hysteresis(scores_source: dict, delta: float, engine: "ConstructionMethods",
                               vix_pct: pd.Series, dates_full: pd.Index, common_index: pd.Index,
                               k: int = 3) -> tuple:
    """
    Hysteresis-ablation walk (original THESIS `run_funnel_with_hysteresis_exact`).
    Returns (s_h, wk_to, wk_reb): OOS return series + weekly one-way turnover +
    weekly two-way weight-drift.
    """
    rets_hyst       = {}
    to_wks_discrete = []
    to_wks_weight   = []
    prev_held       = []
    prev_w_full     = None

    for t in sorted(scores_source.keys()):
        scores_raw = scores_source[t]
        dt      = dates_full[t]
        dt_prev = dates_full[t - 1]

        # Hysteresis: margin-based incumbent protection (delta=0 → simple top-K)
        scores = apply_hysteresis_margin(scores_raw, prev_held if delta > 0 else [], k, delta)

        if len(scores) < 2:
            continue

        final = list(scores.keys())

        if prev_held:
            to_wks_discrete.append(len(set(final) - set(prev_held)) / max(len(final), 1))
            # ← check drift BEFORE updating prev_held
            no_change = set(final) == set(prev_held)
        else:
            no_change = False

        prev_held = list(final)   # ← update AFTER the check

        w = engine.weights_score(final, t, scores)
        w_curr_full = np.zeros(len(engine.assets))
        for idx, tk in enumerate(final):
            w_curr_full[engine.assets.index(tk)] = w[idx]

        if prev_w_full is not None:
            asset_rets_all = engine.prices_mat[t, :] / engine.prices_mat[t - 1, :] - 1.0
            w_drifted = prev_w_full * (1.0 + asset_rets_all)
            sum_drifted = np.sum(w_drifted)
            w_drifted = w_drifted / sum_drifted if sum_drifted > 0 else prev_w_full.copy()

            if no_change:
                kept_idx   = [engine.assets.index(tk) for tk in final]
                drift_only = np.sum(np.abs(w_curr_full[kept_idx] - w_drifted[kept_idx]))
                to_wks_weight.append(drift_only)
            else:
                to_wks_weight.append(0.0)

        prev_w_full = w_curr_full.copy()

        rets_vec = np.array([
            engine.prices_mat[t, engine.assets.index(tk)] / engine.prices_mat[t - 1, engine.assets.index(tk)] - 1
            for tk in final
        ])
        rets_hyst[dt] = float(np.dot(w, rets_vec)) * soft_exposure(vix_pct.get(dt_prev, 0.5))

    s_h      = pd.Series(rets_hyst).sort_index().reindex(common_index).fillna(0)
    wk_to    = float(np.mean(to_wks_discrete)) if to_wks_discrete else 0.0   # weekly one-way
    wk_reb   = float(np.mean(to_wks_weight))   if to_wks_weight   else 0.0   # weekly two-way drift

    return s_h, wk_to, wk_reb
# 10. ML FEATURE UTILITIES & DATA LOADING ──────────────────────────────────────────
def extract_features_at(feat_raw: dict, tkr: str, dt, feature_list: list,
                        reject_inf: bool = True):
    """
    Return [feat_raw[tkr][f].loc[dt] for f in feature_list], or None if any
    value is missing/NaN (and non-finite when reject_inf=True).
    reject_inf=False reproduces the LGB OOS scoring block semantics
    (NaN-only check); reject_inf=True reproduces build_full_panel semantics.
    """
    try:
        vals = [feat_raw[tkr][f].loc[dt] for f in feature_list]
    except KeyError:
        return None
    if any(pd.isna(v) for v in vals):
        return None
    if reject_inf and any(np.isinf(v) for v in vals):
        return None
    return vals


def apply_cs_rank(df: pd.DataFrame, cs_features: list) -> pd.DataFrame:
    """
    Cross-sectional percentile rank centred in [-0.5, +0.5] applied ONLY to
    cs_features (macro features pass through untouched). THESIS LGB pattern.
    """
    df[cs_features] = df[cs_features].rank(pct=True) - 0.5
    return df


# ── Data loading (THESIS cell 5 loader, centralised Wave 3, item 12) ───────────
def load_weekly_universe(excel_path: str, universe_keep, anchor_start: str,
                         benchmark: str = "SPY", excl=("SPY", "QQQ"),
                         skip_sheets=("Riepilogo_Chiusure", "Riepilogo", "Info", "Sheet1"),
                         verbose: bool = True) -> dict:
    """
    Load daily OHLCV from a multi-sheet Excel workbook (+ yfinance fallback for
    benchmark and ^VIX with local xlsx cache), resample to W-FRI weekly bars,
    align all series on the common date index truncated at anchor_start.

    Reproduces the THESIS data-loading cell EXACTLY (including prints and the
    local-cache fallback). Returns dict with keys:
        ohlcv      — {ticker: weekly OHLCV DataFrame on common index}
        common     — common DatetimeIndex
        assets     — sorted list of universe tickers present
        spy_close  — benchmark Close (ffilled; indexed on common)
        vix_weekly — ^VIX Close on common (or constant 20.0 if unavailable)
        raw_data   — raw daily frames (pre-resample)
    """
    import yfinance as yf

    TV_RENAME   = {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}
    OHLCV_COLS  = ["Open","High","Low","Close","Volume"]
    SKIP_SHEETS = set(skip_sheets)

    raw_data = {}
    xl = pd.ExcelFile(excel_path)
    for sheet in xl.sheet_names:
        if sheet in SKIP_SHEETS or sheet.startswith("_"): continue
        try:
            df = xl.parse(sheet, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()].sort_index()
            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns={c: TV_RENAME.get(c.lower(), c) for c in df.columns})
            if "Close" not in df.columns: continue
            df = df[[c for c in OHLCV_COLS if c in df.columns]].copy()
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"])
            if not df.empty:
                raw_data[sheet] = df
                if verbose: print(f"  + {sheet:<14}  {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:
            if verbose: print(f"  ! {sheet}: {e}")

    for tkr in [benchmark, "^VIX"]:
        if tkr in raw_data: continue
            # Nome pulito per il file locale (es. '^VIX' diventa 'vix.xlsx')
        file_cache = f"{tkr.replace('^', '').lower()}.xlsx"
        try:
            df = yf.download(tkr, start=anchor_start, interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df = df[[c for c in OHLCV_COLS if c in df.columns]].dropna(subset=["Close"])
            if df.empty:
                raise ValueError("Dataframe scaricato vuoto")
            raw_data[tkr] = df
            if verbose: print(f"  + {tkr:<14}  yfinance")
            df.to_excel(file_cache, index_label="datetime")
        except Exception as e:
            # 2. FALLBACK LOCALE: Se yfinance fallisce, legge dal file locale salvato in precedenza
            try:
                df_local = pd.read_excel(file_cache, index_col=0, parse_dates=True)
                df_local.index = pd.to_datetime(df_local.index)
                raw_data[tkr] = df_local
                if verbose: print(f"  + {tkr:<14}  caricato da cache locale ('{file_cache}')")
            except Exception:
                # Se fallisce anche il caricamento locale, stampa l'errore originale di yfinance
                if verbose: print(f"  ! {tkr}: {e} (nessun backup locale trovato)")

    assets = sorted([t for t in raw_data if t not in set(excl) and t != "^VIX" and t in set(universe_keep)])

    ohlcv: dict = {}
    for ticker in assets + [benchmark, "^VIX"]:
        if ticker not in raw_data: continue
        d   = raw_data[ticker]
        has = lambda c: c in d.columns
        wk  = pd.DataFrame({
            "Open"  : d["Open"].resample("W-FRI").first()  if has("Open")  else d["Close"].resample("W-FRI").first(),
            "High"  : d["High"].resample("W-FRI").max()    if has("High")  else d["Close"].resample("W-FRI").max(),
            "Low"   : d["Low"].resample("W-FRI").min()     if has("Low")   else d["Close"].resample("W-FRI").min(),
            "Close" : d["Close"].resample("W-FRI").last(),
            "Volume": d["Volume"].resample("W-FRI").sum()  if has("Volume") else
                      pd.Series(1e6, index=d["Close"].resample("W-FRI").last().index),
        }).ffill().dropna(subset=["Close"])
        ohlcv[ticker] = wk

    common = None
    for t in assets:
        idx = ohlcv[t].index
        common = idx if common is None else common.intersection(idx)
    common = common[common >= pd.Timestamp(anchor_start)]
    for t in assets + [benchmark, "^VIX"]:
        if t in ohlcv:
            ohlcv[t] = ohlcv[t].reindex(common).ffill()

    spy_close  = ohlcv[benchmark]["Close"].ffill()
    vix_weekly = (ohlcv["^VIX"]["Close"].reindex(common).ffill()
                  if "^VIX" in ohlcv else pd.Series(20.0, index=common))

    if verbose:
        print(f"\nUniverse : {len(assets)} assets  |  {common[0].date()} -> {common[-1].date()}  ({len(common)}w)")
        print(f"Assets   : {assets}")

    return {"ohlcv": ohlcv, "common": common, "assets": assets,
            "spy_close": spy_close, "vix_weekly": vix_weekly, "raw_data": raw_data}



# ══════════════════════════════════════════════════════════════════════════════
#  ML BENCHMARK HELPERS — XGBoost & LightGBM walk-forward (THESIS Sections 5-7)
#  Heavy libs (xgboost/lightgbm) are imported lazily inside the functions so that
#  importing helpers.py stays cheap for notebooks that do not use them.
# ══════════════════════════════════════════════════════════════════════════════

class BufferPortfolio:
    """
    Hysteresis-buffered top-K portfolio state (original THESIS `last_top_portfolio`
    global + `_portfolio_return`, used by both ML sections). One instance per section;
    no notebook globals. Weights = softmax(scores) x inverse-vol via the engine.
    """

    def __init__(self, engine: "ConstructionMethods", k: int = 3, buffer_threshold: float = 0.0):
        self.engine            = engine
        self.k                 = k
        self.buffer_threshold  = buffer_threshold
        self.last: list = []

    def reset(self) -> None:
        self.last = []

    def step(self, scores: dict, t: int):
        """
        Select top-K with an anti-turnover buffer, weight softmax x inverse-vol and
        return the portfolio's realised return at step t (None if < K candidates).
        """
        if len(scores) < self.k:
            return None

        ranked_candidates = sorted(scores, key=scores.get, reverse=True)

        # 1. Selection with hysteresis buffer
        if not self.last:
            top = ranked_candidates[: self.k]
        else:
            top = list(self.last)
            for candidate in ranked_candidates:
                if candidate not in top:
                    worst_holding = min(top, key=lambda tkr: scores.get(tkr, -999.0))
                    # Replace ONLY if the outsider beats the weakest holding by > buffer
                    if scores[candidate] - scores[worst_holding] > self.buffer_threshold:
                        top.remove(worst_holding)
                        top.append(candidate)

        self.last = list(top)   # state for the next step (t+1)

        # 2. Weights: softmax x inverse-vol
        raw_scores = np.array([max(scores[tkr], 1e-6) for tkr in top])
        exp_s = np.exp(raw_scores - raw_scores.max())
        soft  = exp_s / exp_s.sum()

        vols = []
        for tkr in top:
            gi = self.engine.assets.index(tkr)
            pv = self.engine.prices_mat[max(0, t - self.engine.vol_lookback):t, gi]
            lr = np.log(pv[1:] / (pv[:-1] + 1e-10))
            vols.append(lr.std() * np.sqrt(52) + 1e-6)

        inv_v = 1.0 / np.array(vols)
        w     = soft * inv_v
        w    /= w.sum()

        # 3. Realised return
        rets = [
            self.engine.prices_mat[t, self.engine.assets.index(tkr)] / self.engine.prices_mat[t - 1, self.engine.assets.index(tkr)] - 1
            for tkr in top
        ]
        return float(np.dot(w, rets))


# ── XGBoost helpers (original THESIS cell-32 local defs) ───────────────────

def check_features(feat_raw: dict, tkr: str, features: List[str]) -> bool:
    """Return True only if every required feature column exists for this ticker."""
    if tkr not in feat_raw:
        return False
    missing = [f for f in features if f not in feat_raw[tkr].columns]
    if missing:
        print(f"  [xgb] {tkr}: skipping — missing columns {missing}")
        return False
    return True


def sharpe_target(close: pd.Series) -> pd.Series:
    """Sign-scaled Sharpe contribution as the regression target (original `_sharpe_target`)."""
    cl      = close
    fwd_ret = np.log(cl.shift(-1) / cl)
    rvol    = (
        np.log(cl / cl.shift(1))
        .rolling(4, min_periods=2)
        .std()
        .mul(np.sqrt(52))
        .replace(0, np.nan)
        .ffill()
    )
    return (np.sign(fwd_ret) * fwd_ret.abs() / (rvol + 1e-8)).clip(-3, 3)


def ticker_feature_row(feat_raw: dict, tkr: str, dt, features: List[str], vix_pct: pd.Series,
                       lookback: int = 52) -> "np.ndarray | None":
    """
    Return a (1, n_features) float32 array for one ticker at date dt.
    Returns None if any feature value is non-finite or missing.
    (vix_pct is computed in the row dict but dropped from the training matrix,
    exactly as the original THESIS code did.)
    """
    row = {}
    try:
        for feat in features:
            val = zscore_ts(feat_raw[tkr][feat], lookback).loc[dt]
            if not np.isfinite(val):
                return None
            row[feat] = val
        row["vix_pct"] = float(vix_pct.get(dt, 0.5))
    except KeyError:
        return None
    return np.array([[row[f] for f in features]], dtype=np.float32)


def build_xgb_panel(assets: List[str], feat_raw: dict, ohlcv: dict, vix_pct: pd.Series,
                    features: List[str], up_to_t: int, lookback: int = 52) -> pd.DataFrame:
    """
    Build an expanding-window training panel.
    Rows = (date, ticker); columns = normalised features + vix_pct + target y.
    Tickers missing any required feature column are silently skipped.
    """
    rows = []
    for tkr in assets:
        if not check_features(feat_raw, tkr, features):
            continue

        norm_feats = {feat: zscore_ts(feat_raw[tkr][feat], lookback) for feat in features}

        df_tkr = pd.DataFrame(norm_feats)
        df_tkr["vix_pct"] = vix_pct.reindex(df_tkr.index).fillna(0.5)
        df_tkr["y"]       = sharpe_target(ohlcv[tkr]["Close"])
        df_tkr["asset"]   = tkr
        rows.append(df_tkr.iloc[:up_to_t])

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows)
        .sort_index()
        .dropna(subset=list(features) + ["y"])
    )


def train_xgb(panel: pd.DataFrame, features: List[str], params: dict,
              min_train_dates: int = 78, val_dates: int = 52):
    """
    Fit XGBRegressor with a rolling validation split.
    Returns (model, ic_val) or (None, nan) if data is insufficient.
    """
    import xgboost as _xgb
    from scipy.stats import spearmanr as _spearmanr

    udates = sorted(panel.index.unique())
    if len(udates) < min_train_dates:
        return None, float("nan")

    val_set  = set(udates[-val_dates:])
    tr       = panel[~panel.index.isin(val_set)]
    va       = panel[ panel.index.isin(val_set)]

    X_tr, y_tr = tr[list(features)].values.astype(np.float32), tr["y"].values.astype(np.float32)
    X_va, y_va = va[list(features)].values.astype(np.float32), va["y"].values.astype(np.float32)

    model = _xgb.XGBRegressor(objective="reg:squarederror", **params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    ic_val, _ = _spearmanr(model.predict(X_va), y_va)
    return model, float(ic_val)


def score_assets(model, assets: List[str], feat_raw: dict, features: List[str],
                 vix_pct: pd.Series, dt_prev) -> "dict[str, float]":
    """Score all eligible assets at a given date. Returns {tkr: score}."""
    scores = {}
    for tkr in assets:
        if not check_features(feat_raw, tkr, features):
            continue
        x = ticker_feature_row(feat_raw, tkr, dt_prev, features, vix_pct)
        if x is None:
            continue
        scores[tkr] = float(model.predict(x)[0])
    return scores


# ── LightGBM LambdaRank helpers (original THESIS cell-35 local defs) ────────

def build_lgb_panel(dates_full: pd.Index, assets: List[str], feat_raw: dict, ohlcv: dict,
                    features: List[str], k_lgb: int = 3, n_bins: int = 5,
                    norm_lookback: int = 52) -> pd.DataFrame:
    """
    Builds the LambdaRank panel for all time steps.
    Target: n_bins relevance bins (0..n_bins-1) on excess returns vs market average.
    CS features: percentile rank centered in [-0.5, +0.5]. Macro features left as-is.
    """
    daily_panels = []
    n_steps = len(dates_full) - 2   # -2: l'ultimo step non ha fwd return

    for step in range(norm_lookback, n_steps):
        dt = dates_full[step]
        row_feats, targets = {}, {}

        for tkr in assets:
            if tkr not in feat_raw or tkr not in ohlcv:
                continue
            try:
                # Estrazione valori feature base
                f_vals = extract_features_at(feat_raw, tkr, dt, features)
                if f_vals is None:
                    continue

                cl  = ohlcv[tkr]["Close"]
                loc = cl.index.get_loc(dt)
                if loc + 1 >= len(cl):
                    continue
                p0, p1 = cl.iloc[loc], cl.iloc[loc + 1]
                if pd.isna(p1) or pd.isna(p0) or p0 == 0:
                    continue

                targets[tkr]   = np.log(p1 / p0)
                row_feats[tkr] = f_vals
            except (KeyError, IndexError):
                continue

        if len(row_feats) < k_lgb * 2:
            continue

        # DataFrame temporaneo per la data corrente
        df_dt = pd.DataFrame(row_feats, index=features).T

        # Applicazione rank CS SOLO sulle feature non-macro (CS_FEATURES in THESIS)
        cs_feats = [f for f in features if f != "vix_zscore"]
        df_dt = apply_cs_rank(df_dt, cs_feats)

        # Target su EXCESS RETURN (Alpha): sottraiamo la media cross-sectional del giorno
        s_target = pd.Series(targets)
        s_excess_target = s_target - s_target.mean()

        # Classificazione sui `n_bins` bin dell'excess return
        s_rank = s_excess_target.rank(pct=True, method="first")
        df_dt["y_rel"] = (np.floor(s_rank * n_bins)).clip(0, n_bins - 1).astype(int)

        df_dt["date"] = dt
        daily_panels.append(df_dt)

    if not daily_panels:
        return pd.DataFrame()

    panel = pd.concat(daily_panels, axis=0).reset_index(drop=True)
    print(f"  Panel completo: {len(panel)} righe | "
          f"{panel['date'].nunique()} date | "
          f"{panel.shape[1]-2} feature")
    return panel


def panel_upto(full_panel: pd.DataFrame, dates_full: pd.Index, t: int) -> pd.DataFrame:
    """Ritorna le righe del panel con date < dates_full[t] (train window)."""
    cutoff = dates_full[t]
    return full_panel[full_panel["date"] < cutoff]


def purged_cv_score(params: dict, panel: pd.DataFrame, features: List[str],
                    n_splits: int = 5, embargo_weeks: int = 4) -> float:
    """Walk-forward CV con purging temporale. Returns mean ICIR over folds."""
    import lightgbm as _lgb
    from scipy.stats import spearmanr as _spearmanr

    udates    = sorted(panel["date"].unique())
    fold_size = max(len(udates) // n_splits, 1)
    ic_scores = []

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end   = val_start + fold_size
        val_dates = udates[val_start:val_end]

        embargo_idx    = max(0, val_start - embargo_weeks)
        embargo_cutoff = udates[embargo_idx]
        train_dates    = [d for d in udates if d < embargo_cutoff]

        if len(train_dates) < 50 or len(val_dates) < 10:
            continue

        tr = panel[panel["date"].isin(train_dates)].sort_values("date")
        va = panel[panel["date"].isin(val_dates)].sort_values("date")

        X_tr = tr[list(features)].values.astype(np.float32)
        y_tr = tr["y_rel"].values.astype(int)
        X_va = va[list(features)].values.astype(np.float32)
        y_va = va["y_rel"].values.astype(int)

        group_tr = tr.groupby("date", sort=False).size().values
        group_va = va.groupby("date", sort=False).size().values

        model = _lgb.LGBMRanker(**params)
        model.fit(
            X_tr, y_tr, group=group_tr,
            eval_set=[(X_va, y_va)], eval_group=[group_va],
            callbacks=[_lgb.early_stopping(100, verbose=False)],
        )

        preds = model.predict(X_va)

        fold_ics = []
        va_reset = va.reset_index(drop=True)
        for _, grp in va_reset.groupby("date"):
            idx = grp.index.tolist()
            if len(idx) < 5:
                continue
            ic_val, _ = _spearmanr(preds[idx], y_va[idx])
            if not np.isnan(ic_val):
                fold_ics.append(ic_val)

        if fold_ics:
            ic_scores.append(np.mean(fold_ics))

    if not ic_scores:
        return -999.0

    arr = np.array(ic_scores)
    return float(arr.mean() / (arr.std() + 1e-10))   # ICIR


def random_search_lgb(panel_is: pd.DataFrame, fixed_params: dict, search_space: dict,
                      features: List[str],
                      n_trials: int = 30, n_cv_splits: int = 5, seed: int = 42) -> dict:
    """Random search con purged CV sul panel IS. Returns {**fixed_params, **best_p}."""
    import random as _random

    _random.seed(seed)
    results = []

    print(f"Random search: {n_trials} trial | {n_cv_splits}-fold purged CV")
    print(f"Panel IS: {len(panel_is)} righe | "
          f"{panel_is['date'].nunique()} date")
    print("=" * 65)

    for trial in range(n_trials):
        sampled = {k: _random.choice(v) for k, v in search_space.items()}
        sampled["num_leaves"] = min(sampled["num_leaves"],
                                    2 ** sampled["max_depth"] - 1)
        full_params = {**fixed_params, **sampled}
        score = purged_cv_score(full_params, panel_is, features, n_splits=n_cv_splits)
        results.append((score, sampled))
        print(f"  [{trial+1:02d}/{n_trials}] ICIR={score:+.4f} | "
              f"depth={sampled['max_depth']} leaves={sampled['num_leaves']} "
              f"\u03bb={sampled['reg_lambda']} min_ch={sampled['min_child_samples']}")

    best_score, best_p = max(results, key=lambda x: x[0])

    best_p["max_depth"]          = min(best_p["max_depth"], 5)
    best_p["min_child_samples"]  = max(best_p["min_child_samples"], 10)
    best_p["num_leaves"]         = min(best_p["num_leaves"],
                                       2 ** best_p["max_depth"] - 1)

    print("\n" + "=" * 65)
    print(f"Best CV ICIR : {best_score:+.4f}")
    print(f"Best params  : {best_p}")
    return {**fixed_params, **best_p}


def plot_learning_curve(evals_result: dict, best_iter: int) -> None:
    """
    Plotta NDCG@K su train e val per ogni iterazione, con marcatore verticale a
    best_iteration_. Original THESIS cell-37 `plot_learning_curve`.
    """
    import matplotlib.pyplot as _plt

    metric_key = [k for k in list(evals_result.get("train", {}).keys())][0]

    train_scores = evals_result["train"][metric_key]
    val_scores   = evals_result["val"][metric_key]
    iterations   = list(range(1, len(train_scores) + 1))

    fig, ax = _plt.subplots(figsize=(13, 5))

    ax.plot(iterations, train_scores,
            color="#3b82f6", lw=2.0, label="Train NDCG@3")
    ax.plot(iterations, val_scores,
            color="#10b981", lw=2.0, label="Val NDCG@3")

    # Marcatore verticale a best_iteration_
    ax.axvline(best_iter, color="#f59e0b", lw=1.8, ls="--",
               label=f"best_iteration = {best_iter}")

    # Punto massimo sulla val
    best_val = val_scores[best_iter - 1]
    ax.scatter([best_iter], [best_val],
               color="#f59e0b", s=80, zorder=5)
    ax.annotate(f"  NDCG={best_val:.4f}",
                xy=(best_iter, best_val),
                fontsize=10, color="#f59e0b",
                va="bottom")

    # Gap train-val (proxy overfitting)
    gap = [t - v for t, v in zip(train_scores, val_scores)]
    ax2 = ax.twinx()
    ax2.fill_between(iterations, gap, 0, alpha=0.08, color="#ef4444")
    ax2.plot(iterations, gap, color="#ef4444", lw=1.0, ls=":", label="Gap train-val")
    ax2.set_ylabel("Gap (train − val)", color="#ef4444", fontsize=10)
    ax2.tick_params(axis="y", colors="#ef4444")
    ax2.set_ylim(0, max(gap) * 4)   # scala il gap in basso per non coprire le curve

    ax.set_xlabel("Iterazione (n° alberi)")
    ax.set_ylabel("NDCG@3")
    ax.set_title("Learning Curve — LambdaRank", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    _plt.tight_layout()
    _plt.savefig("learning_curve.png", bbox_inches="tight", dpi=150)
    _plt.show()

    # Diagnostica testuale
    print(f"  Iterazioni totali  : {len(train_scores)}")
    print(f"  best_iteration_    : {best_iter}")
    print(f"  Train NDCG@best    : {train_scores[best_iter-1]:.4f}")
    print(f"  Val   NDCG@best    : {val_scores[best_iter-1]:.4f}")

    print(f"  Gap @best          : {gap[best_iter-1]:.4f}")
    overfitting = gap[best_iter-1] / (val_scores[best_iter-1] + 1e-10)
    print(f"  Gap/Val ratio      : {overfitting:.3f}  "
          f"{'⚠ overfit' if overfitting > 0.05 else '✓ ok'}")


def tree_split_paths(node: dict, feature_list: List[str]) -> list:
    """
    Recursively collect (parent_feature → child_feature) split pairs from a LightGBM
    tree structure node (original THESIS cell-38 `get_splits`).
    """
    splits = []
    if 'split_feature' in node:
        feat = feature_list[node['split_feature']]
        if 'left_child' in node and 'split_feature' in node['left_child']:
            left_feat = feature_list[node['left_child']['split_feature']]
            splits.append((feat, left_feat))
        if 'right_child' in node and 'split_feature' in node['right_child']:
            right_feat = feature_list[node['right_child']['split_feature']]
            splits.append((feat, right_feat))
        # Ricorsione
        if 'left_child' in node:
            splits.extend(tree_split_paths(node['left_child'], feature_list))
        if 'right_child' in node:
            splits.extend(tree_split_paths(node['right_child'], feature_list))
    return splits


class RunStore:
    """
    In-memory dictionary of saved run series (original THESIS `PNL_STORE` +
    `store_run` / `plot_all`). Persists across cells for the whole kernel session.
    """

    def __init__(self) -> None:
        self.runs: Dict[str, pd.Series] = {}

    def add(self, series: pd.Series, name: "str | None" = None) -> "str | None":
        """Chiede il nome (se non passato) e salva la PnL nel dizionario in memoria."""
        if name is None:
            name = input("Nome per questo run: ").strip()
        if not name:
            print("Nome vuoto — run non salvato.")
            return None
        self.runs[name] = series
        print(f"✓ '{name}' salvato. Run in memoria: {list(self.runs.keys())}")
        return name

    def plot(self, include_spy: bool = True, spy_close: "pd.Series | None" = None,
             initial_capital: float = 10_000) -> None:
        if not self.runs:
            print("PNL_STORE vuoto.")
            return

        import matplotlib.pyplot as _plt
        from matplotlib.ticker import (LogLocator, NullFormatter,
                                      FuncFormatter, PercentFormatter)

        # Palette colori: Arancione e Azzurro in cima alla lista
        colors = [
            "#f97316",  # 1° Run: Arancione
            "#0284c7",  # 2° Run: Azzurro
            "#10b981",  # 3° Run: Verde
            "#a855f7",  # 4° Run: Viola
            "#ef4444",  # 5° Run: Rosso
            "#06b6d4",  # 6° Run: Ciano
            "#84cc16",  # 7° Run: Lime
            "#eab308",  # 8° Run: Giallo
        ]

        # Trova la finestra comune a tutti i run
        start = max(s.index[0] for s in self.runs.values())
        end   = min(s.index[-1] for s in self.runs.values())

        fig, (ax1, ax2) = _plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        if include_spy and spy_close is not None:
            ref_idx = list(self.runs.values())[0].loc[start:end].index
            spy_weekly = spy_close.reindex(ref_idx, method="ffill")
            spy_r = spy_weekly.pct_change().dropna()

            # Partenza da 10.000$ per lo SPY
            cum_spy = initial_capital * (1 + spy_r).cumprod()
            dd_spy  = (cum_spy / cum_spy.cummax() - 1).astype(float)

            ax1.plot(cum_spy.index, cum_spy.values,
                     color="#475569", lw=1.2, ls=":", label="SPY")
            ax2.plot(dd_spy.index, dd_spy.values,
                     color="#475569", lw=1.0, ls=":", label="SPY")

        rows = []
        for i, (name, s) in enumerate(self.runs.items()):
            col  = colors[i % len(colors)]
            s    = s.loc[start:end]                  # taglia alla finestra comune

            # Partenza da 10.000$ per la strategia
            cum  = initial_capital * (1 + s).cumprod()
            dd   = (cum / cum.cummax() - 1).astype(float)
            _m   = perf_stats(s)
            cagr, vol, mdd = _m["CAGR"], _m["AnnVol"], _m["MaxDD"]
            # NOTE: geometric Sharpe (CAGR/vol) intentionally kept — differs from
            # perf_stats' arithmetic Sharpe; do NOT replace with _m["Sharpe"].
            sr   = cagr / (vol + 1e-10)

            ax1.plot(cum.index, cum.values, color=col, lw=2.0,
                     label=f"{name}  ({cagr:+.1%})")
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.12, color=col)
            ax2.plot(dd.index, dd.values, color=col, lw=1.2, label=name)
            rows.append({"Run": name, "CAGR": f"{cagr:+.2%}",
                         "Vol": f"{vol:.2%}", "Sharpe": f"{sr:.3f}",
                         "MaxDD": f"{mdd:.2%}"})

        # ── Formattazione Asse Y (Scala Logaritmica in Dollari) ───────────
        ax1.set_yscale("log")

        # 1. Forza i tick principali anche sui valori intermedi (non solo 10^4, 10^5)
        ax1.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0), numticks=12)
        )

        # 2. Rimuovi le etichette dai minor ticks (evita la notazione scientifica 10^x)
        ax1.yaxis.set_minor_formatter(NullFormatter())

        # 3. Formatta i tick principali in dollari trasparenti ($10,000, $15,000, ecc.)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.set_title(f"PnL Comparison  [{start.date()} → {end.date()}]", fontsize=13)
        ax1.set_ylabel("Portfolio Value ($) [Log Scale]")
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(True, which="both", ls=":", alpha=0.4)

        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.set_title("Drawdown")
        ax2.legend(fontsize=9, loc="lower left")
        ax2.grid(True, which="both", ls=":", alpha=0.4)

        _plt.tight_layout()
        _plt.savefig("pnl_comparison.png", bbox_inches="tight", dpi=300)
        _plt.show()

        print("\n" + "=" * 55)
        print(pd.DataFrame(rows).set_index("Run").to_string())
