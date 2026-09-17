"""
Forecasting municipal fiscal variables for 2024: ARIMA baseline vs.
Single-Task (STL) and Multi-Task (MTL) entity-embedded LSTMs, each run with
two feature sets:
    "full"     -> 4 fiscal targets + 7 external drivers
    "internal" -> the 4 fiscal targets only, i.e. forecasts based purely on
                  each variable's own autocorrelation, with no external
                  context. This isolates how much the external drivers are
                  actually contributing.

That gives 5 models compared per target: ARIMA, MTL-full, STL-full,
MTL-internal, STL-internal.

HOW THE MTL/STL MODELS USE EXTERNAL VARIABLES
----------------------------------------------
- Externals are INPUT-only, never forecast. Both MTL and STL only ever have
  four output heads -- one per TARGET_VAR (see build_lstm_model). The 7
  EXTERNAL_VARS never appear as an output; nothing in this script forecasts
  next year's population, GDP, etc. They are only ever consumed as features
  describing the recent past.
- Entity embedding: a city's integer id is mapped through a small learned
  Embedding (EMBEDDING_DIM=4). Concatenating that vector with the LSTM's
  output and passing both through a Dense layer (see build_lstm_model) lets
  the network learn a different effective *slope* per city for how the
  lagged targets/externals map to next year's targets -- not just a
  different constant offset. This is what "entity-embedded" means: one
  shared set of LSTM weights, but city-specific behavior via the embedding.
- MTL vs STL differ only in how many heads sit on top of that shared
  backbone: MTL trains one network with four heads (out_cash_ga, out_una_ga,
  ...) whose gradients all flow back into the same shared_hidden layer
  ("hard parameter sharing" -- a target that's easy to learn can help the
  others). STL trains four completely separate networks (same architecture,
  independent weights), one per target.
- No 2024 external data is used. For window_size=3, predicting a city's
  2024 targets uses that city's *own* 2021, 2022 and 2023 values (targets
  and, for the "full" feature set, externals) as the LSTM input window
  (build_sequences: y at index i+window_size is predicted from features at
  indices [i, i+window_size)). Since the window always ends the year before
  the one being predicted, the 2024 rows in the external-variable columns
  are simply never read for prediction -- this is a genuine one-step-ahead
  forecast, not a fit using same-year externals.

HOW MISSING VALUES ARE HANDLED -- NO IMPUTATION, COMPLETE-CASE CITIES ONLY
----------------------------------------------------------------------------
Earlier drafts filled gaps with forward/back-fill and a cross-sectional
mean. That's gone: nothing in this script invents, interpolates, or
averages a value for a city that doesn't have one. Instead:
- `eligible_cities(df, cols)` returns only the cities that have a complete,
  non-missing value in every column in `cols`, for every year on record.
  A single missing cell anywhere in that city's history drops the WHOLE
  city from any model that needs those columns.
- "internal" models (ARIMA, MTL/STL-internal) only need TARGET_VARS
  complete, so they use `internal_cities`.
- "full" models (MTL/STL-full) need TARGET_VARS + EXTERNAL_VARS complete,
  so they use `full_cities`, a subset of `internal_cities` (a city can be
  clean on targets but still be dropped here for a gap in one external
  driver).
- Because eligibility differs by feature set, the different models are NOT
  all scored on the same cities. `main()` prints how many cities qualify
  for each set, and every row of the results table records `n_cities_used`
  so this is explicit rather than hidden in an average.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================== CONFIG ====
DATA_PATH = "variable.xlsx"
TARGET_VARS = ["cash_ga", "una_ga", "una_ba", "cur_bal_gn"]
EXTERNAL_VARS = ["property_rel", "intg_rev_rel", "pop", "realgdp_pc",
                  "establish", "employment", "personal_incm_pc"]
ENTITY_COL, YEAR_COL = "pid", "year"

WINDOW_SIZE = 3           # years of history the LSTM sees before predicting
TRAIN_END_YEAR = 2023     # last year usable as a TRAINING target
FORECAST_YEAR = 2024      # the held-out year to score against
# Changing the holdout year (e.g. TRAIN_END_YEAR=2022, FORECAST_YEAR=2023 to
# roll the test year back) is a pure config change -- nothing else to touch.
# NOTE -- this is still a ONE-STEP-AHEAD design: whatever FORECAST_YEAR you
# pick, the model's input window is the WINDOW_SIZE actual years immediately
# before it. Setting FORECAST_YEAR more than 1 year past TRAIN_END_YEAR does
# NOT give a true multi-year-ahead forecast -- it would use real, observed
# intervening-year data as input (if present in the file) rather than a
# genuine H-step projection. A real multi-step-ahead horizon needs a
# different (recursive) forecasting setup -- ask if you want that added.

# Validation split for early stopping. Without this, EarlyStopping watches
# TRAINING loss, which never tells you whether the model is overfitting --
# it only tells you when training has stopped improving on data it has
# already memorized. With VAL_YEARS > 0, the most recent VAL_YEARS training
# years are held out from gradient updates and used purely to decide when
# to stop, so "best weights" means best-on-unseen-years, not best-fit.
USE_VALIDATION = True
VAL_YEARS = 1             # e.g. 1 -> the single most recent training year

EMBEDDING_DIM, LSTM_UNITS, SHARED_DENSE_UNITS = 4, 16, 16
EPOCHS, BATCH_SIZE, PATIENCE = 200, 32, 15
# All five knobs above are free to change independently and are the ones
# worth sweeping first: WINDOW_SIZE (more/less history per prediction),
# LSTM_UNITS / SHARED_DENSE_UNITS / EMBEDDING_DIM (model capacity -- bigger
# isn't automatically better with only a few thousand training sequences),
# EPOCHS/PATIENCE (how long training is allowed to keep looking for
# improvement). None of them require touching any code below this point.

FEATURE_SETS = {
    "full": TARGET_VARS + EXTERNAL_VARS,
    "internal": TARGET_VARS,
}

# =========================================================== DATA PREP ====

def load_data(path):
    """Read Excel and keep only the needed columns. No filling/imputation
    happens here or anywhere else in this script -- missing values are
    handled purely by dropping incomplete cities (see eligible_cities)."""
    df = pd.read_excel(path)
    keep = [ENTITY_COL, "city", YEAR_COL] + TARGET_VARS + EXTERNAL_VARS
    return df[keep].sort_values([ENTITY_COL, YEAR_COL]).reset_index(drop=True)


def eligible_cities(df, cols):
    """Cities with NO missing value in any of `cols`, across every year
    they appear. Returns a sorted list of city ids (original, un-recoded)."""
    incomplete = df.loc[df[cols].isnull().any(axis=1), ENTITY_COL].unique()
    return sorted(set(df[ENTITY_COL].unique()) - set(incomplete))


def build_sequences(df, feature_cols, window_size):
    """Slide a window of `window_size` years per city.
    Returns X_seq (n, window, n_features), X_ent (n,), y (n, 4 targets),
    target_year (n,) -- the calendar year each row predicts."""
    X_seq, X_ent, y, target_year = [], [], [], []
    for pid, g in df.groupby(ENTITY_COL):
        g = g.sort_values(YEAR_COL)
        feats = g[feature_cols].values
        targs = g[TARGET_VARS].values
        years = g[YEAR_COL].values
        for i in range(len(g) - window_size):
            X_seq.append(feats[i:i + window_size])
            X_ent.append(pid)
            y.append(targs[i + window_size])
            target_year.append(years[i + window_size])
    return (np.array(X_seq, dtype="float32"), np.array(X_ent, dtype="int32"),
            np.array(y, dtype="float32"), np.array(target_year))

# ======================================================= CITY SCALING =====
# MIN-MAX SCALING -- summary
# - Each variable (every target and every external driver) gets its OWN
#   min/max -- never mixed with any other variable.
# - Each city gets its OWN min/max per variable -- never mixed with any
#   other city's history.
# - Parameters are estimated ONLY from that city's 2013-2023 (training)
#   years, so no 2024 information leaks into scaling.
# - Inputs (X) and targets (y) are scaled with separate scaler objects,
#   even for variables that appear in both roles.
# - A city's actual 2024 value can fall outside [0, 1] after scaling if
#   2024 was a new high/low relative to 2013-2023 -- expected, not a bug.
# - Predictions are inverse-transformed with that SAME city's parameters
#   before computing MAE / sMAPE, so errors are reported in original units.
#
# One MinMaxScaler per city, fit on 2013-2023 only. Same two helpers serve
# X (3D sequences) and y (2D targets) -- `is_sequence` picks the reshape path.

def fit_city_scalers(df, cols):
    train = df[df[YEAR_COL] <= TRAIN_END_YEAR]
    return {pid: MinMaxScaler().fit(g[cols].values) for pid, g in train.groupby(ENTITY_COL)}


def apply_scaler(arr, ent_ids, scalers, is_sequence):
    out = np.zeros_like(arr, dtype=float)
    for pid in np.unique(ent_ids):
        mask = ent_ids == pid
        scaler = scalers[pid]
        if is_sequence:
            n, w, c = arr[mask].shape
            out[mask] = scaler.transform(arr[mask].reshape(-1, c)).reshape(n, w, c)
        else:
            out[mask] = scaler.transform(arr[mask])
    return out


def inverse_full(y_scaled, ent_ids, scalers):
    """Un-scale all 4 target columns at once (MTL predictions)."""
    out = np.zeros_like(y_scaled, dtype=float)
    for pid in np.unique(ent_ids):
        mask = ent_ids == pid
        out[mask] = scalers[pid].inverse_transform(y_scaled[mask])
    return out


def inverse_col(y_scaled_col, ent_ids, scalers, col_index):
    """Un-scale a single target column (STL predictions). MinMaxScaler needs
    all 4 columns to invert, so pad the other 3 with zeros and discard them."""
    out = np.zeros(len(y_scaled_col), dtype=float)
    for pid in np.unique(ent_ids):
        mask = ent_ids == pid
        scaler = scalers[pid]
        dummy = np.zeros((int(mask.sum()), scaler.n_features_in_))
        dummy[:, col_index] = y_scaled_col[mask]
        out[mask] = scaler.inverse_transform(dummy)[:, col_index]
    return out

# ============================================================= MODELS =====

def build_lstm_model(n_features, window_size, n_cities, target_names):
    """Shared backbone: sequence -> LSTM -> concat with city embedding ->
    Dense (this Dense layer is what lets each city get its own effective
    slope, not just an offset). One Dense(1) head per name in target_names:
    pass all 4 TARGET_VARS for the MTL model (hard parameter sharing across
    tasks), or a single target for an STL model."""
    seq_in = Input(shape=(window_size, n_features), name="sequence_input")
    ent_in = Input(shape=(1,), name="entity_input")

    lstm_out = LSTM(LSTM_UNITS, activation="tanh")(seq_in)
    emb = Flatten()(Embedding(n_cities, EMBEDDING_DIM)(ent_in))
    hidden = Dense(SHARED_DENSE_UNITS, activation="relu")(Concatenate()([lstm_out, emb]))

    heads = [Dense(1, name=f"out_{t}")(hidden) for t in target_names]
    outputs = heads if len(heads) > 1 else heads[0]  # STL: unwrap single output

    model = Model([seq_in, ent_in], outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

# ======================================================= ARIMA BASELINE ===

def arima_forecast_all(df, target_var):
    """Per-city ARIMA(1,1,0) on that city's own 2013-2023 history, one step
    ahead. Falls back to a naive "repeat last value" forecast if the fit
    fails. `df` is assumed pre-filtered to complete-data cities, so this
    never sees a missing value; keyed by the original city id."""
    forecasts = {}
    train = df[df[YEAR_COL] <= TRAIN_END_YEAR]
    for pid, g in train.groupby(ENTITY_COL):
        series = g.sort_values(YEAR_COL)[target_var].astype(float).values
        try:
            fitted = ARIMA(series, order=(1, 1, 0)).fit()
            pred = float(np.asarray(fitted.forecast(steps=1))[0])
        except Exception:
            pred = float(series[-1])
        forecasts[pid] = pred
    return forecasts

# ============================================================= METRICS ====

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    """Symmetric MAPE (%); a 0/0 pair contributes 0 error instead of NaN."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.zeros_like(y_true)
    mask = denom != 0
    out[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(out)) * 100

# =============================================================== MAIN =====

def main():
    df = load_data(DATA_PATH)
    n_total_cities = df[ENTITY_COL].nunique()

    internal_cities = eligible_cities(df, TARGET_VARS)
    full_cities = eligible_cities(df, TARGET_VARS + EXTERNAL_VARS)
    print(f"{len(internal_cities)}/{n_total_cities} cities have complete target "
          f"data -> used by ARIMA, MTL-internal, STL-internal")
    print(f"{len(full_cities)}/{n_total_cities} cities have complete target+external "
          f"data -> used by MTL-full, STL-full")

    monitor = "val_loss" if (USE_VALIDATION and VAL_YEARS > 0) else "loss"
    early_stop = EarlyStopping(monitor=monitor, patience=PATIENCE, restore_best_weights=True)
    n_targets = len(TARGET_VARS)
    results = []

    # ---- ARIMA baseline (complete-target cities only) ----
    df_arima = df[df[ENTITY_COL].isin(internal_cities)]
    print(f"\nFitting ARIMA baselines on {len(internal_cities)} cities...")
    arima_by_target = {t: arima_forecast_all(df_arima, t) for t in TARGET_VARS}
    test_rows = df_arima[df_arima[YEAR_COL] == FORECAST_YEAR].set_index(ENTITY_COL)

    for target in TARGET_VARS:
        y_true = test_rows[target].values
        y_pred = np.array([arima_by_target[target].get(pid, np.nan) for pid in test_rows.index])
        valid = ~np.isnan(y_pred)
        results.append({"target_variable": target, "model": "ARIMA",
                         "n_cities_used": len(internal_cities),
                         "n_test_cities": int(valid.sum()),
                         "MAE": round(mae(y_true[valid], y_pred[valid]), 2),
                         "sMAPE_%": round(smape(y_true[valid], y_pred[valid]), 2)})

    # ---- LSTM runs: {full, internal} feature sets x {MTL, STL} ----
    for feature_label, feature_cols in FEATURE_SETS.items():
        cities = full_cities if feature_label == "full" else internal_cities
        n_cities = len(cities)
        print(f"\nTraining LSTM models on '{feature_label}' features ({n_cities} cities)...")

        df_sub = df[df[ENTITY_COL].isin(cities)].copy()
        pid_to_idx = {pid: i for i, pid in enumerate(cities)}  # recoded PER SUBSET
        df_sub[ENTITY_COL] = df_sub[ENTITY_COL].map(pid_to_idx)

        X_seq, X_ent, y, year = build_sequences(df_sub, feature_cols, WINDOW_SIZE)
        train_mask, test_mask = year <= TRAIN_END_YEAR, year == FORECAST_YEAR

        x_scalers = fit_city_scalers(df_sub, feature_cols)
        y_scalers = fit_city_scalers(df_sub, TARGET_VARS)
        X_tr = apply_scaler(X_seq[train_mask], X_ent[train_mask], x_scalers, is_sequence=True)
        X_te = apply_scaler(X_seq[test_mask], X_ent[test_mask], x_scalers, is_sequence=True)
        y_tr = apply_scaler(y[train_mask], X_ent[train_mask], y_scalers, is_sequence=False)
        ent_tr, ent_te, y_te = X_ent[train_mask], X_ent[test_mask], y[test_mask]
        year_tr = year[train_mask]
        n_features = len(feature_cols)

        # Split the training years into "fit" (gets gradient updates) and
        # "val" (held out, only used by EarlyStopping to decide when to
        # stop) when VAL_YEARS > 0. If disabled, everything is "fit" and
        # there's no validation_data -- same behavior as before.
        use_val = USE_VALIDATION and VAL_YEARS > 0
        if use_val:
            fit_sel = year_tr <= (TRAIN_END_YEAR - VAL_YEARS)
            val_sel = ~fit_sel
        else:
            fit_sel = np.ones_like(year_tr, dtype=bool)
            val_sel = np.zeros_like(year_tr, dtype=bool)

        X_fit, ent_fit, y_fit = X_tr[fit_sel], ent_tr[fit_sel], y_tr[fit_sel]
        X_val, ent_val, y_val = X_tr[val_sel], ent_tr[val_sel], y_tr[val_sel]
        if use_val and len(y_val) == 0:
            print(f"  [warning] VAL_YEARS={VAL_YEARS} left no validation rows for "
                  f"'{feature_label}' -- falling back to training-loss early stopping.")
            use_val = False

        # MTL: one model, four heads, trained jointly
        mtl = build_lstm_model(n_features, WINDOW_SIZE, n_cities, TARGET_VARS)
        mtl_fit_kwargs = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])
        if use_val:
            mtl_fit_kwargs["validation_data"] = ([X_val, ent_val], [y_val[:, j] for j in range(n_targets)])
        mtl.fit([X_fit, ent_fit], [y_fit[:, j] for j in range(n_targets)], **mtl_fit_kwargs)
        mtl_pred = inverse_full(np.column_stack(mtl.predict([X_te, ent_te], verbose=0)), ent_te, y_scalers)

        # STL: four independent single-head models
        stl_pred = np.zeros_like(mtl_pred)
        for j, target in enumerate(TARGET_VARS):
            stl = build_lstm_model(n_features, WINDOW_SIZE, n_cities, [target])
            stl_fit_kwargs = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])
            if use_val:
                stl_fit_kwargs["validation_data"] = ([X_val, ent_val], y_val[:, j])
            stl.fit([X_fit, ent_fit], y_fit[:, j], **stl_fit_kwargs)
            pred_s = stl.predict([X_te, ent_te], verbose=0).flatten()
            stl_pred[:, j] = inverse_col(pred_s, ent_te, y_scalers, j)

        for j, target in enumerate(TARGET_VARS):
            y_true = y_te[:, j]
            for model_name, y_pred in [(f"MTL-LSTM ({feature_label})", mtl_pred[:, j]),
                                        (f"STL-LSTM ({feature_label})", stl_pred[:, j])]:
                valid = ~np.isnan(y_pred)
                results.append({"target_variable": target, "model": model_name,
                                 "n_cities_used": n_cities,
                                 "n_test_cities": int(valid.sum()),
                                 "MAE": round(mae(y_true[valid], y_pred[valid]), 2),
                                 "sMAPE_%": round(smape(y_true[valid], y_pred[valid]), 2)})

    # ---- Results ----
    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 120)
    print("\n" + results_df.to_string(index=False))
    results_df.to_csv("forecast_comparison_results.csv", index=False)
    print("\nSaved full results table to forecast_comparison_results.csv")

    avg_mae = results_df.groupby("model")["MAE"].mean().sort_values()
    avg_smape = results_df.groupby("model")["sMAPE_%"].mean().sort_values()
    print("\n--- Average MAE across all 4 targets ---")
    print(avg_mae)
    print("\n--- Average sMAPE across all 4 targets ---")
    print(avg_smape)
    print("\nNote: ARIMA / *-internal models and *-full models are NOT scored "
          "on the same set of cities -- see n_cities_used per row above.")

    print("\nH1 (MTL-LSTM full < ARIMA):",
          "SUPPORTED" if avg_mae["MTL-LSTM (full)"] < avg_mae["ARIMA"] else "NOT SUPPORTED")
    print("H2 (MTL-LSTM full < STL-LSTM full):",
          "SUPPORTED" if avg_mae["MTL-LSTM (full)"] < avg_mae["STL-LSTM (full)"] else "NOT SUPPORTED")
    print("H3 (external drivers help: MTL-LSTM full < MTL-LSTM internal):",
          "SUPPORTED" if avg_mae["MTL-LSTM (full)"] < avg_mae["MTL-LSTM (internal)"] else "NOT SUPPORTED")
    print("H4 (external drivers help: STL-LSTM full < STL-LSTM internal):",
          "SUPPORTED" if avg_mae["STL-LSTM (full)"] < avg_mae["STL-LSTM (internal)"] else "NOT SUPPORTED")


if __name__ == "__main__":
    main()