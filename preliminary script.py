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
TARGET_VARS = ["opr_ratio_gn", "opr_ratio_ep", "cash_ratio_totasst","totdebt_to_asst","capital_to_asst", "funded_ratio_total"]
EXTERNAL_VARS = ["ln_pop_city", "ln_curgdp", "ln_psnl_incm", "employment", "ln_med_homevalue",
                    "property_rel","intg_rev_rel",
                    "disaster_event", "flood_dmg_tot",
                    "under_18%", "age_65_plus%", "white%", "bachelors%", "disability%", "poverty%"
                    ]
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


def med_ae(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))


def med_smape(y_true, y_pred):
    """Symmetric MedAPE (%); a 0/0 pair contributes 0 error instead of NaN."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.zeros_like(y_true)
    mask = denom != 0
    out[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.median(out)) * 100


# =============================================================== MAIN =====

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
                     "MedAE": round(med_ae(y_true[valid], y_pred[valid]), 2),
                     "sMdAPE_%": round(med_smape(y_true[valid], y_pred[valid]), 2)})
 
# ---- LSTM runs: {full, internal} feature sets x {MTL, STL} ----
for feature_label, feature_cols in FEATURE_SETS.items():
    cities = full_cities if feature_label == "full" else internal_cities
    n_cities = len(cities)
    print(f"\nTraining LSTM models on '{feature_label}' features ({n_cities} cities)...")
 
    df_sub = df[df[ENTITY_COL].isin(cities)].copy()
    pid_to_idx = {pid: i for i, pid in enumerate(cities)}
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
                             "MedAE": round(med_ae(y_true[valid], y_pred[valid]), 2),
                             "sMdAPE_%": round(med_smape(y_true[valid], y_pred[valid]), 2)})
 
results_df = pd.DataFrame(results)
print(f"\nDone. {len(results_df)} rows stored in `results_df`. Run the report cell next.")
 

# ===========================================================  REPORT ===
pd.set_option("display.width", 120)
 
MODEL_ORDER_NOTE = ("Note: ARIMA / *-internal models and *-full models are NOT "
                     "scored on the same set of cities -- see n_cities_used row.")
 
# Fixed so every target's table uses the same column order -- makes the
# tables directly comparable side by side instead of re-sorting per target.
MODEL_ORDER = ["ARIMA", "MTL-LSTM (internal)", "STL-LSTM (internal)",
               "MTL-LSTM (full)", "STL-LSTM (full)"]
 
HYPOTHESES = [
    ("H1", "MTL-LSTM (full) < ARIMA",               "MTL-LSTM (full)", "ARIMA"),
    ("H2", "MTL-LSTM (full) < STL-LSTM (full)",      "MTL-LSTM (full)", "STL-LSTM (full)"),
    ("H3", "MTL-LSTM (full) < MTL-LSTM (internal)",  "MTL-LSTM (full)", "MTL-LSTM (internal)"),
    ("H4", "STL-LSTM (full) < STL-LSTM (internal)",  "STL-LSTM (full)", "STL-LSTM (internal)"),
]
 
overview_rows = []
 
for target in TARGET_VARS:
    sub = results_df[results_df["target_variable"] == target].set_index("model")
    sub = sub[["n_cities_used", "n_test_cities", "MedAE", "sMdAPE_%"]]
    sub = sub.reindex([m for m in MODEL_ORDER if m in sub.index])  # unified column order
 
    best_model = sub["MedAE"].idxmin()
    print(f"\n=== {target} (best: {best_model}) ===")
    print(sub.T.to_string())
 
    overview_row = {"target_variable": target}
    for h_id, label, model_a, model_b in HYPOTHESES:
        if model_a in sub.index and model_b in sub.index:
            supported = bool(sub.loc[model_a, "MedAE"] < sub.loc[model_b, "MedAE"])
            print(f"  {h_id} ({label}): "
                  f"{'SUPPORTED' if supported else 'NOT SUPPORTED'} "
                  f"[{sub.loc[model_a, 'MedAE']:.2f} vs {sub.loc[model_b, 'MedAE']:.2f}]")
            overview_row[h_id] = "Y" if supported else "N"
        else:
            print(f"  {h_id} ({label}): N/A (missing model rows)")
            overview_row[h_id] = "N/A"
    overview_rows.append(overview_row)
 
overview_df = pd.DataFrame(overview_rows).set_index("target_variable")
print("\n=== Hypothesis overview (Y = supported, N = not supported, per target) ===")
print(overview_df.to_string())
print(f"\n{MODEL_ORDER_NOTE}")




"""
Forecasting municipal fiscal variables for 2024: ARIMA baseline vs.
Single-Task (STL) and Multi-Task (MTL) entity-embedded XGBoost, plus
Single-Task (STL) and Multi-Task (MTL) entity-embedded KNN -- each run
with two feature sets:
    "full"     -> 4 fiscal targets + 7... (11) external drivers
    "internal" -> the 4 fiscal targets only, i.e. forecasts based purely on
                  each variable's own autocorrelation, with no external
                  context.

This gives 9 models compared per target: ARIMA, MTL-XGB-full, STL-XGB-full,
MTL-XGB-internal, STL-XGB-internal, MTL-KNN-full, STL-KNN-full,
MTL-KNN-internal, STL-KNN-internal.

WHY THIS SCRIPT LOOKS THE WAY IT DOES -- READ BEFORE CHANGING MODELS
----------------------------------------------------------------------------
The original LSTM script got "entity embedding" and "MTL vs STL" for free,
because a neural net can (a) learn a dense per-city vector end-to-end via
an Embedding layer, and (b) share one backbone's weights across multiple
output heads. Neither XGBoost nor KNN can do either of those things
natively, so this script has to construct both properties explicitly.
That construction is NOT cosmetic -- it changes what "MTL" and "entity
embedded" actually mean for each model family, and the differences matter
for interpreting the results:

1. ENTITY EMBEDDING (both XGBoost and KNN)
   Trees split on raw feature values and KNN measures raw Euclidean
   distance -- neither learns a dense representation of "city" from data.
   The standard, well-evidenced workaround (Guo & Berkhahn, "Entity
   Embeddings of Categorical Variables", arXiv:1604.06737, 2016) is:
     (a) train a small neural net with a learned per-city Embedding layer
         on the *same* forecasting task,
     (b) throw away everything except the trained embedding weight matrix,
     (c) concatenate that fixed per-city vector onto every other model's
         input features.
   Guo & Berkhahn showed this measurably improves KNN, random forest and
   gradient-boosted-tree accuracy on held-out data versus using the raw
   categorical id (their reported KNN error dropped by more than half).
   That is exactly what `pretrain_entity_embeddings()` below does: it
   trains a compact LSTM+embedding network per feature set (same
   architecture family as the original script's `build_lstm_model`, with
   EMBEDDING_DIM unchanged), then extracts ONLY the embedding matrix. The
   embedding is learned once per feature set (using the *training* years
   only, so there is no leakage), then reused as a static input feature by
   every XGBoost and KNN model in this script, MTL and STL alike. The
   embedding is a feature-engineering step shared by all models here, not
   itself part of any individual model's "sharing" mechanism.

2. MTL "HARD PARAMETER SHARING" FOR XGBOOST -- genuinely supported
   XGBoost >= 2.0 has a real analog of hard parameter sharing: pass
   `multi_strategy="multi_output_tree"` and every tree in the ensemble
   grows with a vector leaf (one shared tree structure, one leaf value per
   target). All four targets are fit jointly by the same sequence of
   trees -- the boosting-round equivalent of "gradients from all tasks
   flow into the same shared layer". This is still an EXPERIMENTAL
   XGBoost feature (as of 3.x) with a narrower feature set than the
   default one-model-per-target mode (e.g. per-tree column subsampling
   and some objectives are unsupported), which is disclosed here rather
   than silently relied on. STL-XGBoost is the default XGBoost behavior:
   `multi_strategy="one_output_per_tree"`, i.e. four completely
   independent boosters -- the direct tree analog of the original
   script's four independent STL-LSTM networks.

3. MTL / STL FOR KNN -- NOT a clean analog, and this script says so
   K-nearest-neighbors has no trained weights at all, so there is nothing
   for four tasks to "share" in the neural-network sense. A neighbor set
   is chosen purely from the input feature vector and a distance metric;
   it does not depend on which target column you are about to average.
   Concretely: scikit-learn's KNeighborsRegressor already supports
   multi-output regression natively -- given a (n, 4) target matrix, it
   picks ONE neighbor set per query point and averages all four target
   columns from those same neighbors. That single, shared neighbor set is
   the closest thing KNN has to "hard parameter sharing": one shared
   representation (the feature space + distance metric + k) driving every
   target. This script therefore defines:
     - MTL-KNN: ONE value of k, chosen to jointly minimize *average*
       validation error across all four targets, feeding one
       multi-output KNeighborsRegressor call (one shared neighbor set for
       all targets, mirroring "one shared backbone, several heads").
     - STL-KNN: four SEPARATE KNeighborsRegressor calls, each with its
       OWN k chosen to minimize validation error for that target alone
       (mirroring "four independent networks").
   Because both variants use the exact same feature space and distance
   metric, MTL-KNN and STL-KNN can select the same k for a given target
   and produce numerically identical predictions for it -- that is
   expected, not a bug, and is a direct consequence of KNN having no
   learnable shared parameters to differentiate the two conditions beyond
   this k-selection choice. Treat KNN's MTL/STL split as a much weaker,
   hyperparameter-level analogy to the LSTM's architecture-level split,
   not an equivalent claim.

Everything else -- data loading, the definition of "complete-case"
cities, one-step-ahead windowing, per-city per-variable MinMax scaling
fit only on training years, and the MAE/sMAPE-family metrics -- is
carried over unchanged from the LSTM script so the two scripts' outputs
are directly comparable.

HOW MISSING VALUES ARE HANDLED -- NO IMPUTATION, COMPLETE-CASE CITIES ONLY
----------------------------------------------------------------------------
Same policy as the LSTM script: `eligible_cities()` drops any city with a
single missing value anywhere in the columns a given model needs, across
its whole history. "internal" models need TARGET_VARS complete
(`internal_cities`); "full" models need TARGET_VARS + EXTERNAL_VARS
complete (`full_cities`, a subset of `internal_cities`). ARIMA uses
`internal_cities`. Every results row records `n_cities_used` because the
different feature sets are NOT scored on the same cities.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
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
EXTERNAL_VARS = ["ln_pop_city", "ln_curgdp", "ln_psnl_incm", "employment", "ln_med_homevalue",
                    "property_rel", "intg_rev_rel",
                    "disaster_event", "flood_dmg_tot",
                    "under_18%", "age_65_plus%", "white%", "bachelors%", "disability%", "poverty%"
                    ]
ENTITY_COL, YEAR_COL = "pid", "year"

WINDOW_SIZE = 3           # years of history each model sees before predicting
TRAIN_END_YEAR = 2023     # last year usable as a TRAINING target
FORECAST_YEAR = 2024      # the held-out year to score against
# Same one-step-ahead caveat as the LSTM script: this is not a multi-step
# forecaster. Whatever FORECAST_YEAR is set to, the model's input window is
# the WINDOW_SIZE actual years immediately before it.

# Validation split: used for (a) XGBoost early stopping, and (b) choosing k
# for both KNN variants. The most recent VAL_YEARS training years are held
# out from fitting and used only to pick when-to-stop / which-k.
USE_VALIDATION = True
VAL_YEARS = 1

# --- entity-embedding pretraining net (see docstring point 1) ---
EMBEDDING_DIM, LSTM_UNITS, SHARED_DENSE_UNITS = 4, 16, 16
EMB_EPOCHS, EMB_BATCH_SIZE, EMB_PATIENCE = 200, 32, 15

# --- XGBoost ---
XGB_N_ESTIMATORS = 500
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_PARAMS = dict(tree_method="hist", max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, eval_metric="rmse",
                   random_state=42)

# --- KNN ---
KNN_K_GRID = [3, 5, 7, 10, 15, 20, 30]   # candidate neighbor counts to try
KNN_WEIGHTS = "distance"                  # closer neighbors count more

FEATURE_SETS = {
    "full": TARGET_VARS + EXTERNAL_VARS,
    "internal": TARGET_VARS,
}

# =========================================================== DATA PREP ====
# (unchanged from the LSTM script)

def load_data(path):
    """Read Excel and keep only the needed columns. No filling/imputation
    happens here or anywhere else in this script."""
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
    target_year (n,)."""
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


def flatten_sequences(X_seq):
    """(n, window, n_features) -> (n, window*n_features). XGBoost and KNN
    take flat feature vectors, not sequences, so the window is unrolled
    into columns (year0_var0, year0_var1, ..., year(w-1)_var(k-1))."""
    n = X_seq.shape[0]
    return X_seq.reshape(n, -1)

# ======================================================= CITY SCALING =====
# Unchanged from the LSTM script: one MinMaxScaler per city per role
# (features vs. targets), fit on 2013-2023 only, inverted with that same
# city's parameters before scoring. See the LSTM script's header comment
# for the full rationale.

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
    out = np.zeros_like(y_scaled, dtype=float)
    for pid in np.unique(ent_ids):
        mask = ent_ids == pid
        out[mask] = scalers[pid].inverse_transform(y_scaled[mask])
    return out


def inverse_col(y_scaled_col, ent_ids, scalers, col_index):
    out = np.zeros(len(y_scaled_col), dtype=float)
    for pid in np.unique(ent_ids):
        mask = ent_ids == pid
        scaler = scalers[pid]
        dummy = np.zeros((int(mask.sum()), scaler.n_features_in_))
        dummy[:, col_index] = y_scaled_col[mask]
        out[mask] = scaler.inverse_transform(dummy)[:, col_index]
    return out

# ================================================ ENTITY EMBEDDING NET =====
# Trained purely to obtain a per-city vector for XGBoost/KNN to consume
# (see docstring point 1). Same architecture family as the LSTM script's
# MTL model. Nothing downstream of the Embedding layer is kept.

def build_embedding_net(n_features, window_size, n_cities):
    seq_in = Input(shape=(window_size, n_features), name="sequence_input")
    ent_in = Input(shape=(1,), name="entity_input")

    lstm_out = LSTM(LSTM_UNITS, activation="tanh")(seq_in)
    emb_layer = Embedding(n_cities, EMBEDDING_DIM, name="entity_embedding")
    emb = Flatten()(emb_layer(ent_in))
    hidden = Dense(SHARED_DENSE_UNITS, activation="relu")(Concatenate()([lstm_out, emb]))
    heads = [Dense(1, name=f"out_{t}")(hidden) for t in TARGET_VARS]

    model = Model([seq_in, ent_in], heads)
    model.compile(optimizer="adam", loss="mse")
    return model, emb_layer


def pretrain_entity_embeddings(X_fit, ent_fit, y_fit, X_val, ent_val, y_val,
                                n_features, n_cities, use_val, label):
    """Train the embedding net on TRAINING years only, then return a fixed
    lookup array embeddings[pid_idx] -> (EMBEDDING_DIM,) vector. No
    forecast leaves this function -- it exists only to harvest the
    Embedding layer's weights."""
    print(f"  Pretraining entity embeddings for '{label}' feature set "
          f"({n_cities} cities, dim={EMBEDDING_DIM})...")
    model, emb_layer = build_embedding_net(n_features, X_fit.shape[1], n_cities)
    monitor = "val_loss" if use_val else "loss"
    stopper = EarlyStopping(monitor=monitor, patience=EMB_PATIENCE, restore_best_weights=True)
    fit_kwargs = dict(epochs=EMB_EPOCHS, batch_size=EMB_BATCH_SIZE, verbose=0, callbacks=[stopper])
    if use_val:
        fit_kwargs["validation_data"] = ([X_val, ent_val], [y_val[:, j] for j in range(y_fit.shape[1])])
    model.fit([X_fit, ent_fit], [y_fit[:, j] for j in range(y_fit.shape[1])], **fit_kwargs)
    return emb_layer.get_weights()[0]   # (n_cities, EMBEDDING_DIM)


def attach_embeddings(X_flat, ent_ids, embeddings):
    """Concatenate each row's city embedding vector onto its flattened
    window features."""
    return np.hstack([X_flat, embeddings[ent_ids]])

# ============================================================= MODELS =====

def fit_predict_mtl_xgb(X_fit, y_fit, X_val, y_val, X_te, use_val):
    """One booster, vector-leaf trees (multi_strategy='multi_output_tree'):
    all four targets fit jointly, hard-parameter-sharing analog for trees."""
    model = xgb.XGBRegressor(multi_strategy="multi_output_tree",
                              n_estimators=XGB_N_ESTIMATORS,
                              early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS if use_val else None,
                              **XGB_PARAMS)
    fit_kwargs = dict(verbose=False)
    if use_val:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
    model.fit(X_fit, y_fit, **fit_kwargs)
    return model.predict(X_te)   # (n, 4)


def fit_predict_stl_xgb(X_fit, y_fit, X_val, y_val, X_te, use_val, n_targets):
    """Four independent boosters (default multi_strategy='one_output_per_tree'),
    one per target -- the tree analog of four independent STL networks."""
    preds = np.zeros((X_te.shape[0], n_targets))
    for j in range(n_targets):
        model = xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATORS,
                                  early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS if use_val else None,
                                  **XGB_PARAMS)
        fit_kwargs = dict(verbose=False)
        if use_val:
            fit_kwargs["eval_set"] = [(X_val, y_val[:, j])]
        model.fit(X_fit, y_fit[:, j], **fit_kwargs)
        preds[:, j] = model.predict(X_te)
    return preds


def _knn_medae_for_k(k, X_fit, y_fit_col, X_val, y_val_col):
    knn = KNeighborsRegressor(n_neighbors=min(k, len(X_fit)), weights=KNN_WEIGHTS)
    knn.fit(X_fit, y_fit_col)
    pred = knn.predict(X_val)
    return float(np.median(np.abs(y_val_col - pred)))


def fit_predict_mtl_knn(X_fit, y_fit, X_val, y_val, X_te, use_val, n_targets):
    """ONE k, chosen to minimize *average* validation MedAE across all four
    targets, feeding a single multi-output KNeighborsRegressor call -- one
    shared neighbor set/representation used for every target (see docstring
    point 3 for why this, not independently-learned weights, is what
    'shared' can mean for KNN)."""
    if use_val and len(X_val) > 0:
        scores = []
        for k in KNN_K_GRID:
            knn = KNeighborsRegressor(n_neighbors=min(k, len(X_fit)), weights=KNN_WEIGHTS)
            knn.fit(X_fit, y_fit)
            pred = knn.predict(X_val)
            scores.append(float(np.median(np.abs(y_val - pred))))   # avg over targets, elementwise median
        best_k = KNN_K_GRID[int(np.argmin(scores))]
    else:
        best_k = KNN_K_GRID[len(KNN_K_GRID) // 2]
    X_train_full = np.vstack([X_fit, X_val]) if use_val and len(X_val) > 0 else X_fit
    y_train_full = np.vstack([y_fit, y_val]) if use_val and len(X_val) > 0 else y_fit
    knn = KNeighborsRegressor(n_neighbors=min(best_k, len(X_train_full)), weights=KNN_WEIGHTS)
    knn.fit(X_train_full, y_train_full)
    print(f"    MTL-KNN shared k = {best_k}")
    return knn.predict(X_te)


def fit_predict_stl_knn(X_fit, y_fit, X_val, y_val, X_te, use_val, n_targets):
    """Four independent KNeighborsRegressor calls, each with its OWN k
    chosen on validation for that target alone -- the KNN analog of four
    independently-tuned STL networks."""
    preds = np.zeros((X_te.shape[0], n_targets))
    chosen_ks = []
    for j in range(n_targets):
        if use_val and len(X_val) > 0:
            scores = [_knn_medae_for_k(k, X_fit, y_fit[:, j], X_val, y_val[:, j]) for k in KNN_K_GRID]
            best_k = KNN_K_GRID[int(np.argmin(scores))]
        else:
            best_k = KNN_K_GRID[len(KNN_K_GRID) // 2]
        chosen_ks.append(best_k)
        X_train_full = np.vstack([X_fit, X_val]) if use_val and len(X_val) > 0 else X_fit
        y_train_full = np.concatenate([y_fit[:, j], y_val[:, j]]) if use_val and len(X_val) > 0 else y_fit[:, j]
        knn = KNeighborsRegressor(n_neighbors=min(best_k, len(X_train_full)), weights=KNN_WEIGHTS)
        knn.fit(X_train_full, y_train_full)
        preds[:, j] = knn.predict(X_te)
    print(f"    STL-KNN per-target k = {dict(zip(TARGET_VARS, chosen_ks))}")
    return preds

# ======================================================= ARIMA BASELINE ===
# Unchanged from the LSTM script.

def arima_forecast_all(df, target_var):
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
# Unchanged from the LSTM script.

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.zeros_like(y_true)
    mask = denom != 0
    out[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(out)) * 100


def med_ae(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))


def med_smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.zeros_like(y_true)
    mask = denom != 0
    out[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.median(out)) * 100


# =============================================================== MAIN =====

df = load_data(DATA_PATH)
n_total_cities = df[ENTITY_COL].nunique()

internal_cities = eligible_cities(df, TARGET_VARS)
full_cities = eligible_cities(df, TARGET_VARS + EXTERNAL_VARS)
print(f"{len(internal_cities)}/{n_total_cities} cities have complete target "
      f"data -> used by ARIMA, MTL/STL-*-internal")
print(f"{len(full_cities)}/{n_total_cities} cities have complete target+external "
      f"data -> used by MTL/STL-*-full")

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
                     "MedAE": round(med_ae(y_true[valid], y_pred[valid]), 2),
                     "sMdAPE_%": round(med_smape(y_true[valid], y_pred[valid]), 2)})

# ---- XGBoost + KNN runs: {full, internal} feature sets x {MTL, STL} ----
for feature_label, feature_cols in FEATURE_SETS.items():
    cities = full_cities if feature_label == "full" else internal_cities
    n_cities = len(cities)
    print(f"\nRunning tree/KNN models on '{feature_label}' features ({n_cities} cities)...")

    df_sub = df[df[ENTITY_COL].isin(cities)].copy()
    pid_to_idx = {pid: i for i, pid in enumerate(cities)}
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

    use_val = USE_VALIDATION and VAL_YEARS > 0
    if use_val:
        fit_sel = year_tr <= (TRAIN_END_YEAR - VAL_YEARS)
        val_sel = ~fit_sel
    else:
        fit_sel = np.ones_like(year_tr, dtype=bool)
        val_sel = np.zeros_like(year_tr, dtype=bool)

    X_fit_seq, ent_fit, y_fit = X_tr[fit_sel], ent_tr[fit_sel], y_tr[fit_sel]
    X_val_seq, ent_val, y_val = X_tr[val_sel], ent_tr[val_sel], y_tr[val_sel]
    if use_val and len(y_val) == 0:
        print(f"  [warning] VAL_YEARS={VAL_YEARS} left no validation rows for "
              f"'{feature_label}' -- falling back to no early stopping / mid-grid k.")
        use_val = False

    # --- 1. pretrain entity embeddings on TRAINING years only ---
    embeddings = pretrain_entity_embeddings(
        X_fit_seq, ent_fit, y_fit, X_val_seq, ent_val, y_val,
        n_features, n_cities, use_val, feature_label)

    # --- 2. flatten windows + attach the fixed per-city embedding vector ---
    X_fit = attach_embeddings(flatten_sequences(X_fit_seq), ent_fit, embeddings)
    X_val = attach_embeddings(flatten_sequences(X_val_seq), ent_val, embeddings) if use_val else np.empty((0, X_fit.shape[1]))
    X_te_flat = attach_embeddings(flatten_sequences(X_te), ent_te, embeddings)

    # --- 3. XGBoost ---
    mtl_xgb_pred = fit_predict_mtl_xgb(X_fit, y_fit, X_val, y_val, X_te_flat, use_val)
    mtl_xgb_pred = inverse_full(mtl_xgb_pred, ent_te, y_scalers)
    stl_xgb_pred = fit_predict_stl_xgb(X_fit, y_fit, X_val, y_val, X_te_flat, use_val, n_targets)
    stl_xgb_pred_inv = np.column_stack([
        inverse_col(stl_xgb_pred[:, j], ent_te, y_scalers, j) for j in range(n_targets)
    ])

    # --- 4. KNN ---
    mtl_knn_pred = fit_predict_mtl_knn(X_fit, y_fit, X_val, y_val, X_te_flat, use_val, n_targets)
    mtl_knn_pred = inverse_full(mtl_knn_pred, ent_te, y_scalers)
    stl_knn_pred = fit_predict_stl_knn(X_fit, y_fit, X_val, y_val, X_te_flat, use_val, n_targets)
    stl_knn_pred_inv = np.column_stack([
        inverse_col(stl_knn_pred[:, j], ent_te, y_scalers, j) for j in range(n_targets)
    ])

    for j, target in enumerate(TARGET_VARS):
        y_true = y_te[:, j]
        for model_name, y_pred in [
            (f"MTL-XGB ({feature_label})", mtl_xgb_pred[:, j]),
            (f"STL-XGB ({feature_label})", stl_xgb_pred_inv[:, j]),
            (f"MTL-KNN ({feature_label})", mtl_knn_pred[:, j]),
            (f"STL-KNN ({feature_label})", stl_knn_pred_inv[:, j]),
        ]:
            valid = ~np.isnan(y_pred)
            results.append({"target_variable": target, "model": model_name,
                             "n_cities_used": n_cities,
                             "n_test_cities": int(valid.sum()),
                             "MedAE": round(med_ae(y_true[valid], y_pred[valid]), 2),
                             "sMdAPE_%": round(med_smape(y_true[valid], y_pred[valid]), 2)})

results_df = pd.DataFrame(results)
print(f"\nDone. {len(results_df)} rows stored in `results_df`. Run the report cell next.")


# ===========================================================  REPORT ===
pd.set_option("display.width", 120)

MODEL_ORDER_NOTE = ("Note: ARIMA / *-internal models and *-full models are NOT "
                     "scored on the same set of cities -- see n_cities_used row. "
                     "See the file header for what MTL/STL and 'entity embedded' "
                     "specifically mean for XGBoost vs. KNN -- they are not the "
                     "same guarantee the LSTM script's MTL/STL split gives you.")

MODEL_ORDER = ["ARIMA",
               "MTL-XGB (internal)", "STL-XGB (internal)",
               "MTL-XGB (full)", "STL-XGB (full)",
               "MTL-KNN (internal)", "STL-KNN (internal)",
               "MTL-KNN (full)", "STL-KNN (full)"]

HYPOTHESES = [
    ("H1", "MTL-XGB (full) < ARIMA",              "MTL-XGB (full)", "ARIMA"),
    ("H2", "MTL-XGB (full) < STL-XGB (full)",     "MTL-XGB (full)", "STL-XGB (full)"),
    ("H3", "MTL-XGB (full) < MTL-XGB (internal)", "MTL-XGB (full)", "MTL-XGB (internal)"),
    ("H4", "STL-XGB (full) < STL-XGB (internal)", "STL-XGB (full)", "STL-XGB (internal)"),
    ("H5", "MTL-KNN (full) < ARIMA",              "MTL-KNN (full)", "ARIMA"),
    ("H6", "MTL-KNN (full) < STL-KNN (full)",     "MTL-KNN (full)", "STL-KNN (full)"),
    ("H7", "MTL-KNN (full) < MTL-KNN (internal)", "MTL-KNN (full)", "MTL-KNN (internal)"),
    ("H8", "STL-KNN (full) < STL-KNN (internal)", "STL-KNN (full)", "STL-KNN (internal)"),
    ("H9", "MTL-XGB (full) < MTL-KNN (full)",     "MTL-XGB (full)", "MTL-KNN (full)"),
]

overview_rows = []

for target in TARGET_VARS:
    sub = results_df[results_df["target_variable"] == target].set_index("model")
    sub = sub[["n_cities_used", "n_test_cities", "MedAE", "sMdAPE_%"]]
    sub = sub.reindex([m for m in MODEL_ORDER if m in sub.index])

    best_model = sub["MedAE"].idxmin()
    print(f"\n=== {target} (best: {best_model}) ===")
    print(sub.T.to_string())

    overview_row = {"target_variable": target}
    for h_id, label, model_a, model_b in HYPOTHESES:
        if model_a in sub.index and model_b in sub.index:
            supported = bool(sub.loc[model_a, "MedAE"] < sub.loc[model_b, "MedAE"])
            print(f"  {h_id} ({label}): "
                  f"{'SUPPORTED' if supported else 'NOT SUPPORTED'} "
                  f"[{sub.loc[model_a, 'MedAE']:.2f} vs {sub.loc[model_b, 'MedAE']:.2f}]")
            overview_row[h_id] = "Y" if supported else "N"
        else:
            print(f"  {h_id} ({label}): N/A (missing model rows)")
            overview_row[h_id] = "N/A"
    overview_rows.append(overview_row)

overview_df = pd.DataFrame(overview_rows).set_index("target_variable")
print("\n=== Hypothesis overview (Y = supported, N = not supported, per target) ===")
print(overview_df.to_string())
print(f"\n{MODEL_ORDER_NOTE}")