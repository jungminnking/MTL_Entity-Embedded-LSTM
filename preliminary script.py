"""
================================================================================
 ENTITY-EMBEDDED MTL-LSTM vs STL-LSTM vs ARIMA
 Forecasting city fiscal conditions, 1-year-ahead (2024), trained on 2013-2023
================================================================================

WHAT THIS SCRIPT DOES (big picture, plain language)
-----------------------------------------------------
We have a panel of ~274 cities, each with 12 years of data (2013-2024).
We want to predict, for the year 2024, four fiscal numbers for every city:
    cash_ga, una_ga, una_ba, cur_bal_gn

To do that we build THREE competing forecasting approaches and compare
their errors:

  1. ARIMA        - the classic "look only at one city's own history of one
                     variable" baseline. No sharing of information at all.

  2. STL-LSTM      - a neural network (LSTM) that still predicts only ONE
                     variable at a time, but it is allowed to look at a
                     *city identity* (entity embedding) and at *all cities'*
                     histories when it is trained, because we pool every
                     city into one training set. This is "single-task
                     learning": one full network per target variable.

  3. MTL-LSTM      - the star of the show. ONE neural network predicts all
                     FOUR fiscal variables at the same time, sharing a
                     hidden layer ("hard parameter sharing"). It also uses
                     the city identity as an "entity embedding" that is
                     concatenated into the network (not just added as a
                     constant), so the embedding can change the *slope* of
                     the relationship for each city, not just shift the
                     baseline up or down.

We then measure how wrong each approach is using MAE (average absolute
error, in dollars) and MAPE (average percentage error).

HOW THE DATA IS TURNED INTO LSTM INPUT ("sliding windows")
-----------------------------------------------------------
LSTMs need a *sequence* of past years to look at before they predict the
next year. So for every city we slide a window of WINDOW_SIZE consecutive
years across its 12-year history. Each window's job is to look at years
(t-2, t-1, t) and predict year (t+1). Doing this for every possible
starting point inside 2013-2023 gives us many training examples per city
even though each city only has 12 rows of raw data - this pooling is
exactly the "data sparsity" fix described in the write-up: instead of
274 separate 12-point time series (too short to learn from), we get one
big pooled training set.

The very last window in every city (years 2021, 2022, 2023) is reserved to
predict 2024 - this is our held-out test set, matching the 1-year forecast
horizon.

REQUIRED PACKAGES
------------------
pip install pandas numpy scikit-learn tensorflow statsmodels openpyxl
(tensorflow provides Keras, used for the LSTM networks)
================================================================================
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Flatten, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")  # ARIMA is noisy about convergence warnings on short series

# make results repeatable
np.random.seed(42)
tf.random.set_seed(42)


# ==============================================================================
# 1. CONFIGURATION
#    All the "knobs" for this study live here so you can tweak them in one
#    place without hunting through the code.
# ==============================================================================

DATA_PATH = "variable.xlsx"     # path to your uploaded data file

# The four fiscal numbers we are trying to predict (the "tasks" in MTL)
TARGET_VARS = ["cash_ga", "una_ga", "una_ba", "cur_bal_gn"]

# The environmental / contextual variables (external drivers)
EXTERNAL_VARS = [
    "property_rel", "intg_rev_rel", "pop", "realgdp_pc", "establish",
    "employment", "personal_incm_pc", "flood_event", "flood_damage_tot",
]

ENTITY_COL = "pid"     # the city identifier used for entity embedding
YEAR_COL = "year"

WINDOW_SIZE = 3         # how many past years the LSTM looks at before predicting
FORECAST_YEAR = 2024    # the year we are trying to predict (test set)
TRAIN_END_YEAR = 2023   # last year allowed to appear as a *target* in training

EMBEDDING_DIM = 4        # size of the learned "city vector"
LSTM_UNITS = 16          # size of the shared LSTM hidden state
SHARED_DENSE_UNITS = 16  # size of the shared interactive layer (hard sharing point)

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 15            # early-stopping patience


# ==============================================================================
# 2. LOAD AND CLEAN THE DATA
# ==============================================================================

def load_and_clean_data(path):
    """
    Reads the Excel file and fills in small numbers of missing values.

    Why we impute instead of dropping: a handful of external variables
    (realgdp_pc, establish, employment, personal_incm_pc, flood_event,
    flood_damage_tot) are missing for ~24 rows total, and una_ga/una_ba are
    missing for 11 rows. Dropping those rows would break the sliding-window
    sequences for those cities. Instead we:
        (a) sort by city and year, so "forward fill" moves through time
            correctly within each city,
        (b) forward/backward fill within each city (carries the nearest
            known value across the gap),
        (c) as a last resort, fill any value still missing with that
            column's overall (global) mean - this only matters for a
            handful of cells.
    """
    df = pd.read_excel(path)

    keep_cols = [ENTITY_COL, "city", YEAR_COL] + TARGET_VARS + EXTERNAL_VARS
    df = df[keep_cols].copy()

    # sort so within-city forward/backward fill respects time order
    df = df.sort_values([ENTITY_COL, YEAR_COL]).reset_index(drop=True)

    fill_cols = TARGET_VARS + EXTERNAL_VARS
    df[fill_cols] = (
        df.groupby(ENTITY_COL)[fill_cols]
          .apply(lambda g: g.ffill().bfill())
          .reset_index(drop=True)
    )
    # anything still missing (a city missing a variable for its whole history)
    # gets the global column mean as a last resort
    df[fill_cols] = df[fill_cols].fillna(df[fill_cols].mean())

    return df


# ==============================================================================
# 3. BUILD SLIDING-WINDOW SEQUENCES
#    Turns the flat table into (sequence_of_past_years, city_id, target_next_year)
#    triples that a Keras model can train on.
# ==============================================================================

def build_sequences(df, feature_cols, window_size):
    """
    For every city, slides a window of `window_size` consecutive years
    across its history. Each window predicts the single year right after it.

    Returns:
        X_seq   : array of shape (n_samples, window_size, n_features)
                  the past `window_size` years of [targets + external vars]
        X_ent   : array of shape (n_samples,)
                  the integer-coded city id for each sample (for the
                  embedding layer)
        y       : array of shape (n_samples, n_targets)
                  the four target values in the year being predicted
        target_year : array of shape (n_samples,)
                  the calendar year being predicted - lets us split
                  train (<=2023) vs. test (==2024) later
    """
    X_seq, X_ent, y, target_year = [], [], [], []

    for pid, g in df.groupby(ENTITY_COL):
        g = g.sort_values(YEAR_COL).reset_index(drop=True)
        features = g[feature_cols].values
        targets = g[TARGET_VARS].values
        years = g[YEAR_COL].values

        # slide the window: window covers rows [i, i+window_size),
        # and predicts the row right after the window (i+window_size)
        for i in range(len(g) - window_size):
            X_seq.append(features[i: i + window_size])
            X_ent.append(pid)
            y.append(targets[i + window_size])
            target_year.append(years[i + window_size])

    return (
        np.array(X_seq, dtype="float32"),
        np.array(X_ent, dtype="int32"),
        np.array(y, dtype="float32"),
        np.array(target_year),
    )


# ==============================================================================
# 4. SCALING
#    Neural nets train much better when every input column is roughly on the
#    same numeric scale. We fit the scaler ONLY on training data so the test
#    set (2024) never leaks information into the scaling step.
# ==============================================================================

def fit_scalers(X_seq_train, y_train):
    n_samples, window, n_features = X_seq_train.shape

    x_scaler = StandardScaler()
    # temporarily flatten (samples*window, features) so StandardScaler can
    # compute one mean/std per feature column, then reshape back
    x_scaler.fit(X_seq_train.reshape(-1, n_features))

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    return x_scaler, y_scaler


def apply_x_scaler(X_seq, scaler):
    n_samples, window, n_features = X_seq.shape
    flat = X_seq.reshape(-1, n_features)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(n_samples, window, n_features)


# ==============================================================================
# 5. MODEL BUILDERS
# ==============================================================================

def build_entity_lstm_backbone(n_features, window_size, n_cities):
    """
    Builds the shared "backbone" used by both the MTL and STL networks:
      sequence input -> LSTM -> concatenate with a city embedding -> Dense

    Why concatenate (rather than just adding the embedding as a bias term):
    concatenating the embedding vector and then passing everything through
    a Dense layer lets the network multiply embedding values against the
    LSTM's learned features. That interaction is what allows each city to
    end up with a different effective *slope* for how external/lagged
    variables affect its forecast, not just a different constant offset -
    this is the "interactive" entity-embedding behavior we want.

    Returns the two Input layers and the shared hidden representation,
    so callers can either (a) attach ONE output head to it (STL) or
    (b) attach FOUR output heads to it (MTL, hard parameter sharing).
    """
    seq_input = Input(shape=(window_size, n_features), name="sequence_input")
    entity_input = Input(shape=(1,), name="entity_input")

    # the LSTM reads the window of past years and compresses it into one
    # vector that summarizes the recent trend
    lstm_out = LSTM(LSTM_UNITS, activation="tanh", name="shared_lstm")(seq_input)

    # the embedding turns a city's integer id into a small learned vector -
    # cities that behave similarly will end up with similar vectors
    entity_emb = Embedding(
        input_dim=n_cities, output_dim=EMBEDDING_DIM, name="city_embedding"
    )(entity_input)
    entity_emb = Flatten(name="flatten_embedding")(entity_emb)

    # concatenation + Dense = the "interactive" step described above
    merged = Concatenate(name="concat_lstm_entity")([lstm_out, entity_emb])
    shared_hidden = Dense(
        SHARED_DENSE_UNITS, activation="relu", name="shared_interactive_layer"
    )(merged)

    return seq_input, entity_input, shared_hidden


def build_mtl_model(n_features, window_size, n_cities):
    """
    Multi-task model: ONE shared_hidden layer feeds FOUR separate output
    heads (one Dense(1) neuron per target variable). Because all four
    heads read from the same shared_hidden layer, gradient updates from
    every task flow back into the same shared weights - this is exactly
    "hard parameter sharing".
    """
    seq_input, entity_input, shared_hidden = build_entity_lstm_backbone(
        n_features, window_size, n_cities
    )

    outputs = [
        Dense(1, name=f"out_{t}")(shared_hidden) for t in TARGET_VARS
    ]

    model = Model(inputs=[seq_input, entity_input], outputs=outputs, name="MTL_LSTM")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_stl_model(n_features, window_size, n_cities, target_name):
    """
    Single-task model: identical backbone architecture to the MTL model,
    but only ONE output head. We train four of these (one per target
    variable), each with its own independent set of weights - nothing is
    shared *across* target variables, which is the defining difference
    from MTL. (Note: each STL model is still pooled across all cities and
    still uses the entity embedding, so it's a fair comparison against MTL
    on everything except task-sharing.)
    """
    seq_input, entity_input, shared_hidden = build_entity_lstm_backbone(
        n_features, window_size, n_cities
    )
    output = Dense(1, name=f"out_{target_name}")(shared_hidden)

    model = Model(inputs=[seq_input, entity_input], outputs=output,
                  name=f"STL_LSTM_{target_name}")
    model.compile(optimizer="adam", loss="mse")
    return model


# ==============================================================================
# 6. ARIMA BASELINE
#    Fit one ARIMA model per city per target variable, using only that
#    city's own 2013-2023 history, then forecast one step ahead (2024).
# ==============================================================================

def arima_forecast_all(df, target_var):
    """
    Returns a dict {pid: forecast_value} for a single target variable.

    Practical notes:
      - Each city only has 11 training points (2013-2023), which is very
        short for ARIMA. We use a simple, low-order specification
        (1,1,0) - one autoregressive term plus first-differencing to
        handle a trend - rather than searching a large grid, since a
        big search is unstable on 11 observations and prone to overfitting.
      - If ARIMA still fails to fit for a given city (can happen with a
        flat or highly irregular series), we fall back to a "naive"
        forecast: just repeat the last observed value. This keeps the
        baseline usable for every city instead of crashing.
    """
    forecasts = {}
    train = df[df[YEAR_COL] <= TRAIN_END_YEAR]

    for pid, g in train.groupby(ENTITY_COL):
        g = g.sort_values(YEAR_COL)
        series = g[target_var].astype(float).values

        try:
            model = ARIMA(series, order=(1, 1, 0))
            fitted = model.fit()
            pred = fitted.forecast(steps=1)[0]
        except Exception:
            pred = series[-1]  # naive fallback: repeat last known value

        forecasts[pid] = pred

    return forecasts


# ==============================================================================
# 7. ERROR METRICS
# ==============================================================================

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def safe_mape(y_true, y_pred, min_abs_actual=1000):
    """
    Standard MAPE divides by the actual value, which blows up (or is
    undefined) when the actual is zero or very close to zero - and several
    of our fiscal variables (especially una_ga, una_ba, cur_bal_gn) can be
    negative or near zero. To keep MAPE meaningful, we only compute it
    over rows where |actual| >= min_abs_actual (default $1,000), and we
    report how many rows were excluded so the metric isn't silently
    misleading.
    """
    mask = np.abs(y_true) >= min_abs_actual
    n_excluded = int((~mask).sum())
    if mask.sum() == 0:
        return np.nan, n_excluded
    pct_err = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(pct_err)) * 100, n_excluded


# ==============================================================================
# 8. MAIN PIPELINE
# ==============================================================================

def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)

    # re-code city ids to a clean 0..n_cities-1 range for the Embedding layer
    # (Embedding layers require small consecutive integers, not arbitrary ids)
    unique_pids = sorted(df[ENTITY_COL].unique())
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    df["entity_idx"] = df[ENTITY_COL].map(pid_to_idx)
    n_cities = len(unique_pids)

    feature_cols = TARGET_VARS + EXTERNAL_VARS  # LSTM sees targets (autoregressive) + external drivers

    print("Building sliding-window sequences...")
    df_for_seq = df.copy()
    df_for_seq[ENTITY_COL] = df_for_seq["entity_idx"]  # use re-coded id downstream
    X_seq, X_ent, y, target_year = build_sequences(df_for_seq, feature_cols, WINDOW_SIZE)

    train_mask = target_year <= TRAIN_END_YEAR
    test_mask = target_year == FORECAST_YEAR

    X_seq_train, X_ent_train, y_train = X_seq[train_mask], X_ent[train_mask], y[train_mask]
    X_seq_test, X_ent_test, y_test = X_seq[test_mask], X_ent[test_mask], y[test_mask]

    print(f"  training samples: {len(y_train)}   test (2024) samples: {len(y_test)}")

    print("Scaling features and targets (fit on training data only)...")
    x_scaler, y_scaler = fit_scalers(X_seq_train, y_train)
    X_seq_train_s = apply_x_scaler(X_seq_train, x_scaler)
    X_seq_test_s = apply_x_scaler(X_seq_test, x_scaler)
    y_train_s = y_scaler.transform(y_train)
    # y_test stays UN-scaled - we'll inverse-transform predictions back to
    # dollar units before computing errors, so errors are easy to interpret

    n_features = len(feature_cols)
    early_stop = EarlyStopping(monitor="loss", patience=PATIENCE, restore_best_weights=True)

    # ---------------------------------------------------------------- MTL ---
    print("\nTraining MTL-LSTM (shared hidden layer, all 4 targets jointly)...")
    mtl_model = build_mtl_model(n_features, WINDOW_SIZE, n_cities)
    # Keras wants a separate y array per output head for a multi-output model
    y_train_s_list = [y_train_s[:, i] for i in range(len(TARGET_VARS))]
    mtl_model.fit(
        [X_seq_train_s, X_ent_train], y_train_s_list,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop],
    )
    mtl_pred_s = mtl_model.predict([X_seq_test_s, X_ent_test], verbose=0)
    mtl_pred_s = np.column_stack(mtl_pred_s)  # list of 4 arrays -> (n_samples, 4)
    mtl_pred = y_scaler.inverse_transform(mtl_pred_s)

    # ---------------------------------------------------------------- STL ---
    print("Training STL-LSTM (one independent network per target)...")
    stl_pred = np.zeros_like(mtl_pred)
    for i, target in enumerate(TARGET_VARS):
        print(f"  - {target}")
        stl_model = build_stl_model(n_features, WINDOW_SIZE, n_cities, target)
        stl_model.fit(
            [X_seq_train_s, X_ent_train], y_train_s[:, i],
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop],
        )
        pred_s = stl_model.predict([X_seq_test_s, X_ent_test], verbose=0).flatten()
        # inverse-transform just this one column: build a dummy 4-col array,
        # put the prediction in the right column, inverse-scale, then pull it back out
        dummy = np.zeros((len(pred_s), len(TARGET_VARS)))
        dummy[:, i] = pred_s
        stl_pred[:, i] = y_scaler.inverse_transform(dummy)[:, i]

    # -------------------------------------------------------------- ARIMA ---
    print("Fitting per-city ARIMA baselines (this can take a few minutes)...")
    # ARIMA works off the ORIGINAL (unrecoded) pid and full df, not the sequence arrays
    test_df = df[df[YEAR_COL] == FORECAST_YEAR].sort_values(ENTITY_COL)
    arima_pred = np.zeros((len(test_df), len(TARGET_VARS)))
    for i, target in enumerate(TARGET_VARS):
        forecasts = arima_forecast_all(df, target)
        arima_pred[:, i] = [forecasts.get(pid, np.nan) for pid in test_df[ENTITY_COL]]

    # NOTE: the sliding-window test set (X_seq_test/y_test) and the ARIMA
    # test_df are both built by sorting on entity id and isolating year==2024,
    # so their row order lines up city-for-city. y_test is the common
    # "ground truth" both approaches are scored against.

    # ============================================================ METRICS ===
    print("\nComputing error metrics (MAE in dollars, MAPE in %)...\n")
    results = []
    for i, target in enumerate(TARGET_VARS):
        y_true = y_test[:, i]

        for model_name, y_pred in [
            ("ARIMA", arima_pred[:, i]),
            ("STL-LSTM", stl_pred[:, i]),
            ("MTL-LSTM", mtl_pred[:, i]),
        ]:
            valid = ~np.isnan(y_pred)
            mae_val = mae(y_true[valid], y_pred[valid])
            mape_val, excluded = safe_mape(y_true[valid], y_pred[valid])
            results.append({
                "target_variable": target,
                "model": model_name,
                "MAE": round(mae_val, 2),
                "MAPE_%": round(mape_val, 2) if not np.isnan(mape_val) else "n/a",
                "n_excluded_from_MAPE (|actual|<$1000)": excluded,
                "n_test_cities": int(valid.sum()),
            })

    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 120)
    print(results_df.to_string(index=False))
    results_df.to_csv("forecast_comparison_results.csv", index=False)
    print("\nSaved full results table to forecast_comparison_results.csv")

    # quick H1 / H2 style summary printed to console
    print("\n--- Hypothesis check (average MAE across all 4 targets) ---")
    avg_by_model = results_df.groupby("model")["MAE"].mean().sort_values()
    print(avg_by_model)
    print(
        "\nH1 (MTL-LSTM < ARIMA):",
        "SUPPORTED" if avg_by_model["MTL-LSTM"] < avg_by_model["ARIMA"] else "NOT SUPPORTED",
    )
    print(
        "H2 (MTL-LSTM < STL-LSTM):",
        "SUPPORTED" if avg_by_model["MTL-LSTM"] < avg_by_model["STL-LSTM"] else "NOT SUPPORTED",
    )


if __name__ == "__main__":
    main()
