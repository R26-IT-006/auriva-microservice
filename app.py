import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from phoneme_scorer import score_phoneme

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Trajectory model — TASK-04 (Addenda 1, 2, 3)
# ---------------------------------------------------------------------------

# Addendum 3: MODEL_PATH resolved relative to this file, never hardcoded.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "trajectory_model_calibrated.pkl",
)

# Addendum 1: 13-feature list (phase1_applicable added per R-32).
REQUIRED_RAW_FEATURES = [
    "phase1_exposure_ratio",
    "speech_score",
    "phoneme_accuracy",
    "phoneme_error_class",
    "response_latency_ms_phase2",
    "echolalia_flag",
    "response_latency_ms_phase3",
    "first_tap_correct",
    "selection_change_count",
    "prompt_count",
    "difficulty",
    "category",
    "phase1_applicable",
]

trajectory_model = None
trajectory_columns = None
trajectory_calibrator = None    # Addendum 2
trajectory_label_encoder = None  # maps int class indices → 'fast'/'typical'/'struggling'

# TASK-43 — SHAP explainer, built ONCE at import time next to the model load.
# Constructing a TreeExplainer per request would add seconds to every call.
trajectory_explainer = None
# Maps each of the 20 encoded columns back to the raw feature it was produced
# from, so /explain-trajectory can report 13 named inputs instead of dummies.
trajectory_column_groups = None


def _build_column_group_map(columns, raw_features):
    """
    Maps every encoded column to the raw feature it came from.

    `get_dummies` expands `phoneme_error_class` into 6 columns and `category`
    into 3; the remaining 11 columns carry their raw feature's name verbatim.
    Exact matches win over prefix matches, and the longest prefix wins, so a
    raw name that happens to prefix another can never steal its columns.
    """
    mapping = {}
    for col in columns:
        best = None
        for raw in raw_features:
            if col == raw:
                best = raw
                break
            if col.startswith(f"{raw}_") and (best is None or len(raw) > len(best)):
                best = raw
        mapping[col] = best
    return mapping

# Addendum 3: loud startup error when model file is absent — not a crash.
if not os.path.exists(MODEL_PATH):
    print(
        f"[TRAJECTORY] ERROR: model file not found at {MODEL_PATH} — "
        "/predict-trajectory will return 503 for this session. "
        "Place trajectory_model_calibrated.pkl at that path and restart.",
        file=sys.stderr,
    )
    trajectory_model = None
else:
    try:
        # joblib, not raw pickle — the export cell serializes with joblib.dump
        # (joblib.numpy_pickle.NumpyArrayWrapper-wrapped arrays), which plain
        # pickle.load cannot reconstruct.
        _artifact = joblib.load(MODEL_PATH)
        trajectory_model = _artifact["model"]
        trajectory_columns = _artifact["columns"]
        # Addendum 2: load calibrator; warn and fall back if absent.
        # The calibrator is a CalibratedClassifierCV — it takes the same 20-feature
        # matrix as the model (not a probability vector), so call it with df directly.
        if "calibrator" in _artifact:
            trajectory_calibrator = _artifact["calibrator"]
        else:
            print(
                "[TRAJECTORY] WARNING: model artifact missing 'calibrator' key — "
                "falling back to raw model predict_proba (uncalibrated).",
                file=sys.stderr,
            )
            trajectory_calibrator = None
        # label_encoder maps int class indices (0/1/2) → string labels
        # ('fast'/'struggling'/'typical'). The model's own .classes_ are ints,
        # so this is required to satisfy the trajectory-in-{fast,typical,struggling}
        # contract (AC1).
        if "label_encoder" in _artifact:
            trajectory_label_encoder = _artifact["label_encoder"]
        else:
            print(
                "[TRAJECTORY] WARNING: model artifact missing 'label_encoder' key — "
                "trajectory response will be int (0/1/2), not string label.",
                file=sys.stderr,
            )
            trajectory_label_encoder = None
    except Exception as _exc:
        print(
            f"[TRAJECTORY] ERROR: failed to load model artifact: {_exc}",
            file=sys.stderr,
        )
        trajectory_model = None

# TASK-43: build the SHAP explainer once, here, not per request. A failure to
# build it must not take /predict-trajectory down with it — only
# /explain-trajectory degrades (to 503), which the backend already treats as
# "no explanation available" without changing the trajectory it returns.
if trajectory_model is not None:
    try:
        trajectory_explainer = shap.TreeExplainer(trajectory_model)
        trajectory_column_groups = _build_column_group_map(
            trajectory_columns, REQUIRED_RAW_FEATURES
        )
        _unmapped = [c for c, raw in trajectory_column_groups.items() if raw is None]
        if _unmapped:
            # Unmapped columns would silently drop their contribution and break
            # the additivity guarantee (contributions sum to margin − base).
            print(
                "[TRAJECTORY] WARNING: encoded columns with no raw-feature owner "
                f"— their SHAP contributions will be omitted: {_unmapped}",
                file=sys.stderr,
            )
    except Exception as _exc:
        print(
            f"[TRAJECTORY] ERROR: failed to build SHAP explainer: {_exc} — "
            "/explain-trajectory will return 503 for this session.",
            file=sys.stderr,
        )
        trajectory_explainer = None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "trajectory_model_loaded": trajectory_model is not None,
        }
    )


@app.route("/predict-trajectory", methods=["POST"])
def predict_trajectory():
    if trajectory_model is None:
        return jsonify({"error": "trajectory model not loaded"}), 503

    data = request.json or {}

    missing = [f for f in REQUIRED_RAW_FEATURES if f not in data]
    if missing:
        return jsonify({"error": "missing features", "missing": missing}), 400

    features = {k: data[k] for k in REQUIRED_RAW_FEATURES}

    # get_dummies + reindex handles arbitrary category values generically —
    # do NOT modify this block (TASK-04 hard rule).
    df = pd.DataFrame([features])
    df = pd.get_dummies(df)
    df = df.reindex(columns=trajectory_columns, fill_value=0)

    # Addendum 2: CalibratedClassifierCV.predict_proba takes the same feature
    # matrix as the base model — NOT a probability vector. Call it with df directly.
    # If no calibrator, fall back to the raw model's predict_proba.
    if trajectory_calibrator is not None:
        cal_proba = trajectory_calibrator.predict_proba(df)
    else:
        cal_proba = trajectory_model.predict_proba(df)

    class_idx = int(np.argmax(cal_proba[0]))
    # model.classes_ are int indices (0/1/2); decode to string via label_encoder.
    if trajectory_label_encoder is not None:
        trajectory = str(trajectory_label_encoder.inverse_transform([class_idx])[0])
    else:
        trajectory = str(trajectory_model.classes_[class_idx])
    confidence = float(cal_proba[0][class_idx])

    return jsonify({"trajectory": trajectory, "confidence": confidence})


@app.route("/explain-trajectory", methods=["POST"])
def explain_trajectory():
    """
    TASK-43 — SHAP attributions for the trajectory the model just predicted.

    Read-and-explain only: this endpoint reproduces /predict-trajectory's
    prediction path exactly and adds the attribution breakdown. It never
    changes what is predicted.
    """
    if trajectory_model is None:
        return jsonify({"error": "trajectory model not loaded"}), 503
    if trajectory_explainer is None:
        return jsonify({"error": "trajectory explainer not available"}), 503

    data = request.json or {}

    missing = [f for f in REQUIRED_RAW_FEATURES if f not in data]
    if missing:
        return jsonify({"error": "missing features", "missing": missing}), 400

    features = {k: data[k] for k in REQUIRED_RAW_FEATURES}

    # get_dummies + reindex handles arbitrary category values generically —
    # do NOT modify this block (TASK-04 hard rule).
    df = pd.DataFrame([features])
    df = pd.get_dummies(df)
    df = df.reindex(columns=trajectory_columns, fill_value=0)

    # Same prediction path as /predict-trajectory, so the label being explained
    # is byte-identical to the label that endpoint would return.
    if trajectory_calibrator is not None:
        cal_proba = trajectory_calibrator.predict_proba(df)
    else:
        cal_proba = trajectory_model.predict_proba(df)

    class_idx = int(np.argmax(cal_proba[0]))
    if trajectory_label_encoder is not None:
        trajectory = str(trajectory_label_encoder.inverse_transform([class_idx])[0])
    else:
        trajectory = str(trajectory_model.classes_[class_idx])
    confidence = float(cal_proba[0][class_idx])

    # TreeExplainer on a multiclass forest returns per-class values. Current
    # shap lays them out as (n_samples, n_features, n_classes); older versions
    # returned a per-class list, i.e. (n_classes, n_samples, n_features).
    shap_values = np.asarray(trajectory_explainer.shap_values(df))
    expected = np.asarray(trajectory_explainer.expected_value)

    if shap_values.ndim == 3 and shap_values.shape[1] == len(trajectory_columns):
        per_column = shap_values[0, :, class_idx]
    elif shap_values.ndim == 3:
        per_column = shap_values[class_idx][0]
    else:
        # Single-output model — one set of values, no class axis.
        per_column = shap_values[0]

    base_value = float(expected[class_idx] if expected.ndim > 0 else expected)

    # Aggregate the 20 encoded columns back to the 13 raw features: the 6
    # phoneme_error_class dummies and the 3 category dummies each sum into
    # their own raw feature, so the teacher sees named inputs, not dummies.
    contributions = {raw: 0.0 for raw in REQUIRED_RAW_FEATURES}
    for column, value in zip(trajectory_columns, per_column):
        raw = trajectory_column_groups.get(column)
        if raw is None:
            continue
        contributions[raw] += float(value)

    attributions = [
        {
            "feature": raw,
            "value": features[raw],   # raw input, for display
            "contribution": contributions[raw],
        }
        for raw in REQUIRED_RAW_FEATURES
    ]
    attributions.sort(key=lambda a: abs(a["contribution"]), reverse=True)

    return jsonify(
        {
            "trajectory": trajectory,
            "confidence": confidence,
            "base_value": base_value,
            "attributions": attributions,
        }
    )


@app.route('/score-phoneme', methods=['POST'])
def score():
    data = request.json
    result = score_phoneme(
        data.get('transcript', ''),
        data.get('target_word', '')
    )
    return jsonify(result)

if __name__ == '__main__':
    # TASK-26 — debug mode (Werkzeug's interactive debugger + auto-reload)
    # must not be the default for a pilot-facing run; opt in explicitly.
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(port=5001, debug=debug_mode)
