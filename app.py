import os
import sys
import joblib
import numpy as np
import pandas as pd
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
