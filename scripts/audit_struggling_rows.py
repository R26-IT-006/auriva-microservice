"""
audit_struggling_rows.py

Standalone diagnostic — NOT part of the training/calibration pipeline.
Checks whether specific (student_id, word_id) pairs currently labeled
'struggling' reached that status suspiciously fast (total_sessions very
low, especially =1), which would be the signature of the consecutive_fail_count
collision bug fixed under TASK-34 — a bug that could push a word to
'struggling' after a single session, in violation of Rule 2's own
"three CONSECUTIVE SESSIONS" definition, specifically for non-verbal
responses (whose speech_score could never clear the old >=2 pass
threshold regardless of correctness).

This script only reads dialogue_word_progress — it writes nothing, changes
nothing, and is safe to run against a live database at any time.

Usage:
    python audit_struggling_rows.py
"""
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']

# The 8 non-verbal 'struggling' rows found in real_trajectory_validation_set.csv,
# plus the 2 verbal 'struggling' rows for contrast (these should NOT show the
# same total_sessions=1 pattern if the hypothesis is correct, since a verbal
# child's speech_score=3 attempt would have genuinely passed Phase 2 and the
# bug's mechanism wouldn't have applied to them the same way).
NON_VERBAL_STRUGGLING_PAIRS = [
    (25, 15), (25, 17), (29, 15), (28, 16),
    (29, 17), (21, 16), (21, 18), (21, 21),
]
VERBAL_STRUGGLING_PAIRS_FOR_CONTRAST = [
    (25, 16), (21, 17),
]

QUERY = """
SELECT
  student_id, word_id, status, total_sessions,
  consecutive_fail_count, session_pass_count,
  mastery_path, verbal_pass_count, non_verbal_pass_count,
  updated_at
FROM dialogue_word_progress
WHERE (student_id, word_id) IN :pairs;
"""


def get_engine():
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing required environment variables: {', '.join(missing)}")
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url, connect_args={'sslmode': 'require'})


def audit(pairs, label):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text(QUERY), conn,
            params={'pairs': tuple(pairs)},
        )

    print(f"\n=== {label} ({len(pairs)} pairs requested) ===")
    if df.empty:
        print("No matching rows found — check student_id/word_id pairs are still current.")
        return df

    print(df.to_string(index=False))

    suspect = df[df['total_sessions'] <= 1]
    print(f"\n{len(suspect)} of {len(df)} row(s) reached 'struggling' with "
          f"total_sessions <= 1 — this is the bug's signature (Rule 2 requires "
          f"THREE CONSECUTIVE SESSIONS; a single-session struggling verdict "
          f"could only happen through the pre-TASK-34 counter collision).")
    if not suspect.empty:
        print("Flagged rows:")
        print(suspect[['student_id', 'word_id', 'total_sessions', 'status']].to_string(index=False))

    return df


if __name__ == '__main__':
    nv_df = audit(NON_VERBAL_STRUGGLING_PAIRS, "Non-verbal 'struggling' rows (main hypothesis)")
    v_df = audit(VERBAL_STRUGGLING_PAIRS_FOR_CONTRAST, "Verbal 'struggling' rows (contrast group)")

    print("\n=== Interpretation guide ===")
    print("If most/all of the non-verbal group show total_sessions <= 1 and the")
    print("verbal contrast group does NOT show the same pattern: strong evidence")
    print("these 8 rows are bug artifacts, not genuine struggling. Consider")
    print("excluding them from training-relevant analysis, or reprocessing them")
    print("with corrected logic before the next Colab run.")
    print("If total_sessions is genuinely >= 3 for these rows: the struggling")
    print("label may be real, and the earlier speech_score=1/prompt_count=1")
    print("pattern is coincidental — re-examine session 2/3 data for these")
    print("children before concluding either way.")
