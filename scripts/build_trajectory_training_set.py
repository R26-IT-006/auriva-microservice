"""
build_trajectory_training_set.py

REAL-DATA extractor for the trajectory model's schema (see TASK-02/TASK-09,
GAP-01 in STATE.md). Produces a CSV matching generate_synthetic_training_set.py's
column layout as closely as the real schema allows, so a model trained on
synthetic_session1.csv can be *validated* (never retrained/merged) against it.

DESIGN DECISIONS MADE HERE — none of these are confirmed by a task file yet.
Flagging each one explicitly rather than silently picking a default:

1. ONE ROW PER (student_id, word_id), using each child's *first* attempt in
   Phase 2 and first pass's first scenario ('A', attempt_number 1) in Phase 3
   as the baseline snapshot. This mirrors the synthetic generator, which also
   always sets attempt_number=1 — the label is a "given early signals, what
   was the eventual trajectory" prediction, not a per-attempt log.
2. LABEL DERIVATION from dialogue_word_progress (no per-attempt real label
   exists — see PROPOSED_MASTERED_SPLIT below). This is a genuine methodology
   choice that should be confirmed, not just used as-is.
3. Rows where status is 'not_started' or 'in_progress' are EXCLUDED — the
   child's trajectory for that word hasn't resolved yet, so there is no
   ground-truth label to validate against.
4. Rows resolved (mastered/struggling) but with NO matching Phase 3
   scenario-A record are ALSO EXCLUDED, not filled with defaults. Confirmed
   against real pilot data (2026-07-21): some real words only ever got a
   session-completion summary row (scenario_label IS NULL) from
   dialogueService.js's recordPhase3Result — no per-scenario
   response_latency_ms/first_tap_correct/selection_change_count/prompt_count
   was ever captured for them. That's a genuine gap in what was recorded,
   not a bug in this script, so those rows can't be validated against the
   full feature schema and are dropped rather than backfilled with fake zeros.
5. `phase1_exposure_ratio` is read from
   dialogue_word_progress.phase1_exposure_ratio_snapshot (TASK-25,
   FSD-EXPOSURE-SNAPSHOT-001) when present — a point-in-time value
   dialogueService.js writes exactly once, at a word's first-ever Phase 1
   gate pass, before recordPhase3Result's post-session reset can touch it.
   Any row whose snapshot predates this migration (or whose word resolved
   before the write path existed) has `phase1_exposure_ratio_snapshot IS
   NULL` in the database, and this script reports it as NULL here too — R-24
   still stands for all pre-migration data; nothing is backfilled or
   computed from the live (post-reset) counters.
   Category 3 (abilities) is a DIFFERENT case, per R-32: it has no Phase 1
   exposure concept at all (`ActionWordAttempt.js` has no `phase1_*` fields),
   so `phase1_exposure_ratio` is structurally N/A there, not merely missing.
   Abilities rows get a fixed sentinel of -1.0 (outside the metric's valid
   0.0-1.0+ range) instead of NULL, kept visibly distinct from genuine
   pre-migration missingness in greetings/magic_words rows — a non-abilities
   row with an unexplained null triggers a visible warning rather than being
   silently folded into the same sentinel bucket.
6. `phoneme_error_class` is NULL in real data whenever RC1 found no error to
   classify (exact/keyword match short-circuits before RC1 ever runs, an
   exact phoneme match, or a real deviation the Sinhala L1 taxonomy doesn't
   cover) or RC1 was unreachable. `generate_synthetic_training_set.py` never
   writes a raw null for this column — it always writes the literal string
   'none' when phoneme_accuracy > 0.85. Left as NULL here, this column would
   one-hot-encode (via get_dummies) to "no category selected" — a code point
   the model never saw during training, distinct from its learned 'none'
   category. Substituted with 'none' below to keep the feature in-distribution,
   mirroring the identical fix already applied in
   auriva-backend/src/services/trajectoryService.js's buildSession1Features().

Usage:
    python build_trajectory_training_set.py [--since YYYY-MM-DD] [--out PATH]
"""
import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']

TRAJECTORY_CATEGORIES = ('greetings', 'magic_words', 'abilities')  # Hard Rule 4

# PROPOSED_MASTERED_SPLIT: a 'mastered' word is called 'fast' if it took the
# minimum possible route to mastery (Rule 1 = 2 passes on different days);
# otherwise 'typical'. This threshold is a default proposal, not a confirmed
# decision — adjust here if the real definition of "fast" should differ.
FAST_MAX_TOTAL_SESSIONS = 2


def get_engine():
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing required environment variables: {', '.join(missing)}")

    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    # Azure Postgres requires SSL — matches auriva-backend/src/config/database.js
    return create_engine(url, connect_args={'sslmode': 'require'})


# --- Phase 1 exposure ratio + eventual outcome, from dialogue_word_progress ---
PROGRESS_QUERY = """
SELECT
  wp.student_id,
  wp.word_id,
  dw.category,
  dw.difficulty,
  wp.status,
  wp.phase1_exposure_count,
  wp.phase1_required_exposures,
  wp.phase1_exposure_ratio_snapshot,
  wp.total_sessions,
  wp.consecutive_fail_count,
  wp.echolalia_rate
FROM dialogue_word_progress wp
JOIN dialogue_words dw ON dw.id = wp.word_id
WHERE dw.category IN :categories
  AND wp.status IN ('mastered', 'struggling');
"""

# --- Phase 2 baseline (first) attempt, greetings/magic_words — dialogue_word_attempts ---
PHASE2_QUERY = """
SELECT DISTINCT ON (student_id, word_id)
  student_id, word_id, speech_score, phoneme_accuracy, phoneme_error_class,
  echolalia_flag, response_latency_ms AS response_latency_ms_phase2, match_type
FROM dialogue_word_attempts
WHERE phase = 2
  AND attempted_at >= :since
ORDER BY student_id, word_id, attempted_at ASC;
"""

# --- Phase 2 baseline (first) attempt, abilities (Category 3) — action_word_attempts.
# CONFIRMED (2026-07-21, via diagnose_trajectory_data.py): Category 3's Phase 2 signals
# live in this separate table under phase2_* columns, NOT in dialogue_word_attempts —
# using the wrong table here was the original bug that made every real abilities row
# fail to join.
#
# CONFIRMED (2026-07-21, via category3Service.js trace): action_word_attempts commonly
# holds TWO rows per (student_id, word_id) for a word's first attempt, not one —
# recordDragToLine creates a row with session_id=null (drag_to_line_result only,
# phase2_* null) BEFORE any session exists; assessPhase2Speech then mints a brand-new
# session_id (since none was passed yet) and, finding no row under that new id, creates
# a SECOND row holding the phase2_* fields instead of updating the first. The
# chronologically-first row is therefore usually the orphaned drag-to-line-only one —
# picking strictly by attempted_at (as this query originally did) silently grabbed that
# empty row instead of the one with real signal. Filtering to
# phase2_speech_score IS NOT NULL picks the row that actually has Phase 2 data,
# regardless of which physical row it ended up in. This is a real backend row-splitting
# bug (see STATE.md) — worth fixing at the source — but that's a bigger, separate
# decision from just extracting validation data correctly here. ---
PHASE2_ABILITIES_QUERY = """
SELECT DISTINCT ON (student_id, word_id)
  student_id, word_id,
  phase2_speech_score AS speech_score,
  phase2_phoneme_accuracy AS phoneme_accuracy,
  phase2_phoneme_error_class AS phoneme_error_class,
  phase2_echolalia_flag AS echolalia_flag,
  phase2_response_latency_ms AS response_latency_ms_phase2,
  phase2_match_type AS match_type
FROM action_word_attempts
WHERE attempted_at >= :since
  AND phase2_speech_score IS NOT NULL
ORDER BY student_id, word_id, attempted_at ASC;
"""

# --- Phase 3 baseline (first pass, scenario 'A') attempt, from dialogue_phase3_attempts ---
PHASE3_QUERY = """
SELECT DISTINCT ON (dp.student_id, dp.word_id)
  dp.student_id, dp.word_id,
  dp.response_latency_ms AS response_latency_ms_phase3,
  dp.first_tap_correct, dp.selection_change_count, dp.prompt_count
FROM dialogue_phase3_attempts dp
JOIN dialogue_words dw ON dw.id = dp.word_id
WHERE dw.category IN :categories
  AND dp.scenario_label = 'A'
  AND dp.attempted_at >= :since
ORDER BY dp.student_id, dp.word_id, dp.attempted_at ASC;
"""


def derive_label(row) -> str:
    if row['status'] == 'struggling':
        return 'struggling'
    # status == 'mastered' here (progress query already filters to
    # mastered/struggling). is_fast = mastered in the minimum possible
    # number of sessions (Rule 1 = 2 passes). No separate "consecutive
    # fails" check is needed: session_pass_count can never exceed
    # total_sessions, and mastery requires session_pass_count >= 2, so
    # total_sessions <= 2 already forces both sessions to have been
    # passes on its own — confirmed redundant 2026-08-06, not merely
    # broken by TASK-34's fix.
    is_fast = row['total_sessions'] <= FAST_MAX_TOTAL_SESSIONS
    return 'fast' if is_fast else 'typical'


def build_dataset(since: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        progress = pd.read_sql(text(PROGRESS_QUERY), conn, params={'categories': TRAJECTORY_CATEGORIES})
        phase2_standard = pd.read_sql(text(PHASE2_QUERY), conn, params={'since': since})
        phase2_abilities = pd.read_sql(text(PHASE2_ABILITIES_QUERY), conn, params={'since': since})
        phase3 = pd.read_sql(text(PHASE3_QUERY), conn, params={'categories': TRAJECTORY_CATEGORIES, 'since': since})

    if progress.empty:
        print('No resolved (mastered/struggling) words found — check --since and pilot data availability.')
        return progress

    print(f'[trace] resolved progress rows (allowed categories): {len(progress)}')
    print(f'[trace] phase2 baseline rows — dialogue_word_attempts (greetings/magic_words): {len(phase2_standard)}')
    print(f'[trace] phase2 baseline rows — action_word_attempts (abilities): {len(phase2_abilities)}')
    print(f'[trace] phase3 baseline rows (allowed categories, scenario A): {len(phase3)}')

    # Phase 2 baseline source depends on category: abilities (Category 3) lives in
    # action_word_attempts, everything else in dialogue_word_attempts.
    non_abilities = progress[progress['category'] != 'abilities']
    abilities = progress[progress['category'] == 'abilities']
    df_non_abilities = non_abilities.merge(phase2_standard, on=['student_id', 'word_id'], how='inner')
    df_abilities = abilities.merge(phase2_abilities, on=['student_id', 'word_id'], how='inner')
    df = pd.concat([df_non_abilities, df_abilities], ignore_index=True)

    progress_pairs = set(zip(progress['student_id'], progress['word_id']))
    after_phase2_pairs = set(zip(df['student_id'], df['word_id']))
    missing_phase2 = progress_pairs - after_phase2_pairs
    print(f'[trace] rows after merging category-appropriate Phase 2 baseline: {len(df)}')
    if missing_phase2:
        print(f'[trace] resolved pairs with NO Phase 2 baseline in either table ({len(missing_phase2)}): {sorted(missing_phase2)}')

    before_phase3 = len(df)
    df = df.merge(phase3, on=['student_id', 'word_id'], how='inner')
    dropped_phase3 = before_phase3 - len(df)
    print(f'[trace] rows after merging Phase 3 scenario-A baseline: {len(df)}')
    if dropped_phase3:
        print(f'[trace] {dropped_phase3} resolved row(s) dropped here — mastered/struggling, but only a')
        print( '        session-summary Phase 3 record exists (scenario_label IS NULL), so no real')
        print( '        per-scenario telemetry was ever captured. Genuine data gap, not backfilled with defaults.')

    # See docstring item 5: real baseline when a post-migration snapshot exists,
    # NULL otherwise (pre-migration rows, or words resolved before the snapshot
    # write path existed) — never computed from the live post-reset counters.
    df['phase1_exposure_ratio'] = df['phase1_exposure_ratio_snapshot']

    # R-32: abilities has no Phase 1 exposure concept at all (structurally N/A,
    # not missing) — fixed sentinel, kept distinct from genuine pre-migration
    # nulls in greetings/magic_words (see docstring item 5).
    df.loc[df['category'] == 'abilities', 'phase1_exposure_ratio'] = -1.0
    still_null = df[(df['category'] != 'abilities') & (df['phase1_exposure_ratio'].isna())]
    if not still_null.empty:
        print(f'WARNING: {len(still_null)} non-abilities row(s) have genuinely '
              f'missing phase1_exposure_ratio (pre-migration) — see R-24/DEC-06.')

    # See docstring item 6: NULL means "no error to classify" (or RC1 unreachable),
    # never merely missing in a way that should be dropped/warned — the synthetic
    # generator's own 'none' convention is the correct in-distribution substitute.
    df['phoneme_error_class'] = df['phoneme_error_class'].fillna('none')

    # verbal_path: session-1 Phase 2 baseline attempt's own match_type (same row
    # attempt_number=1 already selects) — True unless that attempt was the
    # non-verbal fallback. Additive feature, FSD-MASTERY-PATH-001.
    df['verbal_path'] = df['match_type'] != 'non_verbal'

    df['attempt_number'] = 1  # baseline snapshot, matches the synthetic schema's constant column
    df['label'] = df.apply(derive_label, axis=1)
    df['synthetic'] = False

    return df[OUTPUT_COLUMNS]


OUTPUT_COLUMNS = [
    'student_id', 'word_id', 'attempt_number', 'phase1_exposure_ratio',
    'speech_score', 'phoneme_accuracy', 'phoneme_error_class',
    'response_latency_ms_phase2', 'echolalia_flag', 'response_latency_ms_phase3',
    'first_tap_correct', 'selection_change_count', 'prompt_count', 'difficulty',
    'category', 'verbal_path', 'label', 'synthetic',
]


def print_summary(df: pd.DataFrame) -> None:
    print(f'Total rows: {len(df)}')
    print('Label distribution:')
    print(df['label'].value_counts().to_string())
    print('Category counts:')
    print(df['category'].value_counts().to_string())
    for cls, count in df['label'].value_counts().items():
        if count < 5:
            print(f'WARNING: label class "{cls}" has only {count} row(s) — too thin to validate against alone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--since', default='2020-01-01',
                        help='Only include attempts on/after this date (YYYY-MM-DD)')
    parser.add_argument('--out', default='data/real_trajectory_validation_set.csv')
    args = parser.parse_args()

    dataset = build_dataset(args.since)
    if not dataset.empty:
        dataset.to_csv(args.out, index=False)
        print_summary(dataset)
        print(f'Written to {args.out}')
        print('Reminder: this is a VALIDATION set. Never merge it into synthetic_session1.csv or retrain on it directly.')
