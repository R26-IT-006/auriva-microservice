"""
diagnose_trajectory_data.py

Read-only diagnostic for why build_trajectory_training_set.py returned zero
rows. Prints counts only — never raw student data — at each stage of the
extraction so we can tell whether the gap is:
  (a) no dialogue_word_progress rows in scope at all,
  (b) rows exist but none have resolved to mastered/struggling yet,
  (c) resolved rows exist but lack a matching Phase 2 or Phase 3 baseline
      attempt (the inner-join step in the main script), or
  (d) the --since date filter is excluding everything.

Usage:
    python diagnose_trajectory_data.py [--since YYYY-MM-DD]
"""
import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
TRAJECTORY_CATEGORIES = ('greetings', 'magic_words', 'abilities')


def get_engine():
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing required environment variables: {', '.join(missing)}")
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url, connect_args={'sslmode': 'require'})


def scalar(conn, query, params=None):
    return conn.execute(text(query), params or {}).scalar()


def main(since: str):
    engine = get_engine()
    with engine.connect() as conn:
        print("=== Connection OK ===\n")

        total_progress = scalar(conn, "SELECT COUNT(*) FROM dialogue_word_progress;")
        print(f"(a) Total dialogue_word_progress rows (any category/status): {total_progress}")

        print("\n(a2) dialogue_word_progress status breakdown, joined to allowed categories:")
        status_df = pd.read_sql(text("""
            SELECT wp.status, COUNT(*) AS n
            FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories
            GROUP BY wp.status
            ORDER BY n DESC;
        """), conn, params={'categories': TRAJECTORY_CATEGORIES})
        print(status_df.to_string(index=False) if not status_df.empty else "  (no rows at all for these categories)")

        resolved = scalar(conn, """
            SELECT COUNT(*) FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories AND wp.status IN ('mastered', 'struggling');
        """, {'categories': TRAJECTORY_CATEGORIES})
        print(f"\n(b) Rows with status IN (mastered, struggling): {resolved}")

        phase2_total = scalar(conn, "SELECT COUNT(*) FROM dialogue_word_attempts WHERE phase = 2;")
        phase2_since = scalar(conn, "SELECT COUNT(*) FROM dialogue_word_attempts WHERE phase = 2 AND attempted_at >= :since;", {'since': since})
        print(f"\n(c) Phase 2 attempt rows total: {phase2_total}; on/after --since ({since}): {phase2_since}")

        phase3_total = scalar(conn, """
            SELECT COUNT(*) FROM dialogue_phase3_attempts dp
            JOIN dialogue_words dw ON dw.id = dp.word_id
            WHERE dw.category IN :categories AND dp.scenario_label = 'A';
        """, {'categories': TRAJECTORY_CATEGORIES})
        phase3_since = scalar(conn, """
            SELECT COUNT(*) FROM dialogue_phase3_attempts dp
            JOIN dialogue_words dw ON dw.id = dp.word_id
            WHERE dw.category IN :categories AND dp.scenario_label = 'A' AND dp.attempted_at >= :since;
        """, {'categories': TRAJECTORY_CATEGORIES, 'since': since})
        print(f"    Phase 3 scenario-A rows (allowed categories) total: {phase3_total}; on/after --since: {phase3_since}")

        print("\n(c2) Per-resolved-word detail — what's actually recorded for each of the 11 pairs:")
        detail_df = pd.read_sql(text("""
            SELECT
              wp.student_id, wp.word_id, dw.category, wp.status, wp.total_sessions,
              (SELECT COUNT(*) FROM dialogue_word_attempts a
                 WHERE a.student_id = wp.student_id AND a.word_id = wp.word_id AND a.phase = 2) AS phase2_rows,
              (SELECT COUNT(*) FROM dialogue_phase3_attempts p
                 WHERE p.student_id = wp.student_id AND p.word_id = wp.word_id) AS phase3_rows_any,
              (SELECT COUNT(*) FROM dialogue_phase3_attempts p
                 WHERE p.student_id = wp.student_id AND p.word_id = wp.word_id AND p.scenario_label = 'A') AS phase3_rows_scenario_a,
              (SELECT COUNT(*) FROM dialogue_phase3_attempts p
                 WHERE p.student_id = wp.student_id AND p.word_id = wp.word_id AND p.scenario_label IS NULL) AS phase3_rows_session_summary
            FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories AND wp.status IN ('mastered', 'struggling')
            ORDER BY wp.student_id, wp.word_id;
        """), conn, params={'categories': TRAJECTORY_CATEGORIES})
        print(detail_df.to_string(index=False))

        print("\n(d) Earliest/latest attempted_at on record (sanity check for --since):")
        for table, extra in [("dialogue_word_attempts", "WHERE phase = 2"), ("dialogue_phase3_attempts", "")]:
            row = conn.execute(text(f"SELECT MIN(attempted_at), MAX(attempted_at) FROM {table} {extra};")).fetchone()
            print(f"    {table}: min={row[0]}, max={row[1]}")

    print("\n=== Diagnosis ===")
    if total_progress == 0:
        print("No dialogue_word_progress rows exist at all yet — no real student has attempted a "
              "greetings/magic_words/abilities word in this DB. The pilot likely hasn't started, or "
              "this .env points at a different DB than the one collecting real sessions.")
    elif resolved == 0:
        print("Progress rows exist, but none have resolved to mastered/struggling yet — every real "
              "student is still 'not_started' or 'in_progress'. This is expected early in a pilot: "
              "Rule 1 needs 2 passing sessions on different days before anything can even become "
              "'mastered', and Rule 3 needs 3 consecutive fails before 'struggling'. Nothing to "
              "validate against yet — re-run this after more real sessions accumulate.")
    else:
        print(f"{resolved} resolved row(s) exist. If build_trajectory_training_set.py still returned "
              "0 rows, the gap is likely the inner-join to a Phase 2 or Phase 3 baseline attempt "
              "(check phase2/phase3 counts above) or --since excluding them (check min/max dates above).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--since', default='2020-01-01')
    args = parser.parse_args()
    main(args.since)
