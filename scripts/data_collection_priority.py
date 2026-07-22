"""
data_collection_priority.py

Read-only report (counts + word/category names, no PII) answering:
  - How many distinct students have contributed any real trajectory data?
  - How many have at least one RESOLVED (mastered/struggling) word usable for
    validation?
  - Which students/words are closest to resolving with one more session, so
    outreach can be prioritized?

Does not touch the database — SELECT only.

Usage:
    python data_collection_priority.py
"""
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
TRAJECTORY_CATEGORIES = ('greetings', 'magic_words', 'abilities')

# Rule 1 (mastery) needs session_pass_count >= 2 on different days.
# Rule 2 (struggling) needs consecutive_fail_count >= 3.
# A row already at one less than the threshold is "one session away" either way.
NEAR_MASTERY_THRESHOLD    = 1  # session_pass_count >= this
NEAR_STRUGGLING_THRESHOLD = 2  # consecutive_fail_count >= this


def get_engine():
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing required environment variables: {', '.join(missing)}")
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url, connect_args={'sslmode': 'require'})


def main():
    engine = get_engine()
    with engine.connect() as conn:
        total_students = conn.execute(text("""
            SELECT COUNT(DISTINCT wp.student_id)
            FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories;
        """), {'categories': TRAJECTORY_CATEGORIES}).scalar()

        resolved_students = conn.execute(text("""
            SELECT COUNT(DISTINCT wp.student_id)
            FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories AND wp.status IN ('mastered', 'struggling');
        """), {'categories': TRAJECTORY_CATEGORIES}).scalar()

        resolved_by_class = pd.read_sql(text("""
            SELECT wp.status, dw.category, COUNT(*) AS n
            FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories AND wp.status IN ('mastered', 'struggling')
            GROUP BY wp.status, dw.category
            ORDER BY wp.status, dw.category;
        """), conn, params={'categories': TRAJECTORY_CATEGORIES})

        priority = pd.read_sql(text("""
            SELECT
              wp.student_id, dw.word, dw.category,
              wp.session_pass_count, wp.consecutive_fail_count, wp.total_sessions,
              wp.last_pass_date,
              GREATEST(wp.session_pass_count, wp.consecutive_fail_count) AS closeness
            FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories AND wp.status = 'in_progress'
              AND (wp.session_pass_count >= :near_mastery
                   OR wp.consecutive_fail_count >= :near_struggling)
            ORDER BY closeness DESC, wp.student_id;
        """), conn, params={
            'categories': TRAJECTORY_CATEGORIES,
            'near_mastery': NEAR_MASTERY_THRESHOLD,
            'near_struggling': NEAR_STRUGGLING_THRESHOLD,
        })

        per_student_summary = pd.read_sql(text("""
            SELECT
              wp.student_id,
              COUNT(*) FILTER (WHERE wp.status = 'in_progress') AS in_progress_words,
              COUNT(*) FILTER (WHERE wp.status = 'in_progress' AND wp.session_pass_count >= :near_mastery) AS near_mastery_words,
              COUNT(*) FILTER (WHERE wp.status = 'in_progress' AND wp.consecutive_fail_count >= :near_struggling) AS near_struggling_words
            FROM dialogue_word_progress wp
            JOIN dialogue_words dw ON dw.id = wp.word_id
            WHERE dw.category IN :categories
            GROUP BY wp.student_id
            HAVING COUNT(*) FILTER (WHERE wp.status = 'in_progress' AND (wp.session_pass_count >= :near_mastery OR wp.consecutive_fail_count >= :near_struggling)) > 0
            ORDER BY (
              COUNT(*) FILTER (WHERE wp.status = 'in_progress' AND wp.session_pass_count >= :near_mastery)
              + COUNT(*) FILTER (WHERE wp.status = 'in_progress' AND wp.consecutive_fail_count >= :near_struggling)
            ) DESC;
        """), conn, params={
            'categories': TRAJECTORY_CATEGORIES,
            'near_mastery': NEAR_MASTERY_THRESHOLD,
            'near_struggling': NEAR_STRUGGLING_THRESHOLD,
        })

    print(f"=== Student coverage (greetings/magic_words/abilities) ===")
    print(f"Distinct students with any progress record: {total_students}")
    print(f"Distinct students with >=1 RESOLVED word (mastered/struggling): {resolved_students}")
    print()
    print("=== Resolved words by label/category (what the validation set draws from) ===")
    print(resolved_by_class.to_string(index=False) if not resolved_by_class.empty else "(none)")
    print()
    print(f"=== Students to prioritize for another session (word one step from resolving: ===")
    print(f"    session_pass_count >= {NEAR_MASTERY_THRESHOLD} [one more pass on a different day -> mastered], or")
    print(f"    consecutive_fail_count >= {NEAR_STRUGGLING_THRESHOLD} [one more fail -> struggling]")
    print(per_student_summary.to_string(index=False) if not per_student_summary.empty else "(none currently close)")
    print()
    print("=== Full detail, one row per near-resolution word ===")
    print(priority.to_string(index=False) if not priority.empty else "(none)")


if __name__ == '__main__':
    main()
