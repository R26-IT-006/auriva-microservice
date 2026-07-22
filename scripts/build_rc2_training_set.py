"""
build_rc2_training_set.py

Builds the RC2 (pragmatic comprehension) Random Forest training dataset
from dialogue_phase3_attempts. See FSD-RC2-TRAINING-SET-001 for the
attempt_number and cross_scenario_consistent design decisions this
script implements.

Usage:
    python build_rc2_training_set.py [--since YYYY-MM-DD] [--out PATH]
"""
import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']

PRAGMATIC_CATEGORIES = ('greetings', 'magic_words', 'abilities')


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


EXTRACTION_QUERY = """
SELECT
  dp.id AS attempt_id,
  dp.student_id,
  dp.word_id,
  dw.category,
  dp.session_id,
  dp.scenario_label,
  dp.phase3_correct,
  dp.response_latency_ms,
  dp.first_tap_correct,
  dp.selection_change_count,
  dp.prompt_count,
  dp.attempted_at
FROM dialogue_phase3_attempts dp
JOIN dialogue_words dw ON dw.id = dp.word_id
WHERE dw.category IN :categories
  AND dp.attempted_at >= :since
ORDER BY dp.student_id, dp.word_id, dp.attempted_at;
"""


def extract_raw_rows(engine, since):
    with engine.connect() as conn:
        df = pd.read_sql(
            text(EXTRACTION_QUERY), conn,
            params={'categories': PRAGMATIC_CATEGORIES, 'since': since},
        )
    return df


def compute_attempt_number(df):
    """
    A pass through a word always starts with scenario_label == 'A' —
    true for Magic Words, Greetings, and Category 3 alike (confirmed in
    Phase3ContextualScreen.js / GreetingPhase3ContextualScreen.js /
    Cat3Phase3Screen.js). attempt_number is a running count of 'A' rows
    seen so far for this (student_id, word_id), ordered by attempted_at —
    never session_id, which is optional and often null.
    """
    df = df.sort_values(['student_id', 'word_id', 'attempted_at']).copy()
    is_a = (df['scenario_label'] == 'A').astype(int)
    df['attempt_number'] = is_a.groupby([df['student_id'], df['word_id']]).cumsum()
    return df


def compute_cross_scenario_consistent(df):
    """
    Ground-truth label for RC2, computed per (student_id, word_id,
    attempt_number) pass — never across passes. See FSD-RC2-TRAINING-SET-001
    Section 3 for the full rationale.

    True  -> an A-correct row AND a B-correct row both exist in this pass.
    False -> both an A row and a B row exist, but not both correct.
    NaN   -> no B row exists in this pass (most commonly Category 3,
             correct on attempt 1 -> no attempt 2). NOT the same as False.
             Rows with NaN here must be EXCLUDED from training on this
             label, not filled with 0 or 1.
    """
    def label_group(g):
        a_rows = g[g['scenario_label'] == 'A']
        b_rows = g[g['scenario_label'] == 'B']
        if b_rows.empty:
            return pd.NA
        return bool(a_rows['phase3_correct'].any() and b_rows['phase3_correct'].any())

    labels = (
        df.groupby(['student_id', 'word_id', 'attempt_number'])
        .apply(label_group)
        .rename('cross_scenario_consistent')
        .reset_index()
    )
    return df.merge(labels, on=['student_id', 'word_id', 'attempt_number'], how='left')


OUTPUT_COLUMNS = [
    'attempt_id', 'student_id', 'word_id', 'category', 'session_id',
    'attempt_number', 'scenario_label', 'phase3_correct',
    'response_latency_ms', 'first_tap_correct', 'selection_change_count',
    'prompt_count', 'cross_scenario_consistent', 'attempted_at',
]


def build_dataset(since):
    engine = get_engine()
    df = extract_raw_rows(engine, since)
    if df.empty:
        print('No rows returned — check the category filter and --since date.')
        return df

    df = compute_attempt_number(df)
    df = compute_cross_scenario_consistent(df)
    return df[OUTPUT_COLUMNS]


def print_summary(df):
    print(f'Total rows: {len(df)}')
    print(df['category'].value_counts().to_string())
    na_count = df['cross_scenario_consistent'].isna().sum()
    print(f'cross_scenario_consistent N/A (no second scenario in pass): '
          f'{na_count} ({na_count / len(df):.1%})')
    print(f"response_latency_ms null: {df['response_latency_ms'].isna().sum()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--since', default='2020-01-01',
                        help='Only include attempts on/after this date (YYYY-MM-DD)')
    parser.add_argument('--out', default='rc2_training_data.csv')
    args = parser.parse_args()

    dataset = build_dataset(args.since)
    if not dataset.empty:
        dataset.to_csv(args.out, index=False)
        print_summary(dataset)
        print(f'Written to {args.out}')
