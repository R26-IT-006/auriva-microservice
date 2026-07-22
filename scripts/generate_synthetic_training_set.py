"""
generate_synthetic_training_set.py

SYNTHETIC PIPELINE-VALIDATION DATA ONLY. Generates a CSV matching the
trajectory model's 12-feature schema (see TASK-02 Scope Amendment 1) with
labels assigned by explicit rules mirroring the mastery-speed proposal, so
TASK-03's training script can be developed before real pilot data exists.
Every row is marked `synthetic=True` and must never be merged with real
training rows extracted from the database.

Usage:
    python generate_synthetic_training_set.py --children 10 --words-per-child 4 --seed 42 --out data/synthetic_session1.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

CATEGORIES = ('greetings', 'magic_words', 'abilities')

# th_fronting and v_w_confusion weighted highest — Sinhala L1 transfer taxonomy.
PHONEME_ERROR_CLASSES = [
    'th_fronting', 'v_w_confusion', 'final_consonant_deletion',
    'vowel_substitution', 'other',
]
PHONEME_ERROR_WEIGHTS = [0.3, 0.28, 0.2, 0.14, 0.08]

OUTPUT_COLUMNS = [
    'student_id', 'word_id', 'attempt_number',
    'phase1_exposure_ratio', 'speech_score', 'phoneme_accuracy',
    'phoneme_error_class', 'response_latency_ms_phase2', 'echolalia_flag',
    'response_latency_ms_phase3', 'first_tap_correct',
    'selection_change_count', 'prompt_count', 'difficulty', 'category',
    'label', 'synthetic',
]


def clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def generate_row(rng: np.random.Generator, student_id: str, word_id: str, a: float) -> dict:
    difficulty = int(rng.choice([1, 2, 3]))
    category = str(rng.choice(CATEGORIES))
    s = clip(a - 0.12 * (difficulty - 1) + rng.normal(0, 0.08), 0.0, 1.0)

    phoneme_accuracy = clip(rng.normal(0.35 + 0.6 * s, 0.1), 0.0, 1.0)
    speech_score = 3 if s > 0.72 else 2 if s > 0.5 else 1 if s > 0.28 else 0
    response_latency_ms_phase3 = int(clip(rng.normal(6500 - 4200 * s, 1200), 900, 15000))
    first_tap_correct = bool(rng.random() < clip(0.3 + 0.65 * s, 0.0, 1.0))

    if s > 0.65:
        selection_change_count = 0
    elif s > 0.4:
        selection_change_count = 1
    else:
        selection_change_count = int(rng.choice([1, 2], p=[0.4, 0.6]))

    prompt_count = 1 if s > 0.55 else (2 if s > 0.3 else 3)
    phase1_exposure_ratio = clip(rng.normal(1.15 - 0.45 * s, 0.1), 0.4, 1.0)
    response_latency_ms_phase2 = int(clip(rng.normal(3200 - 1400 * s, 700), 250, 9000))

    echolalia_p = max(0.22 - 0.18 * s, 0.02)
    echolalia_flag = bool(rng.random() < echolalia_p)
    if echolalia_flag:
        # Short latency IS the echolalia signal per Prizant 1983 grounding.
        response_latency_ms_phase2 = int(rng.uniform(250, 750))

    if phoneme_accuracy > 0.85:
        phoneme_error_class = 'none'
    else:
        phoneme_error_class = str(rng.choice(PHONEME_ERROR_CLASSES, p=PHONEME_ERROR_WEIGHTS))

    score = (
        0.45 * s
        + 0.2 * phoneme_accuracy
        + 0.15 * (1 - phase1_exposure_ratio)
        + 0.1 * first_tap_correct
        + 0.1 * (prompt_count == 1)
        - 0.25 * echolalia_flag
    )
    score += rng.normal(0, 0.04)

    if score > 0.62:
        label = 'fast'
    elif score < 0.34:
        label = 'struggling'
    else:
        label = 'typical'

    return {
        'student_id': student_id,
        'word_id': word_id,
        'attempt_number': 1,
        'phase1_exposure_ratio': round(phase1_exposure_ratio, 4),
        'speech_score': speech_score,
        'phoneme_accuracy': round(phoneme_accuracy, 4),
        'phoneme_error_class': phoneme_error_class,
        'response_latency_ms_phase2': response_latency_ms_phase2,
        'echolalia_flag': echolalia_flag,
        'response_latency_ms_phase3': response_latency_ms_phase3,
        'first_tap_correct': first_tap_correct,
        'selection_change_count': selection_change_count,
        'prompt_count': prompt_count,
        'difficulty': difficulty,
        'category': category,
        'label': label,
        'synthetic': True,
    }


def generate_dataset(children: int, words_per_child: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(children):
        student_id = f'synthetic_child_{i + 1:03d}'
        a = clip(rng.normal(0.5, 0.18), 0.05, 0.95)
        for j in range(words_per_child):
            word_id = f'synthetic_word_{j + 1:03d}'
            rows.append(generate_row(rng, student_id, word_id, a))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def print_summary(df: pd.DataFrame) -> None:
    print(f'Total rows: {len(df)}')
    print('Label distribution:')
    print(df['label'].value_counts().to_string())
    print('Category counts:')
    print(df['category'].value_counts().to_string())
    print('Phoneme error class counts:')
    print(df['phoneme_error_class'].value_counts().to_string())
    for cls, count in df['label'].value_counts().items():
        if count < 5:
            print(f'WARNING: label class "{cls}" has only {count} rows (< 5).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--children', type=int, default=10)
    parser.add_argument('--words-per-child', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='data/synthetic_session1.csv')
    args = parser.parse_args()

    dataset = generate_dataset(args.children, args.words_per_child, args.seed)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dataset.to_csv(args.out, index=False)

    print_summary(dataset)
    print(f'Written to {args.out}')
