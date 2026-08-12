"""
generate_synthetic_training_set.py

SYNTHETIC PIPELINE-VALIDATION DATA ONLY. Generates a CSV matching the
trajectory model's 12-feature schema (see TASK-02 Scope Amendment 1) with
labels assigned by explicit rules mirroring the mastery-speed proposal, so
TASK-03's training script can be developed before real pilot data exists.
Every row is marked `synthetic=True` and must never be merged with real
training rows extracted from the database.

Domain randomization (TASK-02 Scope Amendment 2, R-38): with --n-variants > 1,
also emits N additional CSVs, each generated under its own randomly perturbed
(+/-30%) copy of the score-weight / latent-ability / echolalia coefficients,
plus a manifest recording the literal perturbed values per variant. This
replaces SMOTENC-style augmentation on the synthetic side only; the base
--out file (unperturbed, single-config) is always written exactly as before.

Usage:
    python generate_synthetic_training_set.py --children 10 --words-per-child 4 --seed 42 --out data/synthetic_session1.csv
    python generate_synthetic_training_set.py --children 10 --words-per-child 4 --seed 42 --n-variants 5 --out data/synthetic_session1.csv
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

# Base coefficients from the original spec's step 4 label rule, plus the
# latent-ability distribution and echolalia-rate constants from step 3.
# Domain-randomized variants (Amendment 2) perturb each of these by
# Uniform(0.7, 1.3), independently, seeded from --seed + variant index.
BASE_COEFFICIENTS = {
    'score_weight_skill': 0.45,
    'score_weight_phoneme_accuracy': 0.2,
    'score_weight_exposure_deficit': 0.15,
    'score_weight_first_tap_correct': 0.1,
    'score_weight_single_prompt': 0.1,
    'score_weight_echolalia': -0.25,
    'latent_ability_mean': 0.5,
    'latent_ability_std': 0.18,
    'echolalia_base_rate': 0.22,
    'echolalia_slope': 0.18,
}


def clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def perturb_coefficients(seed: int) -> dict:
    """Derive a domain-randomized coefficient set, +/-30% per constant, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    return {key: base_value * rng.uniform(0.7, 1.3) for key, base_value in BASE_COEFFICIENTS.items()}


def generate_row(rng: np.random.Generator, student_id: str, word_id: str, a: float, coeffs: dict) -> dict:
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

    echolalia_p = max(coeffs['echolalia_base_rate'] - coeffs['echolalia_slope'] * s, 0.02)
    echolalia_flag = bool(rng.random() < echolalia_p)
    if echolalia_flag:
        # Short latency IS the echolalia signal per Prizant 1983 grounding.
        response_latency_ms_phase2 = int(rng.uniform(250, 750))

    if phoneme_accuracy > 0.85:
        phoneme_error_class = 'none'
    else:
        phoneme_error_class = str(rng.choice(PHONEME_ERROR_CLASSES, p=PHONEME_ERROR_WEIGHTS))

    score = (
        coeffs['score_weight_skill'] * s
        + coeffs['score_weight_phoneme_accuracy'] * phoneme_accuracy
        + coeffs['score_weight_exposure_deficit'] * (1 - phase1_exposure_ratio)
        + coeffs['score_weight_first_tap_correct'] * first_tap_correct
        + coeffs['score_weight_single_prompt'] * (prompt_count == 1)
        + coeffs['score_weight_echolalia'] * echolalia_flag
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


def generate_dataset(children: int, words_per_child: int, seed: int, coeffs: dict = None) -> pd.DataFrame:
    if coeffs is None:
        coeffs = BASE_COEFFICIENTS
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(children):
        student_id = f'synthetic_child_{i + 1:03d}'
        a = clip(rng.normal(coeffs['latent_ability_mean'], coeffs['latent_ability_std']), 0.05, 0.95)
        for j in range(words_per_child):
            word_id = f'synthetic_word_{j + 1:03d}'
            rows.append(generate_row(rng, student_id, word_id, a, coeffs))
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
    parser.add_argument('--n-variants', type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dataset = generate_dataset(args.children, args.words_per_child, args.seed)
    dataset.to_csv(args.out, index=False)

    print_summary(dataset)
    print(f'Written to {args.out}')

    if args.n_variants > 1:
        base, ext = os.path.splitext(args.out)
        manifest_rows = []
        for i in range(args.n_variants):
            variant_seed = args.seed + i
            coeffs = perturb_coefficients(variant_seed)
            variant_df = generate_dataset(args.children, args.words_per_child, variant_seed, coeffs=coeffs)
            variant_path = f'{base}_variant{i}{ext}'
            variant_df.to_csv(variant_path, index=False)

            print(f'--- Variant {i} (seed={variant_seed}) ---')
            print_summary(variant_df)
            print(f'Written to {variant_path}')

            manifest_row = {'variant_index': i, 'seed': variant_seed, 'output_file': variant_path}
            manifest_row.update(coeffs)
            manifest_rows.append(manifest_row)

        manifest_path = os.path.join(out_dir or '.', 'synthetic_variants_manifest.csv')
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f'Manifest written to {manifest_path}')
