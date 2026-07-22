"""
AURIVA — Phoneme-Aware Speech Assessment
RC1: Phoneme scorer with Sinhala L1 transfer error taxonomy

Replaces fuzzy string matching as the primary Phase 2 speech scorer.
Converts transcript and target word to CMU phoneme sequences,
aligns them, classifies Sinhala L1 transfer errors, and returns
a structured score (0-3) with error class and phoneme accuracy.
"""

import pronouncing
import json
from difflib import SequenceMatcher


# ── Sinhala L1 Transfer Error Taxonomy ──────────────────────────────────────
# Documented phonological transfer errors from Sinhala L1 → English L2
# Sources: Kandiah (1981), Parakrama (1995), Sri Lankan English phonology lit.
#
# Format: "EXPECTED_PHONEME:PRODUCED_PHONEME" -> error classification
#
SINHALA_L1_TAXONOMY = {
    # TH-fronting: dental fricative /θ/ does not exist in Sinhala
    # Child substitutes /t/ or /d/ for /TH/
    "TH:T":  {"error_class": "th_fronting",   "score_impact": 2, "description": "TH→T substitution (Sinhala has no dental fricative)"},
    "TH:D":  {"error_class": "th_stopping",   "score_impact": 2, "description": "TH→D substitution (voiced stopping of dental fricative)"},
    "TH:F":  {"error_class": "th_fronting",   "score_impact": 2, "description": "TH→F substitution (labiodental substitution)"},
    "TH:S":  {"error_class": "th_fronting",   "score_impact": 2, "description": "TH→S substitution (sibilant substitution)"},

    # DH-stopping: voiced dental fricative /ð/ → /d/
    "DH:D":  {"error_class": "dh_stopping",   "score_impact": 2, "description": "DH→D substitution (voiced dental fricative stopping)"},
    "DH:Z":  {"error_class": "dh_stopping",   "score_impact": 2, "description": "DH→Z substitution"},

    # Final consonant cluster reduction: Sinhala is a syllable-timed language
    # Final consonant clusters are commonly reduced or deleted
    "T:":    {"error_class": "final_consonant_deletion", "score_impact": 2, "description": "Final /t/ deleted (cluster reduction)"},
    "K:":    {"error_class": "final_consonant_deletion", "score_impact": 2, "description": "Final /k/ deleted"},
    "D:":    {"error_class": "final_consonant_deletion", "score_impact": 2, "description": "Final /d/ deleted"},
    "Z:":    {"error_class": "final_consonant_deletion", "score_impact": 2, "description": "Final /z/ deleted"},
    "N:":    {"error_class": "final_consonant_deletion", "score_impact": 2, "description": "Final /n/ deleted"},

    # W-V confusion: Sinhala has /w/ but not the English /v/
    # Both directions occur in transfer
    "V:W":   {"error_class": "v_w_confusion",  "score_impact": 2, "description": "V→W substitution (Sinhala lacks /v/)"},
    "W:V":   {"error_class": "v_w_confusion",  "score_impact": 2, "description": "W→V substitution"},

    # Vowel reduction: unstressed vowels commonly produced as full vowels in Sinhala
    "AH:AE": {"error_class": "vowel_substitution", "score_impact": 2, "description": "Schwa→/æ/ (Sinhala syllable timing affects vowel reduction)"},
    "IH:IY": {"error_class": "vowel_substitution", "score_impact": 2, "description": "/ɪ/→/iː/ (short vowel lengthening)"},
    "UH:UW": {"error_class": "vowel_substitution", "score_impact": 2, "description": "/ʊ/→/uː/ (short vowel lengthening)"},
    "EH:EY": {"error_class": "vowel_substitution", "score_impact": 2, "description": "/ɛ/→/eɪ/ (vowel substitution)"},

    # R-colouring: Sinhala /r/ is a trill, English /r/ is approximant
    "R:":    {"error_class": "r_deletion",     "score_impact": 2, "description": "Rhotic deleted (common in non-rhotic transfer)"},

    # Initial consonant cluster reduction
    "S:":    {"error_class": "cluster_reduction", "score_impact": 1, "description": "Initial /s/ deleted in cluster"},
    "P:B":   {"error_class": "voicing_error",  "score_impact": 2, "description": "P→B voicing error"},
    "T:D":   {"error_class": "voicing_error",  "score_impact": 2, "description": "T→D voicing error"},
    "K:G":   {"error_class": "voicing_error",  "score_impact": 2, "description": "K→G voicing error"},
}


# ── Multi-word keyword trigger lists (mirrors backend dialogueSeed.js) ──────
# For phrases like "thank you", "I'm sorry" etc. we need to handle
# multi-word targets that the CMU dict may not have as a single entry.
MULTI_WORD_SPLITS = {
    "thank you":      ["thank", "you"],
    "i'm sorry":      ["i'm", "sorry"],
    "you're welcome": ["you're", "welcome"],
    "please":         ["please"],
    "please excuse me": ["please", "excuse", "me"],
    "good morning":   ["good", "morning"],
    "good afternoon": ["good", "afternoon"],
    "good evening":   ["good", "evening"],
    "goodbye":        ["goodbye"],
    "hello":          ["hello"],
    "i can":          ["i", "can"],
    "yes i can":      ["yes", "i", "can"],
    "no i cant":      ["no", "i", "cant"],
    "monday":         ["monday"],
    "tuesday":        ["tuesday"],
    "wednesday":      ["wednesday"],
    "thursday":       ["thursday"],
    "friday":         ["friday"],
    "saturday":       ["saturday"],
    "sunday":         ["sunday"],
}


def get_phonemes(word):
    """
    Get CMU phoneme sequence for a word.
    Returns list of phonemes (stress markers stripped) or empty list.
    """
    phones_list = pronouncing.phones_for_word(word.lower())
    if not phones_list:
        return []
    # Take first pronunciation, strip stress digits
    phones = phones_list[0].split()
    return [p.rstrip('012') for p in phones]


def get_phonemes_for_phrase(phrase):
    """
    Get phonemes for a potentially multi-word phrase.
    Splits into words and concatenates their phoneme sequences.
    """
    phrase_lower = phrase.lower().strip()

    # Check multi-word splits first
    if phrase_lower in MULTI_WORD_SPLITS:
        words = MULTI_WORD_SPLITS[phrase_lower]
    else:
        words = phrase_lower.split()

    all_phonemes = []
    for word in words:
        phonemes = get_phonemes(word)
        if phonemes:
            all_phonemes.extend(phonemes)

    return all_phonemes


def align_phonemes(target_phones, transcript_phones):
    """
    Align two phoneme sequences using SequenceMatcher.
    Returns list of (target_phone, transcript_phone) pairs.
    Unmatched positions have None on the missing side.
    """
    matcher = SequenceMatcher(None, target_phones, transcript_phones)
    aligned = []

    for opcode, t0, t1, r0, r1 in matcher.get_opcodes():
        if opcode == 'equal':
            for i in range(t1 - t0):
                aligned.append((target_phones[t0 + i], transcript_phones[r0 + i]))
        elif opcode == 'replace':
            t_len = t1 - t0
            r_len = r1 - r0
            for i in range(max(t_len, r_len)):
                t_phone = target_phones[t0 + i] if i < t_len else None
                r_phone = transcript_phones[r0 + i] if i < r_len else None
                aligned.append((t_phone, r_phone))
        elif opcode == 'delete':
            for i in range(t1 - t0):
                aligned.append((target_phones[t0 + i], None))
        elif opcode == 'insert':
            for i in range(r1 - r0):
                aligned.append((None, transcript_phones[r0 + i]))

    return aligned


def classify_errors(aligned_pairs):
    """
    Walk aligned phoneme pairs and classify errors against the taxonomy.
    Returns list of detected errors.
    """
    errors = []
    for idx, (target, produced) in enumerate(aligned_pairs):
        if target == produced:
            continue  # exact match, no error

        # Build taxonomy key
        target_key  = target  if target  else ""
        produced_key = produced if produced else ""
        taxonomy_key = f"{target_key}:{produced_key}"

        if taxonomy_key in SINHALA_L1_TAXONOMY:
            error = SINHALA_L1_TAXONOMY[taxonomy_key].copy()
            error["position"] = idx
            error["expected"] = target
            error["produced"] = produced
            errors.append(error)
        elif target and produced and target != produced:
            # Unknown substitution — log it but don't classify
            errors.append({
                "error_class": "unknown_substitution",
                "score_impact": 1,
                "description": f"Unclassified substitution: {target}→{produced}",
                "position": idx,
                "expected": target,
                "produced": produced,
            })

    return errors


def score_phoneme(transcript, target_word):
    """
    Main scoring function. Takes STT transcript and target word/phrase.
    Returns structured result dict.

    Score meanings:
      3 = exact or near-exact phoneme match
      2 = known Sinhala L1 transfer error (communicative intent clear)
      1 = significant phoneme deviation or unknown errors
      0 = no phoneme overlap / completely wrong word
    """
    # ── Step 1: Handle empty or silence transcript ────────────────────────
    if not transcript or transcript.strip() in ['', '...', '[silence]']:
        return {
            "score": 0,
            "match_type": "no_attempt",
            "error_class": None,
            "phoneme_accuracy": 0.0,
            "flagged_for_teacher": False,
            "target_phonemes": [],
            "transcript_phonemes": [],
            "errors": [],
            "detail": "Empty or silence transcript"
        }

    # ── Step 2: Get phoneme sequences ─────────────────────────────────────
    target_phones     = get_phonemes_for_phrase(target_word)
    transcript_phones = get_phonemes_for_phrase(transcript)

    # If either word is not in CMU dict, fall back gracefully
    if not target_phones:
        return {
            "score": 2,
            "match_type": "dict_miss",
            "error_class": None,
            "phoneme_accuracy": None,
            "flagged_for_teacher": False,
            "target_phonemes": [],
            "transcript_phonemes": transcript_phones,
            "errors": [],
            "detail": f"Target word '{target_word}' not in CMU dictionary — fuzzy fallback recommended"
        }

    if not transcript_phones:
        return {
            "score": 1,
            "match_type": "dict_miss",
            "error_class": None,
            "phoneme_accuracy": 0.0,
            "flagged_for_teacher": True,
            "target_phonemes": target_phones,
            "transcript_phonemes": [],
            "errors": [],
            "detail": f"Transcript '{transcript}' not in CMU dictionary"
        }

    # ── Step 3: Exact phoneme match check ─────────────────────────────────
    if target_phones == transcript_phones:
        return {
            "score": 3,
            "match_type": "exact_phoneme",
            "error_class": None,
            "phoneme_accuracy": 1.0,
            "flagged_for_teacher": False,
            "target_phonemes": target_phones,
            "transcript_phonemes": transcript_phones,
            "errors": [],
            "detail": "Exact phoneme match"
        }

    # ── Step 4: Align and classify errors ─────────────────────────────────
    aligned = align_phonemes(target_phones, transcript_phones)
    errors  = classify_errors(aligned)

    # ── Step 5: Compute phoneme accuracy ──────────────────────────────────
    matched = sum(1 for t, p in aligned if t == p and t is not None)
    total   = len(target_phones)
    phoneme_accuracy = round(matched / total, 3) if total > 0 else 0.0

    # ── Step 6: Determine score ────────────────────────────────────────────
    # If accuracy >= 0.85 with only minor errors → score 3
    if phoneme_accuracy >= 0.85 and all(
        e["error_class"] in ("vowel_substitution", "r_deletion") for e in errors
    ):
        score = 3
        match_type = "near_exact"

    # If all errors are known Sinhala L1 transfer patterns → score 2
    elif all(e["error_class"] != "unknown_substitution" for e in errors) and errors:
        score = 2
        match_type = "phoneme_approximation"

    # Some accuracy but unclassified errors → score 1
    elif phoneme_accuracy >= 0.4:
        score = 1
        match_type = "partial_match"

    # Very low overlap → score 0 (wrong word entirely)
    else:
        score = 0
        match_type = "no_match"

    # ── Step 7: Determine primary error class (most impactful error) ───────
    primary_error = None
    if errors:
        # Sort by score_impact descending, take first
        sorted_errors = sorted(errors, key=lambda e: e.get("score_impact", 0), reverse=True)
        primary_error = sorted_errors[0]["error_class"]

    flagged = score < 3 and primary_error is not None

    return {
        "score": score,
        "match_type": match_type,
        "error_class": primary_error,
        "phoneme_accuracy": phoneme_accuracy,
        "flagged_for_teacher": flagged,
        "target_phonemes": target_phones,
        "transcript_phonemes": transcript_phones,
        "errors": errors,
        "detail": f"{len(errors)} error(s) detected"
    }


# ── Demo: the exact problem we are solving ──────────────────────────────────

def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Score:            {result['score']} / 3")
    print(f"  Match type:       {result['match_type']}")
    print(f"  Error class:      {result['error_class']}")
    print(f"  Phoneme accuracy: {result['phoneme_accuracy']}")
    print(f"  Flag for teacher: {result['flagged_for_teacher']}")
    print(f"  Target phonemes:  {result['target_phonemes']}")
    print(f"  Transcript phones:{result['transcript_phonemes']}")
    if result['errors']:
        print(f"  Errors detected:")
        for e in result['errors']:
            print(f"    [{e['error_class']}] {e.get('expected','?')} → {e.get('produced','?')}: {e['description']}")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AURIVA — Phoneme-Aware Speech Assessment Demo")
    print("  RC1: Sinhala L1 Transfer Error Taxonomy")
    print("="*60)

    print("\n--- THE PROBLEM WITH FUZZY MATCHING ---")
    print("Child says 'tank you' for 'thank you'")
    print("Fuzzball edit distance: 1 character → scores ~95% → ACCEPTED as correct")
    print("This is WRONG. 'th→t' is a known Sinhala phoneme transfer error.")
    print("The child has not learned the /θ/ phoneme.")

    print("\n--- RC1 PHONEME SCORER RESULTS ---")

    # The core example: th-fronting
    print_result(
        "TARGET: 'thank you'  |  TRANSCRIPT: 'tank you'",
        score_phoneme("tank you", "thank you")
    )

    print_result(
        "TARGET: 'thank you'  |  TRANSCRIPT: 'thank you' (correct)",
        score_phoneme("thank you", "thank you")
    )

    print_result(
        "TARGET: 'thank you'  |  TRANSCRIPT: 'dank you' (th-stopping)",
        score_phoneme("dank you", "thank you")
    )

    print_result(
        "TARGET: 'sorry'  |  TRANSCRIPT: 'sowwy' (common child approximation)",
        score_phoneme("sowwy", "sorry")
    )

    print_result(
        "TARGET: 'please'  |  TRANSCRIPT: 'please' (correct)",
        score_phoneme("please", "please")
    )

    print_result(
        "TARGET: 'goodbye'  |  TRANSCRIPT: 'good bye' (split)",
        score_phoneme("good bye", "goodbye")
    )

    print_result(
        "TARGET: 'monday'  |  TRANSCRIPT: 'monday' (correct)",
        score_phoneme("monday", "monday")
    )

    print_result(
        "TARGET: 'monday'  |  TRANSCRIPT: 'mondy' (final vowel deletion)",
        score_phoneme("mondy", "monday")
    )

    print_result(
        "TARGET: 'thank you'  |  TRANSCRIPT: 'hello' (completely wrong word)",
        score_phoneme("hello", "thank you")
    )

    print("\n--- SCORING SUMMARY ---")
    print("Score 3: Correct pronunciation / near-exact phoneme match")
    print("Score 2: Known Sinhala L1 transfer error — communicative intent clear,")
    print("         child attempted the word, specific phoneme needs teaching")
    print("Score 1: Significant deviation — partial attempt")
    print("Score 0: Wrong word or no attempt")
    print("\nThis replaces fuzzy string matching which cannot distinguish")
    print("these cases and accepts transfer errors as correct productions.")
