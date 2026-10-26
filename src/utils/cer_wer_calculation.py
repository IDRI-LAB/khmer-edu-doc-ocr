from khmer_nlp_toolkits.text.segmentation import word_segment
import Levenshtein
from .khmer_norm import clean_and_normalize


def ratio_from_stats(distance, length):
    """
    Turns an (edit_distance, reference_length) pair into a single CER/WER
    ratio. The only edge case is length == 0 (empty reference): ratio is
    1.0 if the prediction still differs (distance > 0, i.e. the prediction
    has extra content the empty reference doesn't have), else 0.0.

    This is also what corpus-level aggregation should call: sum distance
    and length separately across every example, then call this once at the
    end (a MICRO-average), rather than averaging each example's own ratio
    (a MACRO-average). Micro-average is what you want when a few very long
    documents shouldn't be swamped by many short ones in the corpus score.
    """
    if length == 0:
        return 1.0 if distance > 0 else 0.0
    return distance / length


# ============================================================
# Raw CER
# ============================================================

def cer_raw_stats(gt, prediction):
    """
    (edit_distance, reference_length, bounded_length) for raw CER, before normalization.
    """
    distance = Levenshtein.distance(gt, prediction)
    length = len(gt)
    bounded_length = max(len(gt), len(prediction))
    return distance, length, bounded_length


def cer_raw(gt, prediction):
    """
    Character Error Rate without Khmer normalization.
    """
    distance, length, _ = cer_raw_stats(gt, prediction)
    return ratio_from_stats(distance, length)


# ============================================================
# Normalized CER
# ============================================================

def cer_normalized_stats(gt, prediction):
    """
    (edit_distance, reference_length, bounded_length) for CER after Khmer normalization.
    """
    gt = clean_and_normalize(gt)
    prediction = clean_and_normalize(prediction)
    return cer_raw_stats(gt, prediction)


def cer_normalized(gt, prediction):
    """
    Character Error Rate after Khmer normalization.
    """
    distance, length, _ = cer_normalized_stats(gt, prediction)
    return ratio_from_stats(distance, length)


# ============================================================
# Raw WER
# ============================================================

def wer_raw_stats(gt, prediction):
    """
    (edit_distance, reference_word_count, bounded_word_count) for raw WER, before normalization.
    """
    gt_words = word_segment(gt, word_type="com", rt_type="list")
    prediction_words = word_segment(prediction, word_type="com", rt_type="list")
    
    if len(gt_words) == 0 and len(prediction_words) == 0:
        return 0, 0, 0
        
    gt_str = " ".join(gt_words)
    prediction_str = " ".join(prediction_words)
    
    distance = Levenshtein.distance(gt_str.split(), prediction_str.split())
    length = len(gt_words)
    bounded_length = max(len(gt_words), len(prediction_words))
    return distance, length, bounded_length


def wer_raw(gt, prediction):
    """
    Word Error Rate without Khmer normalization.
    """
    distance, length, _ = wer_raw_stats(gt, prediction)
    return ratio_from_stats(distance, length)


# ============================================================
# Normalized WER
# ============================================================

def wer_normalized_stats(gt, prediction):
    """
    (edit_distance, reference_word_count, bounded_word_count) for WER after Khmer normalization.
    """
    gt_words = word_segment(gt, word_type="com", rt_type="list")
    prediction_words = word_segment(prediction, word_type="com", rt_type="list")

    gt_words = [clean_and_normalize(w) for w in gt_words]
    prediction_words = [clean_and_normalize(w) for w in prediction_words]

    if len(gt_words) == 0 and len(prediction_words) == 0:
        return 0, 0, 0
        
    gt_str = " ".join(gt_words)
    prediction_str = " ".join(prediction_words)
    
    distance = Levenshtein.distance(gt_str.split(), prediction_str.split())
    length = len(gt_words)
    bounded_length = max(len(gt_words), len(prediction_words))
    return distance, length, bounded_length


def wer_normalized(gt, prediction):
    """
    Word Error Rate after Khmer normalization.
    """
    distance, length, _ = wer_normalized_stats(gt, prediction)
    return ratio_from_stats(distance, length)


def evaluate_ocr(gt, prediction):
    return {
        "CER_raw": cer_raw(gt, prediction),
        "CER_normalized": cer_normalized(gt, prediction),
        "WER_raw": wer_raw(gt, prediction),
        "WER_normalized": wer_normalized(gt, prediction),
    }


def evaluate_ocr_stats(gt, prediction):
    """
    Same four metrics as evaluate_ocr(), but as (distance, length) pairs
    instead of final ratios — this is the form to collect per-example when
    you plan to combine many examples into a single corpus-level
    micro-average instead of averaging each example's own ratio.
    """
    return {
        "CER_raw": cer_raw_stats(gt, prediction),
        "CER_normalized": cer_normalized_stats(gt, prediction),
        "WER_raw": wer_raw_stats(gt, prediction),
        "WER_normalized": wer_normalized_stats(gt, prediction),
    }