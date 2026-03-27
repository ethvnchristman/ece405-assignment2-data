import os as _os
from pathlib import Path as _Path

_quality_model = None
_MODEL_PATH = _Path(__file__).resolve().parent.parent / "data" / "quality_classifier.bin"


def _get_quality_model():
    global _quality_model
    if _quality_model is None:
        import fasttext
        _quality_model = fasttext.load_model(str(_MODEL_PATH))
    return _quality_model


def classify_quality(text: str) -> tuple[str, float]:
    model = _get_quality_model()
    clean = text.replace("\n", " ")
    labels, scores = model.predict(clean, k=1)
    label = labels[0].replace("__label__", "")
    return label, float(scores[0])


def gopher_quality_filter(text: str) -> bool:
    words = text.split()
    n = len(words)

    if n < 50 or n > 100_000:
        return False

    mean_len = sum(len(w) for w in words) / n
    if mean_len < 3 or mean_len > 10:
        return False

    lines = text.splitlines()
    if lines:
        ellipsis_frac = sum(1 for line in lines if line.rstrip().endswith("...")) / len(lines)
        if ellipsis_frac > 0.30:
            return False

    alpha_frac = sum(1 for w in words if any(c.isalpha() for c in w)) / n
    if alpha_frac < 0.80:
        return False

    return True


def gopher_quality_filter_reasons(text: str) -> tuple[bool, list[str], dict]:
    words = text.split()
    n = len(words)
    lines = text.splitlines()

    mean_len = sum(len(w) for w in words) / n if n else 0
    ellipsis_frac = (
        sum(1 for line in lines if line.rstrip().endswith("...")) / len(lines)
        if lines else 0
    )
    alpha_frac = sum(1 for w in words if any(c.isalpha() for c in w)) / n if n else 0

    metrics = {
        "word_count": n,
        "mean_word_len": round(mean_len, 3),
        "ellipsis_pct": round(ellipsis_frac * 100, 1),
        "alpha_pct": round(alpha_frac * 100, 1),
    }

    failures = []
    if n < 50 or n > 100_000:
        failures.append(f"word_count={n} (need 50–100000)")
    if mean_len < 3 or mean_len > 10:
        failures.append(f"mean_word_len={mean_len:.2f} (need 3–10)")
    if lines and ellipsis_frac > 0.30:
        failures.append(f"ellipsis_pct={ellipsis_frac*100:.1f}% (max 30%)")
    if n and alpha_frac < 0.80:
        failures.append(f"alpha_pct={alpha_frac*100:.1f}% (min 80%)")

    return len(failures) == 0, failures, metrics
