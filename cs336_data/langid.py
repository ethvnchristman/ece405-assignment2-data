import fasttext

_model = None
MODEL_PATH = "/Users/ethanchristman/Downloads/lid.176.bin"


def _get_model():
    global _model
    if _model is None:
        _model = fasttext.load_model(MODEL_PATH)
    return _model


def identify_language(text: str) -> tuple[str, float]:
    model = _get_model()
    clean = text.replace("\n", " ")
    labels, scores = model.predict(clean, k=1)
    lang = labels[0].replace("__label__", "")
    return lang, float(scores[0])
