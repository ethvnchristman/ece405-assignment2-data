import fasttext

NSFW_MODEL_PATH = "/Users/ethanchristman/Downloads/jigsaw_fasttext_bigrams_nsfw_final.bin"
HATE_MODEL_PATH = "/Users/ethanchristman/Downloads/jigsaw_fasttext_bigrams_hatespeech_final.bin"

_nsfw_model = None
_hate_model = None


def _get_nsfw_model():
    global _nsfw_model
    if _nsfw_model is None:
        _nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
    return _nsfw_model


def _get_hate_model():
    global _hate_model
    if _hate_model is None:
        _hate_model = fasttext.load_model(HATE_MODEL_PATH)
    return _hate_model


def _predict(model, text: str) -> tuple[str, float]:
    clean = text.replace("\n", " ")
    labels, scores = model.predict(clean, k=1)
    label = labels[0].replace("__label__", "")
    return label, float(scores[0])


def classify_nsfw(text: str) -> tuple[str, float]:
    return _predict(_get_nsfw_model(), text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    return _predict(_get_hate_model(), text)
