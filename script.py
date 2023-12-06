import nltk
nltk.download("punkt")

from bark.generation import preload_models
preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)