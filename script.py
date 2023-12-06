import nltk
nltk.download("punkt")

import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

from bark.generation import preload_models
preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)