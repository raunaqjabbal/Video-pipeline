import nltk
nltk.download("punkt")


import pathlib
import sys
import os
# maindir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join( "LIHQ", "first_order_model"))
sys.path.append(os.path.join( "LIHQ", "procedures"))


import os

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

from bark.generation import preload_models
preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)
