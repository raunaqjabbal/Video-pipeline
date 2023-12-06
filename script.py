
import sys
import os
sys.path.append(os.path.join("LIHQ", "first_order_model"))
sys.path.append(os.path.join("LIHQ", "procedures"))
os.environ['SUNO_OFFLOAD_CPU'] = 'True'
os.environ['SUNO_USE_SMALL_MODELS'] = 'True'


import nltk
nltk.download("punkt")


from bark.generation import preload_models
preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)

