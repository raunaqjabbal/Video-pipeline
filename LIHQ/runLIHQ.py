import os
import glob
import shutil
from pathlib import Path
import time
import subprocess
import argparse
import gc
from .procedures.av_scripts import *


from .first_order_model.demo import load_checkpoints
from .procedures.fomm_scripts import FOMM_chop_refvid, FOMM_run

from .procedures.wav2lip_scripts import wav2lip_run
from .procedures.qvi_scripts import qvi_config


def run(face, audio_super = 'intermediate', ref_vid = 'inputs/ref_video/syn_reference.mp4', ref_vid_offset = [0], frame_int = None, clear_outputs=True, save_path = None):

    #Miscellaneous things
    print("Initializing")
      #Turning face &offset to arrays as needed
    if not isinstance(face, list):
        face = [face]
    if not isinstance(ref_vid_offset, list):
        ref_vid_offset = [ref_vid_offset]

      #Determining final fps for ffmpeg
    if frame_int is not None:
        fps = 25 * (frame_int + 1)
    else:
        fps = 25

      #Deleteing output files
    if clear_outputs == True:
        for path in Path("./output").glob("**/*"):
            if path.is_file():
                path.unlink()

      #A/V Set up
    R1start = time.time()
    if audio_super[-1:] != '/':
        audio_super = audio_super + '/'
    aud_dir_names = get_auddirnames(audio_super)
    for adir in aud_dir_names:
        combine_audiofiles(adir, audio_super)

      #Expanding face array as needed
    while len(face) < len(aud_dir_names):
        face.append(face[0])
        
    gc.collect()
    #FOMM
      #Cropping reference video
    FOMM_chop_refvid(aud_dir_names, ref_vid, audio_super, ref_vid_offset)
      #Running FOMM (Mimicking facial movements from reference video)
    print("Running First Order Motion Model")
    generator, kp_detector = load_checkpoints(config_path='./LIHQ/first_order_model/config/vox-256.yaml', checkpoint_path='./LIHQ/first_order_model/vox-cpk.pth.tar')
    i = 0
    for adir in aud_dir_names:
        gc.collect()
        sub_clip = f'./intermediate/{adir}/FOMM-chop.mp4'
        FOMM_run(face[i], sub_clip, generator, kp_detector, adir, Round = "1")
        os.remove(sub_clip)
        i+=1
    print("FOMM Success!")
    
    gc.collect()