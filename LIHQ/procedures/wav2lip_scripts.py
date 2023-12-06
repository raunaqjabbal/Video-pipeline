import os
import subprocess
import sys

def wav2lip_run(adir):
  vid_path = f'./intermediate/{adir}/FOMM-complete.mp4'
  aud_path = f'./intermediate/{adir}/Audio.wav'
  out_path = f'./intermediate/{adir}/Avatar.mp4'
  # os.chdir('Wav2Lip')
  
  print("running")
  print(vid_path, aud_path, out_path)
  command = f'python ./LIHQ/Wav2Lip/inference.py --checkpoint_path ./LIHQ/Wav2Lip/checkpoints/wav2lip.pth --face {vid_path} --audio {aud_path} --outfile {out_path}  --pads 0 20 0 0'
  try:
    subprocess.call(command, shell=True)
  except subprocess.CalledProcessError:
    print('!!!!!!! Error with Wav2Lip Paths !!!!!!')
    sys.exit()
  # os.chdir('..')