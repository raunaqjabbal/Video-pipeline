import os
import subprocess
import sys

def wav2lip_run(adir):
  vid_path = f'{os.getcwd()}/intermediate/{adir}/FOMM-complete.mp4'
  aud_path = f'{os.getcwd()}/intermediate/{adir}/Audio.wav'
  out_path = f'{os.getcwd()}/intermediate/{adir}/Avatar.mp4'
  os.chdir('LIHQ/Wav2Lip')
  command = f'python inference.py --checkpoint_path checkpoints/wav2lip.pth --face {vid_path} --audio {aud_path} --outfile {out_path}  --pads 0 20 0 0'
  try:
    subprocess.call(command, shell=True)
  except subprocess.CalledProcessError:
    print('!!!!!!! Error with Wav2Lip Paths !!!!!!')
    sys.exit()
  os.chdir('..')
  os.chdir('..')
  
  if os.path.exists(out_path):
    os.remove(vid_path)
    os.remove
  else:
    print("Wav2ip failed: ", adir)