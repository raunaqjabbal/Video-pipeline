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
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # while True:
    #     output = process.stdout.readline()
    #     if output == '' and process.poll() is not None:
    #         break
    #     if output:
    #         print(output.strip())
    # print(f"Script execution completed with exit code:", process.poll())
  except subprocess.CalledProcessError:
    print('!!!!!!! Error with Wav2Lip Paths !!!!!!')
    sys.exit()
  os.chdir('..')
  os.chdir('..')