#extractvideofromaudio
from moviepy.editor import VideoFileClip as _VideoFileClip
from moviepy.editor import AudioFileClip as _AudioFileClip
from moviepy.editor import concatenate_audioclips as _concatenate_audioclips
from moviepy.editor import concatenate_videoclips as _concatenate_videoclips
from moviepy.editor import CompositeVideoClip as _CompositeVideoClip
from moviepy.editor import vfx as _vfx
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
import os
import scipy
import nltk
import gc
import glob
from tqdm import tqdm
import sys
# codec="libx264"
import shutil
import subprocess


os.environ['SUNO_OFFLOAD_CPU'] = 'True'
os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
from bark.generation import generate_text_semantic,preload_models
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

sys.path.append(os.path.join( "LIHQ", "first_order_model"))
sys.path.append(os.path.join( "LIHQ", "procedures"))

from LIHQ import runLIHQ
from LIHQ.procedures.wav2lip_scripts import wav2lip_run
from LIHQ.procedures.face_align.face_crop import crop_face as _crop_face
from LIHQ.procedures.matting_scripts import image_matting as _image_matting


def preprocess_avatar(inputpath,  backgroundpath, outputpath="inputs/preprocessed_faces"):
    if not os.path.exists("preprocess"):
        os.mkdir(os.path.join("preprocess"))
    
    print("Cropping faces...") 
    croppath = os.path.join("preprocess","cropped")
    if not os.path.exists(croppath):
        os.mkdir(croppath)
    for i in os.listdir(inputpath):
        crop_face(os.path.join(inputpath,i), os.path.join("preprocess","cropped", i))
    
    print("Upscaling faces...") 
    upscalepath = os.path.join("preprocess", "upscaled")
    upscale_image(croppath, "preprocess")
    
    print("Generating image masks...") 
    maskpath = os.path.join("preprocess", "masks")
    if not os.path.exists(maskpath):
        os.mkdir(maskpath)
    image_mask(upscalepath,maskpath)

    print("Adding background...")
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    for i in os.listdir(upscalepath):
        image_matting(os.path.join(upscalepath,i), outputpath, maskpath, backgroundpath)
    
    if len(os.listdir(inputpath))==len(os.listdir(outputpath)):
        shutil.rmtree("preprocess")

def crop_face(inputpath, outputpath):
    _crop_face(inputpath,outputpath)

def upscale_image(inputpath, outputpath):
    filepath = os.path.join("gfpgan", "inference_gfpgan.py")
    subprocess.call(f"python {filepath} -i {inputpath} -o {outputpath} -v 1.3 -s 4 --bg_upsampler realesrgan", shell=True)
    #  -n realesr-animevideov3 / RealESRGAN_x4plus
 
def image_mask(inputpath, outputpath):
    inputpath = os.path.join(os.getcwd(),inputpath)
    outputpath = os.path.join(os.getcwd(),outputpath)
    os.chdir("./LIHQ/MODNet")
    subprocess.call(f"python -m demo.image_matting.colab.inference --input-path {inputpath} --output-path {outputpath} --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt", shell=True)
    os.chdir("..")
    os.chdir("..")
    
def image_matting(inputpath, outputpath, maskpath, backgroundpath):
    _image_matting(backgroundpath, inputpath, maskpath, outputpath)

# def demo():
#     clip = VideoFileClip("drive/MyDrive/Content Video Production/non_interactive_sample (with audio)/FinalAV.mp4")
#     audio = clip.audio
#     audio.write_audiofile("CompleteAudio.mp3")

def generate_audio(textdataset, audiopath="intermediate", speaker = 1):
    print("Generating Audio...")    
    for sample, texts in tqdm(textdataset.items()):
        if not os.path.exists(os.path.join(audiopath, sample)):
            os.makedirs(os.path.join(audiopath, sample))
            
        newtxt=[]
        indexes=[]
        for idx,i in enumerate(texts):
            tokenized_sentence = nltk.sent_tokenize(i)
            indexes += [idx]*len(tokenized_sentence)
            newtxt += tokenized_sentence

        GEN_TEMP = 0.6
        SPEAKER = "v2/en_speaker_"+str(speaker)
        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

        speech_values = []
        for sentence in newtxt:
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=SPEAKER,
                temp=GEN_TEMP,
                min_eos_p=0.05,  # this controls how likely the generation is to end
                silent=True
            )
            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER, silent=True)
            speech_values += [audio_array]
        
        count = 1
        partaudio=[]
        for i,audio in enumerate(speech_values):
            partaudio = np.concatenate([partaudio,audio])
            if i==len(indexes)-1:
                scipy.io.wavfile.write(os.path.join(audiopath,sample,"audio_"+str(count)+".wav"), rate=SAMPLE_RATE, data=np.int16(partaudio / 256.0))
            elif indexes[i]!=indexes[i+1]:
                scipy.io.wavfile.write(os.path.join(audiopath,sample,"audio_"+str(count)+".wav"), rate=SAMPLE_RATE, data=np.int16(partaudio / 256.0))
                partaudio=[]
                count+=1


def generate_audio2(textdataset, audiopath="intermediate", speaker = 1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small",torch_dtype=torch.float16).to(device)
    model.enable_cpu_offload()

    print("Generating Audio...")    
    for sample, texts in tqdm(textdataset.items()):
        if not os.path.exists(os.path.join(audiopath, sample)):
            os.makedirs(os.path.join(audiopath, sample))
            
        newtxt=[]
        indexes=[]
        for idx,i in enumerate(texts):
            tokenized_sentence = nltk.sent_tokenize(i)
            indexes += [idx]*len(tokenized_sentence)
            newtxt += tokenized_sentence

        inputs = processor(text=newtxt, return_tensors="pt", voice_preset="v2/en_speaker_"+str(speaker))
        inputs = {key:value.to(device) for key,value in inputs.items()}
        speech_values = model.generate(**inputs, min_eos_p=0.05).cpu().numpy().squeeze()

        count = 1
        partaudio=[]
        for i,audio in enumerate(speech_values):
            partaudio = np.concatenate([partaudio,audio])
            if i==len(indexes)-1:
                scipy.io.wavfile.write(os.path.join(audiopath,sample,"audio_"+str(count)+".wav"), rate=model.generation_config.sample_rate, data=np.int16(partaudio / 256.0))
            elif indexes[i]!=indexes[i+1]:
                scipy.io.wavfile.write(os.path.join(audiopath,sample,"audio_"+str(count)+".wav"), rate=model.generation_config.sample_rate, data=np.int16(partaudio / 256.0))
                partaudio=[]
                count+=1


def wav2lip(path="intermediate"):
    gc.collect()
    print("Running Wav2Lip")
    for i in tqdm(os.listdir(path)):
        wav2lip_run(i)

def concatenate_videos(videopath="intermediate"):
    print("Concatenting Videos...")
    for i in tqdm(os.listdir(videopath)):
        clips = [_VideoFileClip(os.path.join(j)) for j in sorted(glob.glob(os.path.join(videopath,i,"*video*")))]
        clip = _concatenate_videoclips(clips)
        clip.write_videofile(os.path.join(videopath,i,"Video.mp4"), verbose=False, logger=None)

def concatenate_audios(audiopath="intermediate"):
    print("Concatenating Audios...")
    for i in tqdm(os.listdir(audiopath)):
        clips = [_AudioFileClip(os.path.join(j)) for j in sorted(glob.glob(os.path.join(audiopath,i,"*audio*")))]
        clip = _concatenate_audioclips(clips)
        clip.write_audiofile(os.path.join(audiopath,i,"Audio.wav"), verbose=False, logger=None)

# Merge Audio with Video
def merge_audio_video(audiopath="intermediate",videopath="inputs/videos", type = "delay"):
    print("Merging audios and videos...")
    for i in tqdm(os.listdir(audiopath)):
        idx=1
        for j in sorted(os.listdir(os.path.join(audiopath, i))):
            if j.startswith("audio"):
                audio_clip = _AudioFileClip(os.path.join(audiopath,i,j))
                video_clip = _VideoFileClip(os.path.join(videopath,i,f"video_{idx}.mp4"))

                if type=="delay":
                    concat_clip = video_clip.to_ImageClip(t=video_clip.duration-1, duration=audio_clip.duration-video_clip.duration)
                    concat_clip = _concatenate_videoclips([video_clip, concat_clip])
                    concat_clip = concat_clip.set_audio(audio_clip)
                    concat_clip.write_videofile(os.path.join(audiopath,i,f'video_{idx}.mp4'), verbose = False, logger = None)
                    idx+=1
                elif type=="scale":
                    slowed_clip = video_clip.fx(_vfx.speedx, video_clip.duration / audio_clip.duration)
                    slowed_clip = slowed_clip.set_audio(audio_clip)
                    slowed_clip.write_videofile(os.path.join(audiopath,i,f'video_{idx}.mp4'), verbose = False, logger = None)
                    idx+=1

def upscale_avatar(avatarpath="intermediate"):
    print("Upscaling avatars...")
    for i in tqdm(os.listdir(avatarpath)):
        location = os.path.join(avatarpath,i,"Avatar.mp4")
        os.system(f"python Real-ESRGAN/inference_realesrgan_video.py -i {location} -n realesr-animevideov3 -s 2 -suffix x2 -o {os.path.join(avatarpath,i)}")
        os.remove(location)
        
# def upscale_avatar2(avatarpath):
#     os.system(f"python Real-ESRGAN/inference_realesrgan_video.py -i {avatarpath} -n realesr-animevideov3 -s 2 --suffix x2 -o {avatarpath}")

def merge_video_avatar(avatarpath="intermediate", outputpath="results",  padding=10, location="left", speed=1):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    print("Merging video with avatar... ")
    for i in tqdm(os.listdir(avatarpath)):
        clip = _VideoFileClip(os.path.join(avatarpath,i,"Video.mp4"))
        clip1 = clip.without_audio()

        clip2 = _VideoFileClip(os.path.join(avatarpath,i,"Avatar.mp4"))
        clip2 = clip2.resize(0.20)
        print(clip1.duration==clip2.duration)
        if location=="right":
            video = _CompositeVideoClip([clip1, clip2.set_position((clip1.size[0]-(clip2.size[0]+padding),padding))])
        else:
            video = _CompositeVideoClip([clip1, clip2.set_position((padding,padding))])

        # if speed!=1:
        #     video = video.fx(_vfx.speedx, speed)
        video.fps = clip1.fps
        videopath = os.path.join(outputpath, f"{i}.mp4")
        video.write_videofile(videopath,  verbose = False, logger = None)

        if speed!=1:
            audiopath = os.path.join(outputpath, f"{i}.mp3")
            os.system(f"ffmpeg -i {videopath} -filter:a atempo={speed} -vn {audiopath} -hide_banner -loglevel error")
            video = video.fx(_vfx.speedx, speed)
            video.audio = _AudioFileClip(audiopath)   
            video.write_videofile(videopath,  verbose = False, logger = None)
    
            os.remove(audiopath)

def change_video_speed(videopath, speed=1):
    for i in tqdm(os.listdir(videopath)):
        video = _VideoFileClip(os.path.join(videopath,i))

        if speed!=1:
            audiopath = os.path.join(videopath, "demo.mp3")
            os.system(f"ffmpeg -i {os.path.join(videopath,i)} -filter:a atempo={speed} -vn {audiopath} -hide_banner -loglevel error")
            video = video.fx(_vfx.speedx, speed)
            video.audio = _AudioFileClip(audiopath)   
            video.write_videofile(os.path.join(videopath, f"{i}_{speed}.mp4"),  verbose = False, logger = None)
            os.remove(audiopath)
