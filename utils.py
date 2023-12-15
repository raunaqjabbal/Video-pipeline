#extractvideofromaudio
from moviepy.editor import VideoFileClip as _VideoFileClip
from moviepy.editor import AudioFileClip as _AudioFileClip
from moviepy.editor import concatenate_audioclips as _concatenate_audioclips
from moviepy.editor import concatenate_videoclips as _concatenate_videoclips
from moviepy.editor import CompositeVideoClip as _CompositeVideoClip
from moviepy.editor import vfx as _vfx
from transformers import AutoProcessor, BarkModel
from optimum.bettertransformer import BetterTransformer


import torch
import numpy as np
from PIL import Image
import os
import scipy
import nltk
import gc
import glob
from tqdm import tqdm
import sys
import shutil
import subprocess
import imageio
import librosa
from skimage.transform import resize
from skimage import img_as_ubyte


os.environ['SUNO_OFFLOAD_CPU'] = 'True'
os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE

sys.path.append(os.path.join( "LIHQ", "first_order_model"))
sys.path.append(os.path.join( "LIHQ", "procedures"))

from LIHQ.face_crop import crop_face as crop_face #################################################
from LIHQ.first_order_model.demo import load_checkpoints, make_animation


import multiprocessing as mp

def child(func):
    def wrapper(*args, **kwargs):
        ctx = mp.get_context('spawn')
        process = ctx.Process(target=func, args=args, kwargs=kwargs)
        process.start()
        process.join()
    return wrapper

def preprocess_avatar(inputfolder, backgroundpath=None, outputfolder="inputs/preprocessed_faces"):
    if not os.path.exists("preprocess"):
        os.mkdir(os.path.join("preprocess"))
    
    print("Cropping faces...") 
    cropfolder = os.path.join("preprocess","cropped")
    if not os.path.exists(cropfolder):
        os.mkdir(cropfolder)
    for i in os.listdir(inputfolder):
        crop_face(os.path.join(inputfolder,i), os.path.join("preprocess","cropped", i))
    
    print("Upscaling faces...") 
    upscalefolder = os.path.join("preprocess", "upscaled")
    upscale_image(cropfolder, "preprocess")
    
    print("Generating image masks...") 
    maskfolder = os.path.join("preprocess", "masks")
    if not os.path.exists(maskfolder):
        os.mkdir(maskfolder)
    image_mask(upscalefolder,maskfolder)
    
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
        
    if type(backgroundpath)==str:
        print("Adding background...")
        for i in os.listdir(upscalefolder):
            image_matting(os.path.join(upscalefolder,i), outputfolder, maskfolder, backgroundpath)
    
    else:
        shutil.copytree(upscalefolder, outputfolder, dirs_exist_ok=True)
    
    if len(os.listdir(inputfolder))==len(os.listdir(outputfolder)):
        shutil.rmtree("preprocess")

def upscale_image(inputfolder, outputfolder):
    filepath = os.path.join("gfpgan", "inference_gfpgan.py")
    subprocess.call(f"python {filepath} -i {inputfolder} -o {outputfolder} -v 1.3 -s 4 --bg_upsampler realesrgan", shell=True)
 
def image_mask(inputfolder, outputfolder):
    # inputfolder = os.path.join(os.getcwd(),inputfolder)
    # outputfolder = os.path.join(os.getcwd(),outputfolder)
    # os.chdir("./LIHQ/MODNet")
    subprocess.call(f"python LIHQ/MODNet/inference.py --input-path {inputfolder} --output-path {outputfolder} --ckpt-path LIHQ/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt", shell=True)
    # os.chdir("..")
    # os.chdir("..")
    

def image_matting(inputpath, outputfolder,maskfolder, backgroundpath):
    image_name = inputpath
    matte_name = os.path.basename(image_name).split('.')[0] + '.png'
    background = Image.open(backgroundpath) #Set background image
    image = Image.open(inputpath)
    matte = Image.open(os.path.join(maskfolder, matte_name))

    #Reshaping background as needed
    if background.size != image.size:
        background = background.resize((image.size[0], image.size[1]))

    #Matte transformaiton
    image = np.asarray(image)
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + background * (1 - matte)
    final = Image.fromarray(np.uint8(foreground))
    final.save(os.path.join(outputfolder, matte_name))

@child
def generate_audio(textdataset, projectpath="intermediate", speaker = 1):
    print("Generating Audio...")    
    for sample, texts in tqdm(textdataset.items()):
        if not os.path.exists(os.path.join(projectpath, sample)):
            os.makedirs(os.path.join(projectpath, sample))
            
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
                scipy.io.wavfile.write(os.path.join(projectpath,sample,"audio_"+str(count)+".wav"), rate=SAMPLE_RATE, data=partaudio)
            elif indexes[i]!=indexes[i+1]:
                scipy.io.wavfile.write(os.path.join(projectpath,sample,"audio_"+str(count)+".wav"), rate=SAMPLE_RATE, data=partaudio)
                partaudio=[]
                count+=1

@child
def generate_audio2(textdataset, projectpath="intermediate", speaker = 1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small",torch_dtype=torch.float16).to(device)
    model = BetterTransformer.transform(model, keep_original_model=False)
    model.enable_cpu_offload()

    print("Generating Audio...")    
    for sample, texts in tqdm(textdataset.items()):
        if not os.path.exists(os.path.join(projectpath, sample)):
            os.makedirs(os.path.join(projectpath, sample))
            
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
                scipy.io.wavfile.write(os.path.join(projectpath,sample,"audio_"+str(count)+".wav"), rate=model.generation_config.sample_rate, data=partaudio)
            elif indexes[i]!=indexes[i+1]:
                scipy.io.wavfile.write(os.path.join(projectpath,sample,"audio_"+str(count)+".wav"), rate=model.generation_config.sample_rate, data=partaudio)
                partaudio=[]
                count+=1


def chop_refvid(projectpath="intermediate", ref_vid='inputs/ref_video/syn_reference.mp4'):
    i = 0
    for adir in os.listdir(projectpath):
        # audio = glob.glob(f'{projectpath}/{adir}/*')[0]
        if os.path.exists(os.path.join(projectpath,adir,"Audio.wav")):
            audio = os.path.join(projectpath,adir,"Audio.wav")
        else:
            audio = os.path.join(projectpath,adir,f"{adir}.wav")

        audio_length = librosa.get_duration(filename = audio)
        output_video_path = f'./{projectpath}/{adir}/FOMM.mp4'
        with _VideoFileClip(ref_vid) as video:
            if video.duration < audio_length:
                sys.exit('Reference video is shorter than audio. You can:'
                        'Chop audio to multiple folders, reduce video offset,'
                        'use a longer reference video, use shorter audio.')

            new = video.subclip(0, audio_length)
            new.write_videofile(output_video_path, audio_codec='aac', verbose=False, logger=None)
            i += 1

@child
def FOMM(face, projectpath="intermediate"):
    generator, kp_detector = load_checkpoints(config_path='./LIHQ/first_order_model/vox-256.yaml', checkpoint_path='./LIHQ/first_order_model/vox-cpk.pth.tar')
    
    for adir in os.listdir(projectpath):
        sub_clip = f'./{projectpath}/{adir}/FOMM.mp4'
            
        source_image = imageio.imread(face)
        reader = imageio.get_reader(sub_clip)
        source_image = resize(source_image, (256, 256))[..., :3]
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative = True)

        FOMM_out_path = f'./{projectpath}/{adir}/FOMM.mp4'
        imageio.mimsave(FOMM_out_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        gc.collect()

@child
def wav2lip(projectpath="intermediate"):
    gc.collect()
    print("Running Wav2Lip")
    for i in tqdm(os.listdir(projectpath)):
        # wav2lip_run(i)
        vid_path = f'{os.getcwd()}/intermediate/{i}/FOMM.mp4'
        if os.path.exists(f'{os.getcwd()}/intermediate/{i}/Audio.wav'):
            aud_path = f'{os.getcwd()}/intermediate/{i}/Audio.wav'
        else:
            aud_path = f'{os.getcwd()}/intermediate/{i}/{i}.wav'
        out_path = f'{os.getcwd()}/intermediate/{i}/Avatar.mp4'
        os.chdir('LIHQ/Wav2Lip')
        command = f'python inference.py --checkpoint_path checkpoints/wav2lip.pth --face {vid_path} --audio {aud_path} --outfile {out_path}  --pads 0 20 0 0'
        try:
            # THIS
            subprocess.call(command, shell=True)
            
            # OR ALL THIS
            # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # while True:
            #     output = process.stdout.readline()
            #     if output == '' and process.poll() is not None:
            #         break
            #     if output:
            #         print(output.strip())
            # print(f"Script execution completed with exit code:", process.poll())
        except subprocess.CalledProcessError:
            print("Wav2ip failed: ", i)
        os.chdir('..')
        os.chdir('..')
        if os.path.exists(out_path):
            os.remove(vid_path)
        else:
            print("Wav2ip failed: ", i)        

def concatenate_videos(projectpath="intermediate"):
    print("Concatenting Videos...")
    for i in tqdm(os.listdir(projectpath)):
        clips = [_VideoFileClip(os.path.join(j)) for j in sorted(glob.glob(os.path.join(projectpath,i,"*video*")))]
        clip = _concatenate_videoclips(clips)
        clip.write_videofile(os.path.join(projectpath,i,"Video.mp4"), verbose=False, logger=None)
                
    if os.path.exists(os.path.join(projectpath,i,"Video.mp4")):
        for j in sorted(glob.glob(os.path.join(projectpath,i,"*video*"))):
            if not j.endswith("Video.mp4"):
                os.remove(j)    

def concatenate_audios(projectpath="intermediate"):
    print("Concatenating Audios...")
    for i in tqdm(os.listdir(projectpath)):
        clips = [_AudioFileClip(os.path.join(j)) for j in sorted(glob.glob(os.path.join(projectpath,i,"*audio*")))]
        clip = _concatenate_audioclips(clips)
        clip.write_audiofile(os.path.join(projectpath,i,"Audio.wav"), verbose=False, logger=None)
    
    if os.path.exists(os.path.join(projectpath,i,"Audio.wav")):
        for j in sorted(glob.glob(os.path.join(projectpath,i,"*audio*"))):
            if not j.endswith("Audio.wav"):
                os.remove(j)    

def merge_audio_video(projectpath="intermediate",videopath="inputs/videos", kind = "delay"):
    print("Merging audios and videos...")
    for i in tqdm(os.listdir(projectpath)):
        idx=1
        for j in sorted(os.listdir(os.path.join(projectpath, i))):
            if j.startswith("audio"):
                audio_clip = _AudioFileClip(os.path.join(projectpath,i,j))
                video_clip = _VideoFileClip(os.path.join(videopath,i,f"video_{idx}.mp4"))

                if kind=="delay":
                    concat_clip = video_clip.to_ImageClip(t=video_clip.duration-1, duration=audio_clip.duration-video_clip.duration)
                    concat_clip = _concatenate_videoclips([video_clip, concat_clip])
                    concat_clip = concat_clip.set_audio(audio_clip)
                    concat_clip.write_videofile(os.path.join(projectpath,i,f'video_{idx}.mp4'), verbose = False, logger = None)
                    idx+=1
                elif kind=="scale":
                    slowed_clip = video_clip.fx(_vfx.speedx, video_clip.duration / audio_clip.duration)
                    slowed_clip = slowed_clip.set_audio(audio_clip)
                    slowed_clip.write_videofile(os.path.join(projectpath,i,f'video_{idx}.mp4'), verbose = False, logger = None)
                    idx+=1

def upscale_avatar(projectpath="intermediate", kind= "realesr-animevideov3"):
    print("Upscaling avatars...")
    for i in tqdm(os.listdir(projectpath)):
        location = os.path.join(projectpath,i,"Avatar.mp4")
        os.system(f"python Real-ESRGAN/inference_realesrgan_video.py -i {location} -n {kind} -s 2 -suffix x2 -o {os.path.join(projectpath,i)}")
        ## HELP?

def merge_video_avatar(projectpath="intermediate", outputpath="results",  padding=10, location="left", speed=1):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    print("Merging video with avatar... ")
    for i in tqdm(os.listdir(projectpath)):
        clip = _VideoFileClip(os.path.join(projectpath,i,"Video.mp4"))
        clip1 = clip.without_audio()

        clip2 = _VideoFileClip(os.path.join(projectpath,i,"Avatar.mp4"))
        clip2 = clip2.resize(0.20)
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
