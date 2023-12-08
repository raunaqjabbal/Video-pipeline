import os
import gc
import sys
import glob
import imageio
import librosa
from skimage.transform import resize
from skimage import img_as_ubyte
from moviepy.video.io.VideoFileClip import VideoFileClip


from .first_order_model.demo import load_checkpoints, make_animation

def chop_refvid(projectpath="intermediate", ref_vid='inputs/ref_video/syn_reference.mp4'):
    i = 0
    for adir in os.listdir(projectpath):
        # audio = glob.glob(f'{projectpath}/{adir}/*')[0]
        if os.path.exists(os.path.join(projectpath,adir,"Audio.wav")):
            audio = os.path.join(projectpath,adir,"Audio.wav")
        else:
            audio = os.path.join(projectpath,adir,f"{adir}.wav")

        audio_length = librosa.get_duration(path = audio)
        output_video_path = f'./{projectpath}/{adir}/FOMM.mp4'
        with VideoFileClip(ref_vid) as video:
            if video.duration < audio_length:
                sys.exit('Reference video is shorter than audio. You can:'
                        'Chop audio to multiple folders, reduce video offset,'
                        'use a longer reference video, use shorter audio.')

            new = video.subclip(0, audio_length)
            new.write_videofile(output_video_path, audio_codec='aac', verbose=False, logger=None)
            i += 1

def FOMM(face, projectpath="intermediate"):
    generator, kp_detector = load_checkpoints(config_path='./LIHQ/first_order_model/config/vox-256.yaml', checkpoint_path='./LIHQ/first_order_model/vox-cpk.pth.tar')
    
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