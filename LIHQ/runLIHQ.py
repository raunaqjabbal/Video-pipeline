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

def FOMM_chop_refvid(audio_super, ref_vid):
    i = 0
    for adir in os.listdir(audio_super):
        audio = glob.glob(f'{audio_super}{adir}/*')[0]
        audio_length = librosa.get_duration(filename = audio)
        output_video_path = f'./{audio_super}/{adir}/FOMM.mp4'
        with VideoFileClip(ref_vid) as video:
            if video.duration < audio_length:
                sys.exit('Reference video is shorter than audio. You can:'
                        'Chop audio to multiple folders, reduce video offset,'
                        'use a longer reference video, use shorter audio.')

            new = video.subclip(0, audio_length)
            new.write_videofile(output_video_path, audio_codec='aac')
            i += 1

def FOMM_run(face, audio_super):
    generator, kp_detector = load_checkpoints(config_path='./LIHQ/first_order_model/config/vox-256.yaml', checkpoint_path='./LIHQ/first_order_model/vox-cpk.pth.tar')
    
    for adir in os.listdir(audio_super):
        sub_clip = f'./{audio_super}/{adir}/FOMM.mp4'
            
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

        FOMM_out_path = f'./{audio_super}/{adir}/FOMM.mp4'
        imageio.mimsave(FOMM_out_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        gc.collect()