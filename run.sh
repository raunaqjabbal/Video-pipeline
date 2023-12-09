# cd Video-pipeline
pip install  -r requirements.txt

pip install git+https://github.com/suno-ai/bark.git

# cd Real-ESRGAN
pip install Real-ESRGAN/.
# cd ..

# Face Crop
gdown 1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx -O 'LIHQ/cache'


## ESRGAN
wget "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth" -O "Real-ESRGAN/weights/realesr-animevideov3.pth"
wget "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -O "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"

## GFPGAN
wget "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"  -O "gfpgan/gfpgan/weights/GFPGANv1.3.pth"
wget "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" -O "gfpgan/weights/parsing_parsenet.pth"
wget "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" -O "gfpgan/weights/detection_Resnet50_Final.pth"

## FOMM
gdown 1DbjXD2nS3jlyCWoJu2HGcLZZjhLC9a2J -O 'LIHQ/first_order_model/vox-cpk.pth.tar'

#Wav2Lip
gdown 1eAtM-Ck5RMyMMZoQuoQfYZRU5vDDwBpK -O 'LIHQ/Wav2Lip/checkpoints/wav2lip.pth'
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "LIHQ/Wav2Lip/face_detection/detection/sfd/s3fd.pth"

##MODNET
gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O LIHQ/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt


# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export PATH=$PATH:"$SCRIPT_DIR/LIHQ/first-order-model":"$SCRIPT_DIR/LIHQ/procedures"

export SUNO_OFFLOAD_CPU=True
export SUNO_USE_SMALL_MODELS=True

python3 script.py