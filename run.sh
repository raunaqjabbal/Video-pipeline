# cd Video-pipeline
pip install -q -r requirements.txt

pip install -q git+https://github.com/suno-ai/bark.git


cd Real-ESRGAN
pip install .
cd ..

# Face Crop
gdown 1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx -O './LIHQ/procedures/face_align/cache'


cd LIHQ
pip install -r requirements.txt

## ESRGAN
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -O "Real-ESRGAN/weights/realesr-animevideov3.pth"
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth     -O "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"

## GFPGAN
wget ttps://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" -O "GFPGAN/weights/parsing_parsenet.pth"
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" -O "GFPGAN/weights/detection_Resnet50_Final.pth"

## FOMM

cd first_order_model
# getting model weights
gdown 1DbjXD2nS3jlyCWoJu2HGcLZZjhLC9a2J
cd ..
cd ..

#Wav2Lip
cd LIHQ
cd Wav2Lip
## Downloading model weights
gdown 1eAtM-Ck5RMyMMZoQuoQfYZRU5vDDwBpK -O './checkpoints/wav2lip.pth'
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "./face_detection/detection/sfd/s3fd.pth"
cd ..
cd ..

##MODNET
cd LIHQ/MODNet
gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O pretrained/modnet_photographic_portrait_matting.ckpt

cd ..
cd ..

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export PATH=$PATH:"$SCRIPT_DIR/LIHQ/first-order-model":"$SCRIPT_DIR/LIHQ/procedures"


export SUNO_OFFLOAD_CPU=True
export SUNO_USE_SMALL_MODELS=True

python3 script.py