cd Video-pipeline
pip install -q -r requirements.txt


pip install -q git+https://github.com/suno-ai/bark.git


cd Real-ESRGAN
pip install -qr requirements.txt
pip install -q basicsr facexlib gfpgan ffmpeg-python
pip install .
cd ..

##FOMM

cd LIHQ
pip install -r requirements.txt

cd first_order_model
# getting model weights
gdown 1DbjXD2nS3jlyCWoJu2HGcLZZjhLC9a2J
cd ..
cd ..

#Wav2Lip
cd LIHQ
cd Wav2Lip
## Downloading model weights
gdown 1eAtM-Ck5RMyMMZoQuoQfYZRU5vDDwBpK -O './checkpoints/wav2lip_gan.pth'
gdown 1eAtM-Ck5RMyMMZoQuoQfYZRU5vDDwBpK -O './checkpoints/wav2lip.pth'
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "./face_detection/detection/sfd/s3fd.pth"
cd ..
cd ..

##MODNET
cd LIHQ/MODNet
gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O pretrained/modnet_photographic_portrait_matting.ckpt

cd ..
cd ..

export PATH=$PATH:/LIHQ/first-order-model
export PATH=$PATH:/LIHQ/procedures

python3 script.py