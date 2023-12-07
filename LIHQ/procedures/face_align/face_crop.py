import torchvision
from .shape_predictor import align_face
import dlib
import gdown
import os
######################################
# Stolen (and modified) from PULSE face depixelization
# https://github.com/adamian98/pulse
# Who stole it from someone else

def crop_face(filename, outfile):
    #downloading model weights
    
    # Download the file from Google Drive
    predictor = dlib.shape_predictor("./LIHQ/procedures/face_align/cache")
    

    # f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir="cache", return_path=True)
    # predictor = dlib.shape_predictor(f)

    toPIL = torchvision.transforms.ToPILImage()
    toTensor = torchvision.transforms.ToTensor()

    images = []
    faces = align_face(filename,predictor)
    face = faces[0]
    face = toPIL(toTensor(face).unsqueeze(0).cuda().cpu().detach().clamp(0,1)[0])
    images.append(face)
    face.save(outfile)

    if(len(images)==0): raise Exception("No faces found. Try again with a different image.")