from transformers import AutoImageProcessor, AutoModel 
from PIL import Image 
import torch.nn as nn
import os
import numpy as np
import torch
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu") 
processor = AutoImageProcessor.from_pretrained('dinov2-base') 
model = AutoModel.from_pretrained('dinov2-base').to(device)

gen_folder = "/data/generate/"
result_folder = "/data/feature/gen/"

for n in os.listdir(gen_folder):
    temp_path = gen_folder + n
    with open(result_folder+n+".txt", "w") as f:
        for img in sorted(os.listdir(temp_path), key=lambda x: int(x.split(".")[0])):
            print(img)
            re_img = Image.open(os.path.join(temp_path, img))
            with torch.no_grad(): 
                inputs1 = processor(images=re_img, return_tensors="pt").to(device) 
                outputs1 = model(**inputs1) 
                image_features1 = outputs1.last_hidden_state 
                image_features1 = image_features1.mean(dim=1).cpu()
            f.write(" ".join([str(i) for i in image_features1[0].tolist()])+"\n")