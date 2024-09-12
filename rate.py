import os
import torch.nn as nn
import torch

cos = nn.CosineSimilarity(dim=0)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
gen_path = "/data/feature/gen/"
train_path = "/data/feature/train/"
rate_path = "/data/rate/"

file = os.listdir(train_path)
for i in file:
    with open(gen_path+i, "r") as f:
        f_gen = f.readlines()
    with open(train_path+i, "r") as f:
        f_trn = f.readlines()
    rate = []
    print(i)
    for g1 in f_gen:
        k1 = torch.tensor([float(j) for j in g1.split()]).to(device)
        #print(g1)
        best = 0
        for g2 in f_trn:
            k2 = torch.tensor([float(j) for j in g2.split()]).to(device)
            c = cos(k1,k2).item()
            img_cos = (c+1)/2
            #print(img_cos)
            if img_cos > best:
                best = img_cos
        rate.append(str(best)+"\n")
    with open(rate_path+i, "w") as f:
        f.writelines(rate)