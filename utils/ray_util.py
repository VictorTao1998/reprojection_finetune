import torch

def build_rays(h,w,n):
    xs = torch.randint(0,h,[1,n])
    ys = torch.randint(0,w,[1,n])

    

