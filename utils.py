# import torch
from PIL import Image

def tensor2pil(t):
    image_np = t.squeeze().mul(255).clamp(0,255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image
