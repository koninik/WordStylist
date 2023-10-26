import torch
import torch.nn as nn
import argparse
import copy
from torch import optim
from train import setup_logging, Diffusion, EMA
from unet import UNetModel
from diffusers import AutoencoderKL
import os
import random
import torchvision
import cv2
import numpy as np


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--img_size", type=int, default=(256, 256))
    parser.add_argument("--save_path", help="Folder for saving generated images")
    parser.add_argument("--latent", action="store_false")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--stable_diffusion_path", default='./stable-diffusion-v1-5', help="Folder of stable-diffusion-v1-5 repo")
    parser.add_argument("--models_path", help="Folder of models")
    parser.add_argument("--style_id", type=int, default=0, help="Index of style")
    parser.add_argument("--text", help="Content for generated images")
    args = parser.parse_args()

    style_id = args.style_id
    style_id = torch.tensor([style_id]).long().to(args.device)
    text = args.text #produce, greater, music, queer, clearly, edifice, freedom, MOVE, life, sweet, several, months

    num_classes = 339 
    vocab_size = 53 
    if args.latent:
        unet = UNetModel(image_size=args.img_size, in_channels=4, model_channels=320, out_channels=4, num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4, num_classes=num_classes, context_dim=320, vocab_size=vocab_size, args=args).to(args.device)
    else:
        unet = UNetModel(image_size=args.img_size, in_channels=3, model_channels=128, out_channels=3, num_res_blocks=1, attention_resolutions=(1, 2), num_heads=1, num_classes=num_classes, context_dim=128, vocab_size=vocab_size).to(args.device)
    #unet = nn.DataParallel(unet, device_ids = [0,1,2,3,4]) #,5,6,7])
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    unet.load_state_dict(torch.load(f'{args.models_path}/models/ckpt.pt', map_location=args.device))
    optimizer.load_state_dict(torch.load(f'{args.models_path}/models/optim.pt', map_location=args.device))
    unet.eval()

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load(f'{args.models_path}/models/ema_ckpt.pt', map_location=args.device))
    #ema_model = ema_model.to(args.device)
    ema_model.eval()
    
    if args.latent:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_diffusion_path, subfolder="vae")
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    # generate images
    diffusion = Diffusion(img_size=args.img_size, args=args)
    img = diffusion.sampling(ema_model, vae, n=len(style_id), x_text=text, labels=style_id, args=args)
    img = img.cpu().numpy()[0].transpose((1, 2, 0))  # CHW -> HWC
    img = img[..., ::-1]  # RGB-> BGR
    cv2.imshow("generated", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
