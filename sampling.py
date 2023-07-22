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
from PIL import Image
import cv2
import numpy as np

def crop_whitespace(img):
    img_gray = img.convert("L")
    img_gray = np.array(img_gray)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)

def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        
    im.save(path)
    return im

def save_single_images(images, path, args, **kwargs):
    #grid = torchvision.utils.make_grid(images, **kwargs)
    image = images.squeeze(0)
    print('images', image.shape)
    
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(image)
        #im = image.permute(1, 2, 0).to('cpu').numpy()
        im = crop_whitespace(im)
        im = Image.fromarray(im)
    else:
        print('no latent')

    im.save(path)
    return im

def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--img_size', type=int, default=(64, 256)) 
    parser.add_argument('--save_path', type=str, default='/path/to/save/generated/images')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--single_image', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=int, default=1)
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--models_path', type=str, default='/path/to/trained/models')
    parser.add_argument('--words', type=list, default=['hello', 'MOVE'])
    
    args = parser.parse_args()
    
    setup_logging(args)
    if args.single_image == True:
        s=[random.randint(0, 339)] #style index for random style or pick a value from 0 to 339 for a specific style
        
        labels = torch.tensor(s).long().to(args.device)
        print('style', labels)
    else:
        print('16 classes')
        labels = torch.arange(16).long().to(args.device)
    words = args.words #produce, greater, music, queer, clearly, edifice, freedom, MOVE, life, sweet, several, months
    
    print('words', words)
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    num_classes = 339 
    vocab_size = 53 
    if args.latent == True:
        unet = UNetModel(image_size = args.img_size, in_channels=4, model_channels=args.emb_dim, out_channels=4, num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=num_classes, context_dim=args.emb_dim, vocab_size=vocab_size, args=args).to(args.device)
    else:
        unet = UNetModel(image_size = args.img_size, in_channels=3, model_channels=128, out_channels=3, num_res_blocks=1, attention_resolutions=(1, 2), num_heads=1, num_classes=num_classes, context_dim=128, vocab_size=vocab_size).to(args.device)
    #unet = nn.DataParallel(unet, device_ids = [0,1,2,3,4]) #,5,6,7])
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    unet.load_state_dict(torch.load(f'{args.models_path}/models/ckpt.pt'))
    optimizer.load_state_dict(torch.load(f'{args.models_path}/models/optim.pt'))
    
    unet.eval()
    
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load(f'{args.models_path}/models/ema_ckpt.pt'))
    #ema_model = ema_model.to(args.device)
    ema_model.eval()
    
    if args.latent==True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    
    for x_text in words:
        if not args.single_image:
            ema_sampled_images = diffusion.sample(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args)
            sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}.jpg"), args)
        else:
            if args.interpolation == True:
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args)
                sampled_ema = save_single_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{args.mix_rate}.png"), args)
            else:
                #for i in range(10):
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args)
                sampled_ema = save_single_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{s[0]}.png"), args)
    
if __name__ == "__main__":
    main()
