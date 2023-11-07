import torch
import torch.nn as nn
import argparse
import copy

import tqdm
from torch import optim
from train import Diffusion, EMA
from unet import UNetModel
from diffusers import AutoencoderKL
import os
import random
import torchvision
import cv2
import numpy as np
import configparser


def crop_from_padding(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    if (h, w) == img_gray.shape[:2]:
        thresholded = cv2.bitwise_not(thresholded)
        coords = cv2.findNonZero(thresholded)
        x, y, w, h = cv2.boundingRect(coords)
    crop = img[y: y + h, x: x + w, :]
    return crop


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--save_path", help="Folder for saving generated images")
    parser.add_argument("--latent", action="store_false")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--models_path", help="Folder of models")
    parser.add_argument("--style_ids", nargs='+', help="List of style ids")
    parser.add_argument("--texts", nargs='+', help="List of texts")
    args = parser.parse_args()

    unet_ckpt = args.models_path + os.sep + "unet_ckpt.pt"
    unet_optim = args.models_path + os.sep + "unet_optim.pt"
    ema_ckpt = args.models_path + os.sep + "ema_ckpt.pt"

    # Read parameters from ini:
    ini = configparser.ConfigParser()
    ini.optionxform = lambda option: option

    ini.read(args.models_path + os.sep + "model.ini")
    num_styles = len([k for k in ini["style"] if k.startswith("style")])
    img_width = ini.getint("model", "width")
    img_height = ini.getint("model", "height")
    output_max_len = ini.getint("model", "output_max_len")
    c_classes = ini.get("model", "c_classes")
    vocab_size = ini.getint("model", "vocab_size")
    stable_diffusion_path = ini.get("stable_diffusion", "stable_diffusion_path")
    tokens = dict([(key, ini.getint("tokens", key)) for key in ini["tokens"]])
    letter2index = dict([(key, ini.getint("letter2index", key)) for key in ini["letter2index"]])

    save_path = args.save_path
    assert not os.path.exists(save_path), f"{save_path} existed!"
    os.makedirs(save_path)

    style_ids_texts = [(int(float(style_id)), text) for style_id, text in zip(args.style_ids, args.texts)]

    img_size = (img_height, img_width)
    if args.latent:
        unet = UNetModel(image_size=img_size, in_channels=4, model_channels=320, out_channels=4, num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4, num_classes=num_styles, context_dim=320, vocab_size=vocab_size, args=args).to(args.device)
    else:
        unet = UNetModel(image_size=img_size, in_channels=3, model_channels=128, out_channels=3, num_res_blocks=1, attention_resolutions=(1, 2), num_heads=1, num_classes=num_styles, context_dim=128, vocab_size=vocab_size).to(args.device)
    #unet = nn.DataParallel(unet, device_ids = [0,1,2,3,4]) #,5,6,7])
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=args.device))
    optimizer.load_state_dict(torch.load(unet_optim, map_location=args.device))
    unet.eval()

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load(ema_ckpt, map_location=args.device))
    #ema_model = ema_model.to(args.device)
    ema_model.eval()
    
    if args.latent:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae")
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    # generate images
    diffusion = Diffusion(output_max_len=output_max_len, tokens=tokens, letter2index=letter2index, img_size=img_size)
    for i, (style_id, text) in enumerate(tqdm.tqdm(style_ids_texts)):
        img = diffusion.sampling(ema_model, vae, n=1, x_text=text, labels=torch.tensor([style_id]).long().to(args.device), args=args)
        img = img.cpu().numpy()[0].transpose((1, 2, 0))  # CHW -> HWC
        img = (img * 255).astype(np.uint8)
        img = crop_from_padding(img)
        img = img[..., ::-1]  # RGB-> BGR

        # cv2.imshow("generated", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(save_path + os.sep + text + f"_{str(style_id)}_{i}.jpg", img)

if __name__ == "__main__":
    main()
