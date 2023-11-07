import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import configparser
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import json
from diffusers import AutoencoderKL
from unet import UNetModel
import wandb


def create_release(release_folder, size, style_ids, output_max_len, c_classes, letter2index, tokens, vocab_size, stable_diffusion_path):

    height, width = size

    result_ini = configparser.ConfigParser()
    result_ini.optionxform = lambda option: option

    result_ini["style"] = {}
    for style_id in style_ids:
            key = "style_id_" + str(style_id)
            result_ini["style"][key] = str(style_id)

    result_ini["model"] = {}
    result_ini["model"]["width"] = str(width)
    result_ini["model"]["height"] = str(height)

    result_ini["model"]["output_max_len"] = str(output_max_len)
    result_ini["model"]["c_classes"] = c_classes
    result_ini["model"]["vocab_size"] = str(vocab_size)

    result_ini["tokens"] = {}
    for key in tokens:
        result_ini["tokens"][key] = str(tokens[key])

    result_ini["letter2index"] = {}
    for key in letter2index:
        result_ini["letter2index"][key] = str(letter2index[key])

    result_ini["stable_diffusion"] = {}
    result_ini["stable_diffusion"]["stable_diffusion_path"] = stable_diffusion_path

    with open(release_folder + os.sep + "model.ini", 'w') as configfile:
        result_ini.write(configfile)


### Borrowed from GANwriting ###
def label_padding(labels, letter2index, tokens, output_max_len):

    num_tokens = len(tokens.keys())
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    #ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = output_max_len - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


class IAMDataset(Dataset):
    def __init__(self, full_dict, image_path, writer_dict, output_max_len, tokens, letter2index, transforms=None):

        self.data_dict = full_dict
        self.image_path = image_path
        self.writer_dict = writer_dict
    
        self.transforms = transforms
        self.output_max_len = output_max_len
        self.n_samples_per_class = 16
        self.indices = list(full_dict.keys())

        self.letter2index = letter2index
        self.tokens = tokens
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)
        img_path = os.path.join(self.image_path, image_name)
        
        image = cv2.imread(img_path)[..., ::-1]  # BGR->RGB
        image = np.ascontiguousarray(image)
        image = self.transforms(image)
        
        word_embedding = label_padding(label, self.letter2index, self.tokens, self.output_max_len)
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()    
        
        return image, word_embedding, wr_id


class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Diffusion:
    def __init__(self, output_max_len, tokens, letter2index, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), device="cuda:0"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device
        self.output_max_len = output_max_len

        self.letter2index = letter2index
        self.tokens = tokens

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling(self, model, vae, n, x_text, labels, args, mix_rate=None, cfg_scale=3):
        model.eval()
        tensor_list = []
        #if mix_rate is not None:
         #   print('mix rate', mix_rate)
        with torch.no_grad():
            
            words = [x_text]*n
            for word in words:
                transcript = label_padding(word, self.letter2index, self.tokens, self.output_max_len) #self.transform_text(transcript)
                word_embedding = np.array(transcript, dtype="int64")
                word_embedding = torch.from_numpy(word_embedding).long()#float()
                tensor_list.append(word_embedding)
            text_features = torch.stack(tensor_list)
            text_features = text_features.to(args.device)
            
            if args.latent:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                if cfg_scale > 0:
                    # uncond_predicted_noise = model(x, t, text_features, sid)
                    # predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    uncond_predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        if args.latent:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
    
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, num_classes, vocab_size, transforms, args, save_folder):
    model.train()
    
    print('Training started....')
    patience = 0
    loss_min = np.inf
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)

        losses = []
        for i, (images, word, s_id) in enumerate(pbar):
            images = images.to(args.device)
            original_images = images
            text_features = word.to(args.device)
            
            s_id = s_id.to(args.device)
            
            if args.latent == True:
                images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images
            
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            x_t, noise = diffusion.noise_images(images, t)
            
            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = model(x_t, original_images=original_images, timesteps=t, context=text_features, y=s_id, or_images=None)
            
            loss = mse_loss(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            losses.append(loss.item())
        loss_mean = np.mean(losses)
        if loss_mean < loss_min:
            loss_min = loss_mean
            patience = 0

            torch.save(model.state_dict(), save_folder + os.sep + "unet_ckpt.pt")
            torch.save(ema_model.state_dict(), save_folder + os.sep + "ema_ckpt.pt")
            torch.save(optimizer.state_dict(), save_folder + os.sep + "unet_optim.pt")
        else:
            patience += 1
            print(f"----------> current loss: {loss_mean}, minimum loss: {loss_min}, remaining patience: {args.patience - patience}")
            if patience >= args.patience:
                print("Stop training...")
                break


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=256)
    parser.add_argument('--iam_path', type=str, help='path to images')
    parser.add_argument('--gt_train', type=str, help='annotations')
    # string parameters
    parser.add_argument('--c_classes', default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    parser.add_argument('--output_max_len', type=int, default=16, help="max length of output #+ 2  # <GO>+groundtruth+<END>")
    # UNET parameters
    parser.add_argument('--save_path', help=" output folder of the result")
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--latent', action="store_false")
    parser.add_argument('--interpolation', action="store_true")
    parser.add_argument('--stable_diffusion_path', default='./stable-diffusion-v1-5', help='path to stable diffusion')
    args = parser.parse_args()

    save_folder = args.save_path + os.sep + "models"
    os.makedirs(save_folder, exist_ok=True)

    img_size = (args.img_height, args.img_width)

    # vocabulary
    c_classes = args.c_classes
    output_max_len = args.output_max_len
    letter2index = {label: n for n, label in enumerate(c_classes)}

    tok = False
    if not tok:
        tokens = {"PAD_TOKEN": len(c_classes)}
    else:
        tokens = {"GO_TOKEN": len(c_classes), "END_TOKEN": len(c_classes) + 1, "PAD_TOKEN": len(c_classes) + 2}
    num_tokens = len(tokens.keys())
    print('num_tokens', num_tokens)

    num_classes = len(c_classes)
    print('num of character classes', num_classes)
    vocab_size = num_classes + num_tokens
    print('character vocabulary size', vocab_size)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    with open(args.gt_train, "r") as f:
        lines = f.readlines()
        annotations = [line.strip().split(" ") for line in lines]
        full_dict = {}
        writer_dict = {}
        writer_idx = 0
        for idx, annotation in enumerate(annotations):
            style_id, filename = annotation[0].split(",")
            text = annotation[1]
            full_dict[idx] = {"image": filename, "s_id": style_id, "label": text}
            if style_id not in writer_dict.keys():
                writer_dict[style_id] = writer_idx
                writer_idx += 1

        print("number of train writer styles", len(writer_dict))
        num_styles = len(writer_dict)

    train_ds = IAMDataset(full_dict, args.iam_path, writer_dict, output_max_len, tokens, letter2index, transforms=transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    unet = UNetModel(image_size=img_size, in_channels=4, model_channels=320, out_channels=4, num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4, num_classes=num_styles, context_dim=320, vocab_size=vocab_size, args=args, max_seq_len=output_max_len).to(args.device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(output_max_len=output_max_len, tokens=tokens, letter2index=letter2index, img_size=img_size)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    if args.latent:
        print('Latent is true - Working on latent space')
        vae = AutoencoderKL.from_pretrained(args.stable_diffusion_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        print('Latent is false - Working on pixel space')
        vae = None

    train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, num_styles, vocab_size, transforms, args, save_folder)
    # release
    create_release(save_folder, img_size, sorted(writer_dict.keys()), output_max_len, c_classes, letter2index, tokens, vocab_size, os.path.abspath(args.stable_diffusion_path))

if __name__ == "__main__":
    main()
  
  
