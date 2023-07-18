# Official PyTorch Implementation of "WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models" - ICDAR 2023

<!-- 
[arXiv](https://arxiv.org/pdf/2303.16576.pdf) 
  -->
 <p align='center'>
  <b>
    <a href="https://arxiv.org/pdf/2303.16576.pdf">ArXiv</a>
  </b>
</p> 

 
 <p align="center">
<img src=figs/wordstylist.png width="700"/>
</p>

> **Abstract:** 
>*Text-to-Image synthesis is the task of generating an image according to a specific text description. Generative Adversarial Networks have been considered the standard method for image synthesis virtually since their introduction. Denoising Diffusion Probabilistic Models are recently setting a new baseline, with remarkable results in Text-to-Image synthesis, among other fields. Aside its usefulness per se, it can also be particularly relevant as a tool for data augmentation to aid training models for other document image processing tasks. In this work, we present a latent diffusion-based method for styled text-to-text-content-image generation on word-level. Our proposed method is able to generate realistic word image samples from different writer styles, by using class index styles and text content prompts without the need of adversarial training, writer recognition, or text recognition. We gauge system performance with the Fréchet Inception Distance, writer recognition accuracy, and writer retrieval. We show that the proposed model produces samples that are aesthetically pleasing, help boosting text recognition performance, and get similar writer retrieval score as real data.*


## Dataset & Pre-processing

Download the ```data/words.tgz``` of IAM Handwriting Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.

Then, pre-process the word images by running:
```
python prepare_images.py
```
Before running the ```prepare_images.py``` code make sure you have changed the ```iam_path``` and ```save_dir``` to the corresponding ```data/words.tgz``` and the path to save the processed images.

## Training

To train the diffusion model run:
```
python train.py --iam_path path/to/processed/images --save_path path/to/save/models/and/results
```

## Sampling - Regenerating IAM

## Citation

If you find the code useful for your research, please cite our paper:
```
@article{nikolaidou2023wordstylist,
  title={{WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models}},
  author={Nikolaidou, Konstantina and Retsinas, George and Christlein, Vincent and Seuret, Mathias and Sfikas, Giorgos and Smith, Elisa Barney and Mokayed, Hamam and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2303.16576},
  year={2023}
}
```
