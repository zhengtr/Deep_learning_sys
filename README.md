# CSCI-GA 3033-091 (Deep Learning System, Spring 2022) Final Project
## Semi-Supervised Image-to-Image Traslation

Created by *Tanran Zheng* and *Yuxiang Zhu*.

Courant Institute of Mathematical Sciences - New York University
# Introduction
#TODO: briefly introduce our project in a few sentences
# Demo

#TODO: show demo visualization

# Setup
Two parts: (1) pre-train a ResNet-50 backbone with *SwAV* model using unlabeled data, and (2) train a U-Net conditional GAN or CycleGAN for the downstream task. 

## Part 1
Train ResNet-50 Backbone using SwAV or Simsiam

**Data**  
Unlabled Data

### Module:  swav_ddp
Train swav with multiple GPU on single node  

**Scripts**  
`swav_ddp.py`: contains the main function to run the implementation  
`swav_ddp.slurm`: run script on NYU Greene HPC

**To run**  

Run
```
python -u swav_ddp.py \
--workers 2 \
--epochs 100 \
--batch_size 32 \
--epsilon 0.05 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3072 \
--nmb_prototypes 2000 \
--epoch_queue_starts 30 \
--sync_bn pytorch \
--syncbn_process_group_size 4 \
--world_size 1 \
--rank 0 \
--arch resnet50
```
or run

`swav_ddp.slurm`: slurm file on NYU Greene hpc.

### Module:  swav_simplified
Train swav with single GPU on single node  

**Scripts**  
`main_swav.py`: contains the main function to run the implementation  
`resnet50.py`: modified file contains impl of ResNet architecture 

**To run**  

Run
```
python -u main_swav.py \
--epochs 100 \
--queue_length 3840 \
--batch_size 128 \
--base_lr 0.01 \
--final_lr 0.00001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet50 \
--freeze_prototypes_niters 5000 \
--checkpoint_freq 3 \
--data_path <path\to\data> \
--nmb_prototypes 3000\
```

### Module:  simsiam_ddp
Train SimSiam with multiple GPU on single node  

**Scripts**  
`simsiam_ddp.py`: contains the main function to run the implementation  
`simsiam_ddp.slurm`: run script on NYU Greene HPC

**To run**  

Run
```
python -u simsiam_ddp.py \
--dist-url "tcp://127.0.0.1:50000" \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--arch resnet50
```
or run

`simsiam_ddp.slurm`: slurm file on NYU Greene hpc.

## Part 2
Train Image-to-Image Downstream task

**Data**  
Data from COCO dataset

### Module:  fintune (#TODO: change name to conditional_GAN)
**Scripts**  
`main.py`: train conditional GAN model
`load_swav.py`: contains class and function needed to load swav pretrained backbone checkpoints

**To run**  

Run
```
#TODO
```


## Others
#TODO

# Major reference

### SwAV: 
https://github.com/facebookresearch/swav
```
@article{DBLP:journals/corr/abs-2006-09882,
  author    = {Mathilde Caron and
               Ishan Misra and
               Julien Mairal and
               Priya Goyal and
               Piotr Bojanowski and
               Armand Joulin},
  title     = {Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  journal   = {CoRR},
  volume    = {abs/2006.09882},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.09882},
  eprinttype = {arXiv},
}
```
### SimSiam: 
https://github.com/facebookresearch/simsiam  
```
@Article{chen2020simsiam,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```

### Conditional GAN:  
https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8

### CycleGAN:  
https://junyanz.github.io/CycleGAN/

```
@article{CycleGAN2017,
  author    = {Jun{-}Yan Zhu and
               Taesung Park and
               Phillip Isola and
               Alexei A. Efros},
  title     = {Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
               Networks},
  journal   = {CoRR},
  volume    = {abs/1703.10593},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.10593},
  eprinttype = {arXiv},
  eprint    = {1703.10593},
  timestamp = {Mon, 13 Aug 2018 16:48:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/ZhuPIE17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
