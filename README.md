# CSCI-GA 3033-091 (Deep Learning System, Spring 2022) Final Project
## Semi-Supervised Image-to-Image Traslation

Created by *Tanran Zheng* and *Yuxiang Zhu*.

Courant Institute of Mathematical Sciences - New York University


## (Note: This Readme includes only instructions and info about this repo. For detailed reports, see `final_project_report.pdf`)
# Introduction
1. Aming to solve Image-to-Image translation problem with fewer domain specific data.
2. Leveraging self-supervised learning (SSL) to train backbone with unlabled data.
3. Applying transfer learning for downstream tasks.
4. Achieving better result with less data comparing to the baseline.
# Demo
1. Colorization results from different models comparison
![Colorization results](https://github.com/zhengtr/Deep_learning_sys/blob/main/res/Colorization_Result_Compare.png)
2. Changing image style to Monet result
![Change image style](https://github.com/zhengtr/Deep_learning_sys/blob/main/res/Monet.png)

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

### Module:  fintune
**Scripts**  
`main.py`: train conditional GAN model
`load_swav.py`: contains class and function needed to load swav pretrained backbone checkpoints

**To run**  

Run
1. Go to the "*fintune*" folder.
2. Download "*swav_ckp_190.pth*" model file under the "*fintune*" folder.
3. Download you **SwAV** pretrained model here. (For example, it's called "*my_model.pth*").  

To transfer train your model
``` 
python main.py \
--train True \
--modelPath my_new_model.pt \
--epochs 20 \
--givenModel my_model.pth
```
To evaluate your trained model
```
python main.py \
--saveImage True \
--train False \
--visualNum 10 \
--visualFreq 20 \
--givenModel my_model.pth
```

### Module:  cycleGAN
**Scripts**  
`train.py`: train cycleGAN model
`test.py`: evaluate cycleGAN model (put pretrained weight in `checkpoints`)

**To run**  
Put pretrained SwAV in pwd  
then run  

Colorization:
```
python train.py --dataroot ./datasets/mini_colorization --name colorization--model colorization --use_wandb --dataset_mode colorization --input_nc 1 --output_nc 2
```

Monet:
```
python train.py --dataroot ./datasets/monet2photo --name monet2photo --model cycle_gan --use_wandb 
```

**To test**  
put pretrained weight in `checkpoints`
then run  

Colorization:
```
python test.py --dataroot ./datasets/mini_colorization --name colorization--model colorization --dataset_mode colorization --input_nc 1 --output_nc 2
```

Monet:
```
python test.py --dataroot ./datasets/monet2photo --name monet2photo --model cycle_gan
```
## Results
1. Different self-supervised learning model train loss comparison
![Self-supervised learning](https://github.com/zhengtr/Deep_learning_sys/blob/main/res/SwAV_Train_Loss.png)
2. Self-supervised learning training loss between baseline and SwAV  
![Baseline vs SwAV](https://github.com/zhengtr/Deep_learning_sys/blob/main/res/SWAV_COND_GAN_Train_Loss.png)
3. Training loss between SwAV and SimSiam
![SwAV vs SimSiam](https://github.com/zhengtr/Deep_learning_sys/blob/main/res/SWAV_SIMSIAM_Train_Loss_compare.png)

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
