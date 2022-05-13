# How to train
python main.py --train *True* --modelPath *your-resnet-model* --epochs *10* 
# How to eval
python main.py --saveImage *True* --train *False* --visualNum 5 --visualFreq 20 --modelPath *your-resnet-model*;
# Colorization Result
The first row shows gray scaled images. The second row is obtained by model colorization. The thrid row shows the orignal images.
![Result1](https://github.com/zhengtr/Deep_learning_sys/blob/main/fintune/colorization_1652408008.7100573.png)
