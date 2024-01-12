# ABC-Norm Regularization for Fine-Grained and Long-Tailed Image Classification <img src="https://raw.githubusercontent.com/ABC-Norm/ABC-Norm/main/example/ABC.jpg" width=50px> 



PyTorch Implementation for paper [ABC-Norm Regularization for Fine-Grained and Long-Tailed Image Classification](https://ieeexplore.ieee.org/document/10179261), IEEE Transactions on Image Processing.

[Yen-Chi Hsu]() <sup>1,2*</sup>,
[Cheng-Yao Hong](https://r03943158.github.io/) <sup>1*</sup>,
[Ming-Sui Lee](https://www.inm.ntu.edu.tw/en/Departmentmember/Faculty/%E6%9D%8E%E6%98%8E%E7%A9%97-MS-Lee-23127066) <sup>2</sup>,
[Davi Geiger](https://cs.nyu.edu/~geiger/) <sup>3</sup>,
[Tyng-Luh Liu](https://homepage.iis.sinica.edu.tw/pages/liutyng/index_en.html) <sup>1</sup>
<br>
<sup>1</sup>Institute of Information Science, Academia Sinica,  <sup>2</sup>National Taiwan University,  <sup>3</sup>New York University

<sup>*</sup>denotes equal contribution.

## TL;DR abstract

Image classification for real-world applications often involves complicated data distributions such as fine-grained and long-tailed. To address the two challenging issues simultaneously, we propose a new regularization technique that yields an adversarial loss to strengthen the model learning. Specifically, for each training batch, we construct an adaptive batch prediction (ABP) matrix and establish its corresponding adaptive batch confusion norm (ABC-Norm). The ABP matrix is a composition of two parts, including an adaptive component to class-wise encode the imbalanced data distribution, and the other component to batch-wise assess the softmax predictions. The ABC-Norm leads to a norm-based regularization loss, which can be theoretically shown to be an upper bound for an objective function closely related to rank minimization. By coupling with the conventional cross-entropy loss, the ABC-Norm regularization could introduce adaptive classification confusion and thus trigger adversarial learning to improve the effectiveness of model learning. Different from most of state-of-the-art techniques in solving either fine-grained or long-tailed problems, our method is characterized with its simple and efficient design, and most distinctively, provides a unified solution. In the experiments, we compare ABC-Norm with relevant techniques and demonstrate its efficacy on several benchmark datasets, including (CUB-LT, iNaturalist2018); (CUB, CAR, AIR); and (ImageNet-LT), which respectively correspond to the real-world, fine-grained, and long-tailed scenarios.


## Installation 💬
The code is tested with `python=3.8` and `torch=1.10.0+cu111` on an V100 GPU.
```
git clone --recurse-submodules https://github.com/ABC-Norm/ABC-Norm
conda create -n ABC-Norm python=3.8
conda activate ABC-Norm
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib tensorboard scipy opencv-python tqdm tensorboardX configargparse ipdb kornia imageio[ffmpeg]
```

## Data Preparation  ([CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) as example.) ⚡
```
# Data Preparation 
$ mkdir dataset
$ mkdir dataset/cub
$ ln -sT <your-data-path>/dataset dataset/cub
```

## Training 🤔
```
$ CUDA_VISIBLE_DEVICES=0 python run_cub.py --train
```
## Testing 😄
```
$ CUDA_VISIBLE_DEVICES=0 python run_cub.py
```
