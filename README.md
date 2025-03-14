<div align="center">
<h1> SITSMamba for Crop Classification based on <br /> Satellite Image Time Serie</h1>
  
[Xiaolei Qin](https://github.com/XiaoleiQinn), [Xin Su](http://jszy.whu.edu.cn/xinsu_rs/zh_CN/index.htm), [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)

<a href="https://arxiv.org/abs/2409.09673"><img src="https://img.shields.io/badge/arxiv_Paper-%23808EDC"></a>
</div>

## :snake:Highlight
SITSMamba adopts Mamba for Satellite image time series classification for the first time.
<figure>
<div align="center">
<img src=Pics/Architecture.jpg width="90%">
</div>
</figure>

## :ear_of_rice:Dataset
[PASTIS](https://github.com/VSainteuf/pastis-benchmark)<br />
All the 20 classes are used for training and evaluation in our study.
<br />
[MTLCC](https://github.com/TUM-LMF/MTLCC-pytorch)<br />
All the 17 classes are used for training and evaluation in our study.
<br />
Please refer to GitHub - MarcCoru/MTLCC or GitHub - TUM-LMF/MTLCC-pytorch for instructions on loading the dataset and obtaining the eval.tileids.
## :hammer:Environment
You can install mamba according to the following instruction.
```
conda create -n mamba python==3.8
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
wget https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.2cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.2.2+cu122torch2.2cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```



## :blue_heart:Thanks
The ConvBlock is from [UTAE](https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py).
The temporal encoder, Mamba block, is from [mamba](https://github.com/state-spaces/mamba).
