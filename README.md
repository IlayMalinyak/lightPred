# LightPred - Deep Learning model for stellar period predicitions
by
Hagai Perets,
Ilay Kamai,

official implementation of
> "New Rotation Period Measurements for Kepler Stars Using Deep Learning: The 100K Sample"
> <br>
[![arXiv](https://img.shields.io/badge/arXiv-2407.06858-B31B1B.svg)](https://arxiv.org/abs/2407.06858)

LightPred is a deep learning model to learn stellar period
using self supervised and simulation based learning. 

![alt text](https://github.com/ilayMalinyak/lightPred/blob/master/images/lightPred.drawio.png?raw=true)
> 
![alt text](https://github.com/ilayMalinyak/lightPred/blob/master/images/period_exp47_scatter.png?raw=true)

## Setup Environment

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/IlayMalinyak/lightPred.git
    cd lightPred
    pip install -r requirements.txt

## Creating Simulated Samples
   to create simualated lightcurves we used **[butterpy](https://github.com/zclaytor/butterpy)** package.
note that we are currently using the deprecated version. this is why we use **butterpy_local** folder.
the script to generate lightcurves is in [dataset\butter.py](https://github.com/IlayMalinyak/lightPred/blob/master/dataset/butter.py)
## Run Experiments

experiments can be found in [experiments](https://github.com/IlayMalinyak/lightPred/tree/master/experiments)
folder.


## Acknowledgements

- the implementation of Astroconf is based on: https://github.com/panjiashu/Astroconformer. we slightly modified the architecture
- we are using implementations from https://github.com/zclaytor/butterpy to simulate lightcurves
- some of the transformations in [transforms](https://github.com/IlayMalinyak/lightPred/tree/master/transforms) are based on https://github.com/mariomorvan/Denoising-Time-Series-Transformer
