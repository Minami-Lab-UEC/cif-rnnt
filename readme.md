# CIF-RNNT

This repository contains the official PyTorch inference codes of "CIF-RNNT: Streaming ASR Via Acoustic Word Embeddings with Continuous Integrate-and-Fire and RNN-Transducers" presented in ICASSP 2024 (https://ieeexplore.ieee.org/document/10448492). 
Please cite [[1](#citation)] in your work when using this code in your experiments.

Contact Teo Wen Shen @ teouuenshen@gmail.com to obtain the trained model. 

Refer [sample.ipynb](sample.ipynb) for usage.

# Installation

This repository itself is not a Python package. Please follow the instructions below to run the codes. 

## Install PyTorch

https://pytorch.org/get-started/previous-versions/

```
# CUDA 11.8
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Install sentencepiece

```
pip install sentencepiece
```

## Install k2

https://k2-fsa.github.io/k2/cuda.html
https://k2-fsa.github.io/k2/installation/from_wheels.html 

```
pip install k2==1.24.4.dev20241030+cuda11.8.torch2.5.0 -f https://k2-fsa.github.io/k2/cuda.html
```

# Citation

[1] Teo, W. S., & Minami, Y. (2024, April). CIF-RNNT: Streaming ASR Via Acoustic Word Embeddings with Continuous Integrate-and-Fire and RNN-Transducers. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 10561-10565). IEEE.
```latex
@inproceedings{teo2024cif,
  title={CIF-RNNT: Streaming ASR Via Acoustic Word Embeddings with Continuous Integrate-and-Fire and RNN-Transducers},
  author={Teo, Wen Shen and Minami, Yasuhiro},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10561--10565},
  year={2024},
  organization={IEEE}
}
```
