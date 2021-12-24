---
title: 'Speech separation with voice identity'
date: 2021-12-24
permalink: /posts/2021/12/24/speech-separation
tags:
  - machine learning
  - voice identification
  - audio
  - mir  
---
We usually claim that audio-visual methods performs better than audio-only in blind sound source separation. We are gonna check the performance of audio-only methods
by doing a simple experiment. To code a U-Net to perform source separation using identiy embeddings. 

## Introduction  
Audio-visual source separation has proven to be sucessful in for [Speech](https://vision.cs.utexas.edu/projects/VisualVoice/), [Singing voice](https://ipcv.github.io/Acappella/),
or [Musical instruments](https://arxiv.org/pdf/1904.05979.pdf). How good is it compared to audio-only source separation?  

## The experiment  
To compare the performance of audio-only vs audio-visual methods we are gonna use [Acappella Dataset](https://ipcv.github.io/Acappella/), a dataset for audio-visual singing
voice separation. There are pretrained models (Y-Net and its variants) and metrics available. Y-Net is an audio-visual sound separation network which uses a U-Net as backbone.
This U-Net is conditioned with either face landmarks processed by a graph CNN or raw video processed by a spatio-temporal CNN. The core idea is the network can use
lips motion to guide source separation leading to nice results.  
![Y-Net](https://ipcv.github.io/Acappella/img/model.png)

On the other side, we are gonna train exactly the same backbone U-Net contrained in identity embeddings. These embeddings summarizes the voice identity. This way,
the U-Net should learn to identify the voice characteristics indicated by the embeddings to carry out the separation.  
These embeddings are extracted with [Resemblyzer](https://github.com/resemble-ai/Resemblyzer), a implementation from [Resemble.ai](https://www.resemble.ai/) of [Generalized end-to-end loss for speaker verification](https://arxiv.org/pdf/1710.10467.pdf).  
The code is available in [GitHub](https://github.com/JuanFMontesinos/AudioIdentitySpeechSeparation).  

