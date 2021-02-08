---
title: 'Masks in sound source separation: An ablation'
date: 2021-02-08
permalink: /posts/2021/02/08/bss-masking/
tags:
  - complex mask
  - binary mask
---
In this blogpost I explain how masking works in sound source separation. It adresses binary mask and complex masks. 
An ablation study on their performance for the two-sources case is carried out.  

## Mix-and-separate. Artificially creating mixtures to train source separation algorithms.  
The deep learning algorithms are the state-of-the art for many tasks nowadays. One of this tasks is blind source separation. This is, given a mixture of signals, recovering 
the signals this mixture os composed by. From an audio perspective, this problem is called sound source separation (for generical sounds), cocktail party (the sources are
a ser of speakers talking), speech enhancement etcetera...

There are roughly two main ways of adressing these problems. On the one hand, we can work on the temporal domain and treating audio as 1D signals. On the other hand we can 
apply a transformation to work on the time-frequency space.  
One of the most commontly used transformations is the Short-time Fourier transform (STFT). It is widely used since it's invertible and its computation can be parallelized using GPUs.
A really short and interesting blogpost to learn more about STFT and speech is [Analyzing Speech Signals in Time and Frequency](https://bastibe.de/2019-09-20-analyzing-speech-signals-in-time-and-frequency.html).

An STFT for speech would look like this:
![](https://bastibe.de/static/2019-09/stft.png)

An awesome property of our way of representing audio as waveforms is they are linear, thus, a mixture of waveforms is nothing but a sum of waveforms. Since STFT is linear,
this properly can be exploided in this domain too. We can take advantage of the linearity to easily generate a a dataset with ground-truth signals to train DNN. We just have to sum some waveforms
and we will have a mixture and its separated sources.  

## Masks and DNN.  
Generating a whole signal from the scratch is a difficult task even for DNN. We can denote a spectrogram as $$S[t,f]$$. A mixture of sources can be defined as
$$S_{mix}[t,f]=\sum S_i[t,f]$$. One way of easing the source separation task is training
DNN to choose whether a time-frequency point belongs to one source or another. This is equivalent to define a mask $$M[t,f]$$ such that $$M_i \cdot S_{mix} = S_i$$ (where the operator is
an element-wise multiplication). Which kind of masks one may propose?  

There are several types of masks and losses which can be used. You can find more info in this study [Michaelsanti et al. 2018](https://arxiv.org/pdf/1811.06234.pdf)
Here I'm going to adress binary masks and complex masks.  
## Binary mask  

