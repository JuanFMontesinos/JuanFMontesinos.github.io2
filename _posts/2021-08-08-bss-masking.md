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
### Binary mask  
Since spectrograms are defined in the complex domain, the binary masks are defined over the magnitude spectrograms the following way:  
$$M_i[t,f] =1 if |S_i[t,f]|>|S_j[t,f]$$ for $$f \in 1..N$$, 0 otherwise, where N is the amount of sources. In short, if the magnite at a point [T,F] is higher than the magnite at the same point for any other source, then the mask at that point equals 1. Zero otherwise. 
This type of mask assumes each time-frequency point belongs to a single source only.  
### Complex mask  
The complex mask can be defined as $$M[t,f] = S_i[t,f]/S_{mix}[t,f]}$$ where the operator represents the complex division. 
A good paper explaining complex masks and their underlying maths is [Complex Ratio Masking for Monaural SpeechSeparation](http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.2016.pdf)

## Which mask is better? Binary vs Complex, an ablation study  
This is in fact the worthy part of the blog post. In order to compare the reconstruction quality using these masks we will use  the Signal to distorsion ratio (SDR). The higher, the better.  

One interesting point to take into account is that, by definition, the complex mask is better. Why? The complex mask is the ratio between each source and the mixture, therefore it allows to perfectly recover each source. It's a lossless mask. On contrary, the binary mask is a lossy mask. It's possible (but for corner cases) to recover the original source perfectly.  

What we are going to study here is the quality of the reconstructed source when downsampling the spectrogram. This downsampling is a technique used to reduce computations and save memory while training and doing inference. In this ablation, the frequency dimension is downsampled by a factor of 2, such that a mixture of shape F x T is downsampled to F/2 x T.  
### Experiment setup  
The whole experiment is carried out using pytorch (and its fft package and upsampling functions). I tried 3 different upsampling methods, nearest, bilinear and bicubic. I tried interpolating the spectrogram as magnitude and phase and as real and imag. Lastly, I compared between interpolating the mask and then multiplying the interpolated mask by the mixture spectrogram or interpolating the downsampled  predicted spectrogram (downsampling the mixture, multiplying it by the mask and the upsampling).  The audio sample used is a female voice singing acapella mixted with the same voice at a different point of the song. 
### Results
Results are the following
```
mask upsampled using nearest, mask type: complex,
 SDR:18.30, L1:8.814e-03cartesian space
mask upsampled using bicubic, mask type: complex,
 SDR:17.47, L1:1.017e-02cartesian space
mask upsampled using bilinear, mask type: complex,
 SDR:17.44, L1:9.819e-03cartesian space
mask upsampled using bicubic, mask type: binary,
 SDR:14.40, L1:4.167e-02polar space
mask upsampled using bicubic, mask type: binary,
 SDR:14.40, L1:4.167e-02cartesian space
mask upsampled using bilinear, mask type: binary,
 SDR:14.32, L1:4.208e-02polar space
mask upsampled using bilinear, mask type: binary,
 SDR:14.32, L1:4.208e-02cartesian space
mask upsampled using nearest, mask type: binary,
 SDR:14.18, L1:4.179e-02polar space
mask upsampled using nearest, mask type: binary,
 SDR:14.18, L1:4.179e-02cartesian space
mask upsampled using bilinear, mask type: complex,
 SDR:7.62, L1:3.324e-02polar space
mask upsampled using bicubic, mask type: complex,
 SDR:6.52, L1:3.683e-02polar space
mask upsampled using nearest, mask type: complex,
 SDR:6.15, L1:3.786e-02polar space
spectrogram upsampled using nearest, mask type: complex,
 SDR:5.78, L1:6.882e-02cartesian space
spectrogram upsampled using nearest, mask type: binary,
 SDR:4.83, L1:7.444e-02cartesian space
spectrogram upsampled using bilinear, mask type: complex,
 SDR:3.29, L1:7.583e-02cartesian space
spectrogram upsampled using bilinear, mask type: binary,
 SDR:2.74, L1:7.807e-02cartesian space
spectrogram upsampled using bicubic, mask type: complex,
 SDR:2.30, L1:7.532e-02cartesian space
spectrogram upsampled using bicubic, mask type: binary,
 SDR:1.81, L1:7.777e-02cartesian space
spectrogram upsampled using bilinear, mask type: complex,
 SDR:-7.21, L1:8.310e-02polar space
spectrogram upsampled using bicubic, mask type: complex,
 SDR:-7.76, L1:8.443e-02polar space
spectrogram upsampled using nearest, mask type: complex,
 SDR:-8.26, L1:8.884e-02polar space
spectrogram upsampled using bilinear, mask type: binary,
 SDR:-9.72, L1:8.244e-02polar space
spectrogram upsampled using bicubic, mask type: binary,
 SDR:-10.27, L1:8.344e-02polar space
spectrogram upsampled using nearest, mask type: binary,
 SDR:-10.62, L1:8.668e-02polar space

Process finished with exit code 0

```

