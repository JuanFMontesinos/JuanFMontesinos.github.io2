---
title: 'Audio Preprocessing'
date: 2020-06-04
permalink: /posts/2020/05/25/audio-preprocessing/
tags:
  - machine learning
  - audio
  - preprocessing
---
Let's show how to preprocess audio data to be used in Deep Learning.
To do so we are going to use two very standard libraries. `numpy` and `librosa`.

Nowadays, most state-of-the-art works in the audio field uses Time-Frequency representations of audio signals. This is benefitial as it is possible to take advantage of all the work already done in Computer Vision, which treats with images (2D arrays). Depending on the application we can find different transforms as [MFC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) or [CQT](https://en.wikipedia.org/wiki/Constant-Q_transform). It is not really clear which one is a more powerful representation for DL methods. However, STFT is widely used since it is a [linear transformation](https://en.wikipedia.org/wiki/Linear_map).  

Sounds are ideally linear, this means that, given a mixture of sounds $S_m$ it can be decompose on the sum of each source: $s_m = \sum s_i$. This allows to ease task such as noise supression, in which you consider that a sound source is the mixture of the main source + noise; source separation, in which you consider the mixture of several sound sources and any type of sound modification, since it is a matter of adding or substracting values to a waveform. The fact STFT preserves linearity means that all these properties can be exploited in the frequency representation too. 
