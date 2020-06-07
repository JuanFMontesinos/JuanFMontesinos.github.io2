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

Nowadays, most state-of-the-art works in the audio field uses Time-Frequency representations of audio signals. This is benefitial as it is possible to take advantage of all the work already done in Computer Vision, which treats with images (2D arrays). Depending on the application we can find different transforms as [MFC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) or [CQT](https://en.wikipedia.org/wiki/Constant-Q_transform). It is not really clear which one is a more powerful representation for DL methods. However, STFT is widely used since it is a linear transform.  

We are going to address how to perform STFT and 
