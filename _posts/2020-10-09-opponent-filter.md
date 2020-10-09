---
title: 'Paper: Visually Guided Sound Source Separation using Cascaded Opponent Filter Network'
date: 2020-10-09
permalink: /posts/2020/10/09/opponent-filter
tags:
  - machine learning
  - audiovisual
  - bss
  - source separation
---
[PDF](https://arxiv.org/pdf/2006.03028.pdf)  
**Year**: 2020  
**Goal**: Audio-visual sound source separation







Note: Parts of the paper may be quoted without indication.  

# Paper Resume: Visually Guided Sound Source Separation using Cascaded Opponent Filter Network  
## Contributions:   
* Opponent Filters (transferring sound components in a cascade )
* Refined sound source localization

Given a mixture of sounds to separate. The objective of the work is to recover the original sound sources. 
To do so, authors propose a cascade network in 1 + N stages.  
![img](/images/papers/opponent_filter.png)  

The first stage consist of a late-fusion model, a U-Net generates audio features which are constrained by visual information (videos/frames/OF). The novelty comes here:  
In the second (and further) stages, an **Opponent Filter** module transfer sound components between sources. The output is whether passed to the next stage or used as final output. 

## Visual Representation  
Previous works shows that late fusion is usually not capable to extract motion information to improve source separation. However it's interesting to see how does it work following a cascade separation.  
  
The authors try the following inputs:
- Single RGB frame.  
- Dynamic images [CVPR PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bilen_Dynamic_Image_Networks_CVPR_2016_paper.pdf) which summarises the appearence and motion into a single RGB image.  
- RGB video sequence (process through 3D CNN)  
- Optical Flow video sequence (process through 3D CNN)  
- Two-streams modules of pairs of the previous inputs.  

## Audio Representation  
They use STFT representations of the sound, predicting binary masks. In order to generate features and produce the separation they use U-Net.  
The U-Net is used both in the first stage and in the N subsequent stages.  
## The opponent filter  
![img](/images/papers/opponent_filter_arch.PNG)  
The input is always a spectrogram.  
Audio components are extracted by U-Net.  
For each audio in the mixture there should be associated video features. Instead of pairing each video with its own audio (like a contrainer over the spectrogram) they pair each video with the different opposite source (for 2 sources). This way they obtain the components of the audio which belongs to the other source.  
Then they just update the predictions and return the output or pass this to the next stage.

## Sound Source Location Masking Network  
The results for source localization talk by them selves.  It seems they are capable to predict an accurate region (which respects the shape of different instruments).  On contrary, previous works usually create a gaussian probability map centered at the zone of motion (moving hand to play string instruments for example).  
![img](/images/papers/localization_opponent_filters.PNG)  
To do so, they have an auxiliary network which estimate a location mask which is applied over the input, the **Sound Source Location Masking Network** (SSLM)  
$$\mathcal{L}=\sum^J\left\| \widehat{M}_j^{SSLM}-\widehat{M_j }\right\|_1+\frac{\lambda}{q}\left\| SSLM(I) \right\|_1$$  
The first term of the loss minimizes the different between the output of the network with and without using this module. 
The second term regularizes the generated mask, reducing the amount of non-zero values.  

