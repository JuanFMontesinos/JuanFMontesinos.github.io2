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
