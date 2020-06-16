---
title: 'Paper: Self-supervised Learning of Interpretable Keypoints from Unlabelled Videos'
date: 2020-06-16
permalink: /posts/2020/06/16/skeletons-weakly
tags:
  - machine learning
  - weakly upervised
  - skeletons
---

 # Paper Resume:  Self-supervised Learning of Interpretable Keypoints from Unlabelled Videos  
 [**PDF**](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jakab_Self-Supervised_Learning_of_Interpretable_Keypoints_From_Unlabelled_Videos_CVPR_2020_paper.pdf)
 **Year**: 2020  
 **Conference**: CVPR  
 **Goal:** Weakly supervised pose detection  
 Note: This post may contain paper quotesç.  
 ## Contribution  
 TODO
 ## Core idea  
During a video an object usually maintains its intrinsic appearance but changes
its pose. Hence, the concept of pose can be learned by modelling the differences between video frames.  

The idea is to formulate this problem as a conditional image generation. On one side, a network extracts pose information discarding appearance.
On the other side, another network reconstructs the image given the pose + a different image with the same appearance.  
![Fig. 1](/images/papers/jakab2020.PNG)  

### Pose bottleneck  
They consider a dual representation of the pose bottleneck, as 2D coordinates and as a pictorial representation of those coords in an image (see Fig. 1).  
The pose bottleneck is controlled via a discriminator learned adversarially. This discriminator probably requires supervision (which is why I define this as weakly supervised instead of self-supervised).
