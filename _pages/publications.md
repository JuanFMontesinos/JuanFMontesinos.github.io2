---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---
## Real-time visually guided voice separation  
Currently working 
<details>
<summary>Description</summary>
<p>

Audio-visual methods usually outperforms audio-only methods but, can they be used in real-time applications at a reasonable cost?  
Developing SOTA speech enhancement methods with bounded computational power requirements.  
</p>
</details> 

## VoViT: Low Latency Graph-based Audio-Visual Voice Separation Transformer  
Venkatesh S. Kadandale, Juan F. Montesinos, Gloria Haro,  
Under review 
<details>
<summary>Brief description</summary>
<p>
This paper presents an audio-visual approach for voice separation which outperforms state-of-the-
art methods at a low latency in two scenarios: speech and singing voice. The model is based on
a two-stage network. Motion cues are obtained with a lightweight graph convolutional network
that processes face landmarks. Then, both audio and motion features are fed to an audio-visual
transformer which produces a fairly good estimation of the isolated target source. In a second stage,
the predominant voice is enhanced with an audio-only network.
</p>
</details>  

## VocaLiST: An Audio-Visual Synchronisation Model for Lips and Voices    
Venkatesh S. Kadandale, Juan F. Montesinos, Gloria Haro,  
Under review 
<details>
<summary>Brief description</summary>
<p>
we address the problem of lip-voice synchronisation in videos containing human face and voice. Our approach is based on determining if the lips motion and the voice in a video are synchronised or not, depending on their audio-visual correspondence score. We propose an audio-visual cross-modal transformer-based model that outperforms several baseline models in the audio-visual synchronisation task on the standard lip-reading speech benchmark dataset LRS2. While the existing methods focus mainly on the lip synchronisation in speech videos, we also consider the special case of singing voice
</p>
</details>  

## Acappella: Audio-visual singing voice separation  
Juan F. Montesinos, Venkatesh S. Kadandale, Gloria Haro,  
British Machine Vision Conference 2021  
<details>
<summary>Bibtex Citation</summary>
<p>

```
@inproceedings{montesinos2021cappella,
  title={A cappella: Audio-visual Singing Voice Separation},
  author={Montesinos, Juan F and Kadandale, Venkatesh S and Haro, Gloria},
  booktitle={32nd British Machine Vision Conference, BMVC 2021},
  year={2021}
}
```
</p>
</details>  
<details>
<summary>Brief description</summary>
<p>
We explore the single-channel singing voice separation problem from a multimodal perspective, by jointly learning from audio and visual modalities.  
We propose a model which makes use of Graph convolutional neural networks to contrain a U-Net, a encoder-decoder architecture.  
We evaluate:  
  i) presence of overlapping voices in the audio mixtures  
  ii) the target voice set to lower volume levels in the mix  
  iii) combination of i) and ii). The third one being the most challenging evaluation setup. We demonstrate that our model outperforms the baseline models in the singing voice separation task in the most challenging evaluation setup.  
</p>
</details>  

[Project Page](https://ipcv.github.io/Acappella/)  || [Github Code](https://github.com/JuanFMontesinos/Acappella-YNet) || [Arxiv](https://arxiv.org/abs/2104.09946)
* Poster presentation  at BMVC21  
* Poster presentation at Barcelona Deep Learning Symposium  

## Estimating Individual A Cappella Voices in Music Videos with Singing Faces   
Venkatesh S. Kadandale, Juan F. Montesinos, GLoria Haro.  
Sight and Sound Workshop, Computer Vision and Pattern Recognition 2021 (S&S CVPR21)  

[Paper](https://sightsound.org/papers/2021/Venkatesh_Shenoy_Kadandale_Estimating_Individual_A_Cappella_Voices_in_Music_Videos_with_Singing_Faces.pdf)  || [GitHub Code](https://github.com/JuanFMontesinos/Acappella-YNet)  
* Oral presentation  at S&S CVPR21 [YouTube](https://www.youtube.com/watch?v=IEFuj7WGO-c&t=986s&ab_channel=SightandSound)  

## Multi-channel U-Net for Music Source Separation  
Venkatesh S. Kadandale, Juan F. Montesinos, Gloria Haro, Emilia Gómez  
IEEE International Workshop on Multimedia Signal Processing  
[Project Page](https://vskadandale.github.io/multi-channel-unet/)  || [Arxiv](https://arxiv.org/abs/2003.10414)  || [IEEE Arxiv](https://ieeexplore.ieee.org/document/9287108/)  
<details>
<summary>Abstract</summary>
<p>
A fairly straightforward approach for music source separation is to train independent models, wherein each model is dedicated for estimating only a specific source. Training a single model to estimate multiple sources generally does not perform as well as the independent dedicated models. However, Conditioned U-Net (C-U-Net) uses a control mechanism to train a single model for multi-source separation and attempts to achieve a performance comparable to that of the dedicated models. We propose a multi-channel U-Net (M-U-Net) trained using a weighted multi-task loss as an alternative to the C-U-Net. We investigate two weighting strategies for our multi-task loss: 1) Dynamic Weighted Average (DWA), and 2) Energy Based Weighting (EBW). DWA determines the weights by tracking the rate of change of loss of each task during training. EBW aims to neutralize the effect of the training bias arising from the difference in energy levels of each of the sources in a mixture. Our methods provide three-fold advantages compared to C-U-Net: 1) Fewer effective training iterations per epoch, 2) Fewer trainable network parameters (no control parameters), and 3) Faster processing at inference. Our methods achieve performance comparable to that of C-U-Net and the dedicated U-Nets at a much lower training cost.
</p>
</details>  
  
* Oral presentation at IEEE MMSP20 [YouTube](https://www.youtube.com/watch?v=6dtXjOan4Qo)  

## Solos: A Dataset for Audio-Visual Music Analysis  
Juan F. Montesinos, Olga Slizovskaia, Gloria Haro,  
IEEE  International Workshop on Multimedia Signal Processing  
<details>
<summary>Bibtex Citation</summary>
<p>

```
@inproceedings{montesinos2020solos,
    author    = {Juan F. Montesinos and
                 Olga Slizovskaia and
                 Gloria Haro},
    title     = {Solos: A Dataset for Audio-Visual Music Analysis},
    booktitle = {22st {IEEE} International Workshop on Multimedia Signal Processing,
                {MMSP} 2020, Tampere, Finland, September 21-24, 2020},
               
    publisher = {IEEE},
    year      = {2020},

}
```
</p>
</details>  
<details>
<summary>Abstract</summary>
<p>
In this paper, we present a new dataset of music
performance videos which can be used for training machine
learning methods for multiple tasks such as audio-visual blind
source separation and localization, cross-modal correspondences,
cross-modal generation and, in general, any audio-visual self-
supervised task. These videos, gathered from YouTube, consist of
solo musical performances of 13 different instruments. Compared
to previously proposed audio-visual datasets, Solos is cleaner
since a big amount of its recordings are auditions and manually
checked recordings, ensuring there is no background noise nor
effects added in the video post-processing. Besides, it is, up
to the best of our knowledge, the only dataset that contains
the whole set of instruments present in the URMP [1] dataset,
a high-quality dataset of 44 audio-visual recordings of multi-
instrument classical music pieces with individual audio tracks.
URMP was intented to be used for source separation, thus, we
evaluate the performance on the URMP dataset of two different
source-separation models trained on Solo
</p>
</details>  

[Project Page](juanmontesinos.com/Solos)  || [Github Code](https://github.com/JuanFMontesinos/Solos) || [Arxiv](https://arxiv.org/pdf/2006.07931.pdf)
* Poster presentation  at IEEE MMSP20 [YouTube](https://youtu.be/nesxriwTd8Y)    

