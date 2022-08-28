---
title: 'Always inspect your data: The Grid Audio-Visual Speech Corpus'
date: 2022-08-28
permalink: /posts/2022/85/28/the-grid-corpus/
tags:
  - machine learning
  - data
  - grid
---

Inspecting the training data and the results is usually underrated task which can help us to understand how 
our models works. Today I will show how apparently bad quality samples generated for lip reading revealed issues
with the [The Grid Audio-Visual Speech Corpus](https://zenodo.org/record/3625687) dataset.  

I'm currently working in speech generation from lip motion. For the initial tests we were using the GRID dataset, formally known as the [The Grid Audio-Visual Speech Corpus](https://zenodo.org/record/3625687).  
From their website:  
> The Grid Corpus is a large multitalker audiovisual sentence corpus designed to support joint computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female), for a total of 34000 sentences. Sentences are of the form "put red at G9 now.

### Machine learning and the importance of inspecting the data  
Inspecting and understanding the data is hard. It takes time. It usually requires to create specific code to be able to browse through the data, moreover if the data is 
stored in a server. (Who wanne be downloading the files from a server via scp...).  
Lot of times we simply rely on some metrics to assume whether a model or an idea works or not. Let me show you a very funny case where a data issue helped to understand
how the different proposed work and their advantages and disadvantages.  



# Case study: s34@srwfzs  
## The Grid Audio-Visual Speech Corpus has unsync videos 
I claim that. The audio and video streams are not always aligned temporally and the sample `srwfzs` from the speaker 34 is one of those, among (at least) some others.  

By training different models to predict the speech corresponding to a given person speaking, we realised there were some missing works, clearly audible misalignments between
the ground truth and the prediction. Lack of words or even repeated syllables.  
## The thread from which to pull  
![](/images/grid_post/srwfzs_hole.png)  
The above image (which corresponds to 1.6 seconds of audio) was a headache. It could be a bug in the code, in the preprocessing pipeline, in the postprocessing or... in the data. As I'm a wonderful programmer it couldn't be my code so I started inspecting the data (jokes).  
And after inspecting the sample we couldn't find a clear misaligment betwen the video and the audio stream. Although it did not feel natural at all.  
After loading the sample in a video editor we realised there indeed was a misaligment problem.  
![](/images/grid_post/srwfzs_video.png)  
As you can observe, the video starts with the person already starting the word, meanwhile the audio starts with a short silence.  
![](/images/grid_post/srwfzs_sync.png)  
After correcting the misaligment the best we could, we could mesure qualitative the offset: 0.26 s, which matches our inspection from the spectrogram (~0.3 s).  
We expected the issue to come from the preprocessing steps. As you can see the images do not correspond to the original dataset. Sadly, after downloading the original sample and inspecting
it, we could track the issue back to the original dataset :(  


## Conclusion  
Inspect your data, your results and be careful with GRID!  
