---
title: 'Paper: End-to-end weakly-supervised semantic alignment'
date: 2020-06-15
permalink: /posts/paper/weakly-supervised
tags:
  - machine learning
  - preprocessing
  - weakly upervised
---
Note: Parts of the paper may be quoted without indication.
# Paper Resume: End-to-end weakly-supervised semantic alignment  
[Arxiv](https://arxiv.org/pdf/1712.06861.pdf)  
**Year**: 2018  
**Goal**:  
> Aigning two mages depicting objects of the same category  

## Contributions:   
* End-to-end CNN 
* Weakly supervised (image pairs)
* differentiable soft inlier scoring module, inspired by the RANSAC  
## Introduction:  
In this work authors study the problem of **finding category-level correspondence**,
or semantic alignment, where the goal is to establish **dense correspondence between different objects belonging to the same category**.
This is also an extremely challenging task because of the
large intra-class variation, changes in viewpoint and presence of background clutter.  

The work is supposed to address previous limitations:
* the image representation and the geometric alignment model are not trained
together in an end-to-end manner.  
* Supervised methods require strong supervision in the form of ground truth correspondences.  

The outcome is that the image representation can be
trained from rich appearance variations present in different
but semantically related image pairs.

![Example](https://camo.githubusercontent.com/315c1bcefc0db56ac1d0d25ffbb5896bcac80fd1/687474703a2f2f7777772e64692e656e732e66722f77696c6c6f772f72657365617263682f7765616b616c69676e2f696d616765732f7465617365722e6a7067)
