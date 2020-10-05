---
title: 'Library: MIR Eval library ported to pytorch'
date: 2020-10-05
permalink: /posts/2020/10/05/mir-eval-for-pytorch
tags:
  - machine learning
  - audio
  - mir  
---

I've ported [MIR eval library](https://craffel.github.io/mir_eval/) to PyTorch enabling a faster evaluation.  
It's is available here --> [torch_mir_eval](https://github.com/JuanFMontesinos/torch_mir_eval)  
* Only the Source separation functions has been ported, even though the project is opensource and open to grow.  

# Benchmark  
In order to check the suitability of using a GPU to carry out. To do so we are gonna use the following components  

**CPU**: AMD Ryzen Threadripper 1920X 12-Core Processor  
**GPU**: Quadro P6000  
**Driver Version**: 440.100  
**CUDA Version**: 10.2  
**Pytorch**: 1.6.0  
**numpy**: 1.16.4 

TODO
