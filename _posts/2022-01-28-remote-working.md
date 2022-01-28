---
title: 'Optimal Remote working'
date: 2022-01-28
permalink: /posts/2022/01/28/optimal-remote
tags:
  - python  
  - pycharm  
  - remote
---
After the COVID19 pandemic, working on remote/hybrid is starting to be really common. In this post I show some tools to work from remote being data scientist, ML engineer, python developer 
or similar.  

## Coding in python  
There are many ways of writing code in Python. Some people prefer to use VIM and forget about editors. This is one of the most simplest and powerful ways. You require nothing 
but a internet connection and ssh. Another interesting option is to edit files remotely with an enriched text editor. This simply allows to open a remote file in local,
use a the text editor and save the file remotely. Although text editor usually implement highlighted syntax, there are a set of tools we won't be using this way. Next step 
(in complexity) are IDEs. There are dozens of them but the most powerful ones are PyCharm, Visual Studio and Spyder (being the last the less powerful by far).  

### Remote SSH kernel, the right way.  
The most confortable way you can find to work is by using remote kernels.  
One just simply needs two copies of the code to run, one in local and another one in remote. Files are modified locally and then automatically copied to our remote server.
At the time of running the code, we can use all the IDE tools, breakpoints, debugging, variable inspection, enviroment options, argparse...  
In short, it feels as if you were connected to the server directly. The only drawback is you cannot use remote server's GUI tools, for example OpenCV tools to display. However you can
still visualize matplotlib figures and similar ones.  
### Remote Kernel in Pycharm  
