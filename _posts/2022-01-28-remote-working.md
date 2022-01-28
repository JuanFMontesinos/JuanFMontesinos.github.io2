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
To configure a remote kernel in Pycharm we do need the PRO version. Remote kernel is not available in the community version.  
We need to go to `File>Settings`. Once in the window go to `Build,Execution, Deployment`
![](/images/deployment.png)  
Here we add a new `SSH configuration` (just a typical ssh connection)  
In the tap `mappings` we relate where the local files are located in the remote server. For example, my project folder in local is `/home/juan/project_example`. In my remote server the analogous folder is `/home/serverAMD500/projects/pycharm/project_example_master`. Note that both folders can have different names.  
Lastly, assume there is a folder which is really heavy (data, examples, etcetera...). We can tell the system to ignore it, like .gitignore file in GIT. This can be done at the tab `Excluded Paths`.  

To run the code in the remote server we need to configure the project to use that kernel. We simply go to `Build, Execution, Deployment > Console > Python Console` and choose the remote kernel. Usually named like `Remote Python 3.6.9 (sftp://servername@10.50.72:22...`. We can add a small script to be executed each time we open a console.  
![](/images/remote_console.png)  
