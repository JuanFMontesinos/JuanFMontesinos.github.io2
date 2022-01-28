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
Now we can use pycharm as if we were in local!  

**Extra hints:** The process of saving local changes remotely is not automatic. To enable it we have to go to:  
![](/images/automatic_upload.png)  
and turn `automatic upload` on.  

## Other interesting tool  
### Termius  
Termius is a all-in-one ssh client. It has SFTP protocol with GUI, port forwarding, SSH and snippets. Everything saving username etcetera... It's multidevice (Windows, Linux, Mac, Android...) and synchronises the devices. This simplify a lot to forward a port or to open a console.  
![](/images/termius.png)  

### Tensorboard  
![image](https://user-images.githubusercontent.com/32466310/151636323-451f59c3-5cd0-410f-9d69-74e3aa9c283f.png)  
Tensorboard needs no introduction. What some users doesn't know is we can visualize tensorboard from a remote computer. When we open tensorboard it displays the content in the local address given a port (usually 6006, 6007 and so).  
Port forwarding is ssh protocol which just maps the remote port of our choose to port in our local computer. This way we can just visualize tensorboard from any device with SSH.  

### Streamlit  
[Streamlit](https://streamlit.io/) is an awesome python library wich allows us to create a webapp to make demos or visualize data in a very simple way:  
![image](https://user-images.githubusercontent.com/32466310/151636575-bb9ef9de-de2f-402f-882b-653d8e7ad8ae.png)  
The really nice thing is we can also visualize a streamlit instance opened in a remote computer in our local by forwarding the port!  

## Drawbacks and hints  
This way of working is extremely confortable and feels as if you were connected to the server but...  
**SSH AND SFTP PROTOCOLS ARE HAAARDLY AFFECTED BY NETWORK'S LATENCY.**  
What does this mean? It means no matter how good your connection is, if you are really far away from the server this won't really work. We are talking about being in different continents, like America and Europe. Even if you have a few latency (200 ms) that's enough to mess the whole system.  

Thanks for reading and have fun at home!  

