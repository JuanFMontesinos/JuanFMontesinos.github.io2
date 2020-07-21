---
title: 'Tutorial: Loading AudioVisual Content with Nvidia DALI '
date: 2020-07-21
permalink: /posts/2020/07/21/audiovisual-dali
tags:
  - nvidia
  - DALI
  - audiovisual
  -data loading
---
# Tutorial: Loading AudioVisual Content with Nvidia DALI  

Nvidia [DALI](https://developer.nvidia.com/DALI) is a high performance dataloader developed by Nvidia. 
Nvidia DALI is framework agnostic, however it contains several features for PyTorch, Tensorflow, Caffe and other 
DL frameworks. It also has a (bit complicted) Python API.  

In this post I'm going to show how to load synchronized audio and video stream to work with audiovisual content. 
DALI is highly optimized to load 3 main types of data: audio, video and images. It is possible to read data from 
TFRecord as well as numpy arrays.

Official tutorials can be found [here](https://github.com/NVIDIA/DALI/tree/master/docs/examples)  

This library is relatively easy to use to carry out the main tasks of deep learning: classification, segmentation and 
some other. You can basically drop a directory (following the specific format) on it, and the performance will be very good.
However it becomes tricky at the time of loading very detailed information. For example, specific frames from a video stream,
audio given timestamps...  

##  Loading synchronized audiovisual content  
The "drawback" of DALI is it's been created to load batches of data. Therefore, manipulating which exact file is loaded to
conform a batch is not straightforward.  
### Data format  
DALI makes use of hardware accelerated video decoding. This means it is really fast loading video as it is done with GPU on 
contrary to any other Python library nowadays.  

**Restrictions**  
* It only support the codec `H264`, which means video files have to be of format `.mp4` or `.mov`. 
* It also is restricted to *constant framerate* 

**Advantages** 
* Seek by frame index and to choose how many frames to load
* Very accurate at the time of loading video for DL purposes. 
* GPU accelerated  

With respect to the audio, there is no way too seek for specific stamps. We will load audio manually, not making use of
DALI tools, however audio is light to be loaded wrt video, thus, it's still a big improvement.  

## Installing DALI  
You can check the official installation [docs](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)

As you may imagine, loading video makes sense only if it's possible to resize it. As of today, stable versions only allow 
to load video (but not resize it). That's why I recommend to install a nightly version.
This tutorial is made with the version `0.25`.  
For CUDA 10: 
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
```
For CUDA 11: 
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

## Step by step with DALI  
The most important class in DALI is `Pipeline`
```
from nvidia.dali.pipeline import Pipeline
help(Pipeline)
```

This class is the main class in which we set the workflow, from were to load files, batchsize, number of workers etcetera... 
To import the main tools of DALI:
```
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except:
    raise ImportError('No nvidia DALI found')
```
`Pipeline` would be equivalent to Pytorch's `dataset` class, `DALIGenericIterator` to Pytorch's `Dataloader` and
`ops` to different loaders and transforms.  

With the following code we are going to: 
1. Load a video choosing specific init frames and end frames.
2. Resize the video to 224x224
3. Load an specific audio file seeking 
4. Compute the STFT  


```
class AudioVisualPipe(Pipeline):
    def __init__(self, batch_size: int, num_threads: int, device_id: int, seed: int):
        super(AudioVisualPipe, self).__init__(batch_size, num_threads, device_id, seed=seed,
                                              prefetch_queue_depth=1)
        self.batch_size = batch_size
        self.input = ops.VideoReaderResize(device="gpu", file_list=DST,
                                           sequence_length=10,
                                           shard_id=0, num_shards=1, file_list_frame_num=True,
                                           random_shuffle=False, skip_vfr_check=True,
                                           resize_x=224, resize_y=224)
        self.normalize = ops.Normalize(device='gpu')

        self.spectrogram = ops.Spectrogram(device="gpu",
                                           nfft=1022,
                                           window_length=1022,
                                           window_step=256)
        self.audio_data = ops.ExternalSource()
        # Divide the list in chunks of size=batch_size
        self.idx = 0
        self.audio_files = [
            {'path': AUDIO_PATH, 'offset': 3, 'duration': 1},
            {'path': AUDIO_PATH, 'offset': 5, 'duration': 1},
            {'path': AUDIO_PATH, 'offset': 8, 'duration': 1},
            {'path': AUDIO_PATH, 'offset': 1, 'duration': 1},
        ]

    def define_graph(self):
        video = self.input(name='video')  # This outputs the video, label. That's why we take element 0
        video = self.normalize(video[0])
        audio = self.audio_data(name='audio_main').gpu()
        sp = self.spectrogram(audio)

        return sp, video

    def iter_setup(self):
        # Iter setup is called each time we load a batch
        audio = []
        for _ in range(self.batch_size):
            audio.append(load(sr=None, **self.audio_files[self.idx]))
        self.feed_input(self.audio_main, audio)
        self.idx += 1
```
This is the general structure of a `Pipeline` subclass. Inside the `__init__` function we have to define the DALI 
operations to carry out, but you can define any other information of your concern (such as list of files, transformations,
etcetera...)
Let's analyze everything element by element.  

The `super` args:
```
        super(AudioVisualPipe, self).__init__(batch_size, num_threads, device_id, seed=seed,
                                              prefetch_queue_depth=PREFETCH_QUEUE_DEPTH)
```
`prefetch_queue_depth` defines the length of the queue of the dataloader. This is an important parameter since we will
 be using GPU. Data is stored in the GPU memory (which is usually limited). The higher the value, the faster the system will be
 but we can't prevent memory errors by reducing the size.  
 
 ```
         self.input = ops.VideoReaderResize(device="gpu", file_list=DALI_VIDEO_DATASET_PATH,
                                           sequence_length=N_VIDEO_FRAMES,
                                           shard_id=0, num_shards=1, file_list_frame_num=True,
                                           random_shuffle=False, skip_vfr_check=True,
                                           resize_x=224, resize_y=224)
 ```
 
 This is the video reader. As of today, in order to choose which frames to load, we need to save a list of video paths
 In order this list to be loaded sequentially (and therefore to ensure video is paired with audio) we need to disable 
 random shuffle.
 `self.audio_data = ops.ExternalSource()` indicates that audio data will be loaded without using DALI tools. 
This reader only accept numpy arrays as well as CuPy arrays.  
The format of each line of the `file_list` is:  
`path label frame_init frame_end` of type 
`string int int int`  
For example:  
```
/home/jfm/.local/lib/python3.6/site-packages/skvideo/datasets/data/bigbuckbunny.mp4 0 10 20
/home/jfm/.local/lib/python3.6/site-packages/skvideo/datasets/data/bigbuckbunny.mp4 0 50 60
/home/jfm/.local/lib/python3.6/site-packages/skvideo/datasets/data/bigbuckbunny.mp4 0 60 70
/home/jfm/.local/lib/python3.6/site-packages/skvideo/datasets/data/bigbuckbunny.mp4 0 70 80
```
Label is used for classification, in case it is necessary.  
We created a list of audio with stamps which should paired to the video written in text. Dissabling shuffle in video reader
we can load synchronized audio and video streams. 
 
 `define_graph` states how operators are related. In this case we read the video and resize it using `VideoReaderResize`,
 then we normalize by mean and std using operator `Normalize`. Output of `VideoReaderResize` are `uint8` by default. 
 Applying normalization will convert it into float. 
 Additionally we are loading audio using an external resource and computing its spectrogram.  
 
 
 `iter_setup` is a method called each time we load a batch. There can load the data injected in the pipe by custom code.
 In case we only want to use DALI readers, it's not necessary to define that method. We are basically loading a batch
 of audio (2 samples) using `librosa` and telling the pipe that it should consider that as the audio input. Realize 
 librosa reads audio as numpy arrays.  
 
 
 ## Full running code  
 ```
 from multiprocessing import cpu_count

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except:
    raise ImportError('No nvidia DALI found')

import matplotlib.pyplot as plt
import numpy as np
from librosa import load

import librosa.util
import skvideo.datasets

VIDEO_PATH = skvideo.datasets.bigbuckbunny()  # 132 frames. 132x720x1280x3
AUDIO_PATH = librosa.util.example_audio_file()
DST = './file_list.txt'

BATCH_SIZE = 2
FRAMES = [
    VIDEO_PATH + ' %d %d %d\n' % (0, 10, 20),
    VIDEO_PATH + ' %d %d %d\n' % (0, 50, 60),
    VIDEO_PATH + ' %d %d %d\n' % (0, 60, 70),
    VIDEO_PATH + ' %d %d %d\n' % (0, 70, 80),
]


def write_file_list():
    with open(DST, 'w') as file:
        for line in FRAMES:
            file.write(line)
            print(line)
    return DST


class AudioVisualPipe(Pipeline):
    def __init__(self, batch_size: int, num_threads: int, device_id: int, seed: int):
        super(AudioVisualPipe, self).__init__(batch_size, num_threads, device_id, seed=seed,
                                              prefetch_queue_depth=1)
        self.input = ops.VideoReaderResize(device="gpu", file_list=DST,
                                           sequence_length=10,
                                           shard_id=0, num_shards=1, file_list_frame_num=True,
                                           random_shuffle=False, skip_vfr_check=True,
                                           resize_x=224, resize_y=224)
        self.normalize = ops.Normalize(device='gpu')

        self.spectrogram = ops.Spectrogram(device="gpu",
                                           nfft=1022,
                                           window_length=1022,
                                           window_step=256)
        self.audio_data = ops.ExternalSource()
        # Divide the list in chunks of size=batch_size
        self.idx = 0
        self.audio_files = [
            {'path': AUDIO_PATH, 'offset': 3, 'duration': 1},
            {'path': AUDIO_PATH, 'offset': 5, 'duration': 1},
            {'path': AUDIO_PATH, 'offset': 8, 'duration': 1},
            {'path': AUDIO_PATH, 'offset': 1, 'duration': 1},
        ]

    def define_graph(self):
        video = self.input(name='video')  # This outputs the video, label. That's why we take element 0
        video = self.normalize(video[0])
        self.audio = self.audio_data().gpu()
        sp = self.spectrogram(self.audio)
        return sp, video

    def iter_setup(self):
        print(self.idx)
        # Iter setup is called each time we load a batch
        audio = []
        for idx in range(self.idx, self.idx + BATCH_SIZE):
            audio.append(load(sr=None, **self.audio_files[idx])[0])
        self.feed_input(self.audio, audio)
        self.idx += 1



if __name__ == '__main__':
    write_file_list()
    pipe = AudioVisualPipe(batch_size=BATCH_SIZE, num_threads=cpu_count(), device_id=0, seed=5)
    pipe.build()
    train_loader = DALIGenericIterator([pipe], output_map=['sp', 'video'],size=pipe.epoch_size('video'))
    for sample in train_loader:
        sample = sample[0]
        print(sample['sp'].shape)
        print(sample['video'].shape)
 ```
