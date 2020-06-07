---
title: 'Audio Preprocessing'
date: 2020-06-04
permalink: /posts/2020/05/25/audio-preprocessing/
tags:
  - machine learning
  - audio
  - preprocessing
---
Let's show how to preprocess audio data to be used in Deep Learning.
To do so we are going to use two very standard libraries. `numpy` and `librosa`.

Nowadays, most state-of-the-art works in the audio field use Time-Frequency representations of audio signals. This is benefitial as it is possible to take advantage of all the work already done in Computer Vision, which treats with images (2D arrays). Depending on the application, we can find different transforms as [MFC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) or [CQT](https://en.wikipedia.org/wiki/Constant-Q_transform). It is not really clear which one is the most powerful representation for DL methods and they are chosen depending on the specifict task to solve. STFT is widely used since it is a [linear transformation](https://en.wikipedia.org/wiki/Linear_map).  

Sounds are ideally linear, this means that, given a mixture of sounds $S_m$, it can be decomposed as the sum of each source: $s_m = \sum s_i$. This allows to ease task such as noise supression, in which you consider that a sound source is the mixture of the main source + noise; source separation, in which you consider the mixture of several sound sources and any type of sound modification in general, since it is a matter of adding or substracting values to a waveform. The fact STFT preserves linearity means that all these properties can be exploited in the frequency representation too.  

Let's jump into details. How to perform preprocessing for deep learning architectures?  
## Preprocessing in Machine learning  
Machine learning and deep learning models adapts to data distribution through learnable parameters and bias. Imagine having two images whose content is exactly the same but one is ligther and the other one is darker. The "only" difference in a dummy case would be nothing but an offset, however, the range of values would be totally different.  
Preprocessing and normalization layers (such as the famous [Batch normalization](https://en.wikipedia.org/wiki/Batch_normalization)) allows to regularize inputs making the network more robust against different data, improving convergence and reducing training time.
What's is the issue with audio then? Let's have a quick look

## Preprocessing in audio  
As we aforementioned, most of the DL models work with time-frequency representations of the audio. 
Let's load an audio file 
```
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# Load the waveform from the example track
y, sr = librosa.load(librosa.util.example_audio_file())
y=y[44000*10:44000*20]
librosa.display.waveplot(y, sr=sr)
```
![Waveform](/images/waveform.png)
As we can see, the waveform is bounded between -1 and 1 (stored as floats). Depending on the file format and the library used to read it we may find different rages, being a 16bit unsigned integer a very typical one. 
The Discrete Short-time Fourier Transform  can be computed as:

$$S[t,f] = \sum_{k=0}^{L-1}s[k]w[k-t]e^{-j\pi fk/L}$$

Since $s[k] \in [-1,1]$ it can be easily shown that:
$$    |S[t,f]|  \leq \sum_{k=0}^{L-1}|w[k-t]|$$
i.e., that the  magnitude STFT of an audio signal bounded between [-1,1] is bounded between $[0,\sum |w[k]|]$.

Since fourier transform assumes periodic signals, it is common to use a window which maps values in the extremes to zero. We can plot the window and calculate the upperbound of the magnitude spectrogram which is 1024 for a hanning windown length of 2048.  
![Waveform](/images/win.png)  
Once we have the signal and the window, we can compute the STFT.
It's as simple as using librosa:
```
stft = np.abs(librosa.stft(y))
D = librosa.amplitude_to_db(stft, ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
```
![STFT](/images/stft.png)  
Next we will address the real problem of STFT and audio preprocessing. 
If we analyze the distribution of values we can observe most of the values are zero. To do so we plot the log-histogram of all the values. Realize that log-histogram implies that the first bin is two orders of magnitude bigger than the second bin. Roughly there are 1 Million zeros in the spectrogram, ~10000 cells between 0-5. Around 50 cells >50 and less than 100 cells > 100. However the least amount of time-frequency points contain almost all the energy of the spectrogram.    
![histogram](/images/hist_raw.png)  
This is expected. Most of the time-frequency points contain no energy meanwhile the whole energy is exponentially  distributed across all the spectrogram. However, this distribution is really inefficient for machine learning:
We are forcing the architecture to be able to statistically model an exponential distribution. This means that it has to be able to be sensitive to small differencies in a compressed range (basically as most of the values are small the model have to provide a wide resolution for an small range) but at the same time to be adapted for way higher ranges at higher values. In short, a mess. 

A very simple way to adress these problems is to compute the logarithm of the spectrogram. Since there are zero values and  $$log(0) = -\inf$$ we need to sum an small epsilon $$\log (stft + \epsilon)$$
```
epsilon = 1e-4
stft_log = np.log(stft + epsilon)
plt.hist(stft_log.flatten(),bins=200,log=False)
plt.show()
```
![STFT](/images/hist_norm.png)  
We achieve to great properties by doing this:
- Log-spectrograms follow normal distributions
- We compress the range of values the spectrogram can take, which implies similar resolution at different ranges.  

Code available to run in Colab:
<script src="https://gist.github.com/JuanFMontesinos/0780d80bca94e9fdef64f4dc06d87843.js"></script>
