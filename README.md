# AudioSignalProcessing
### Time Audio Features
In the 'TimeAudioFeatures' directory, I have extracted 3 time based features from scratch (Used Librosa only for loading the audio files). The features extracted are:
1. Amplitude Envelope (AE) -> The maximum amplitude of all the samples present in a frame of size FRAME_SIZE = 1024. The frames are overlapping to make sure none of the information is lost because of hopping of frames of size = HOP_SIZE = 512.<br/> Amplitude Envelope for a frame t of FRAME_SIZE = K can be calculated as:<br/>
<p align="center"> $AE_{t}  =  \sum_{k = t.K}^{(t+1).K-1} s(k)$ </p> <br/>
2. Root Mean Squared Energy (RMSE) -> Used instead of AE as it is less sensitive to outliers. The RMSE for a frmae t of size FRAME_SIZE = K can be calculated as:<br/>
<p align="center"> $RMS_{t} = \sqrt{\frac{\sum_{k = t.K}^{(t+1).K-1} s(k)^{2}}K}$ </p><br/>
3. Zero Crossing Rate (ZCR) -> The number of times a signal crosses the horizontal axis. The ZCR for a frame t of size FRAME_SIZE = K can be calculated as: <br/>
<p align="center"> ZCR_{t} = $\frac{\sum_{k=t.K}^{(t+1).K-1}\left| sgn(s(k))-sgn(s(k+1))\right|}{2}$ </p><br/>

### Fourier Transform
In the 'FourierTransform' directory, I have taken the Fourier Transformer of some signals from different music instruments. The Fourier Trnasform was taken using the np.fft.fft() function which takes the Fast Fourier Transform (O(n*logn)). Then, I have taken the Inverse Fourier Transfroms of the Fourier Transform and compared the absolute value of the original signal to validate that information is preserved in these transforms. 
Mathematically, the Fourier Transform is represented as: <br/>
<p align="center">$\hat{x}(\frac{k}{N}) = \sum_{n=0}^{N-1} x(n).e^{-i.2\pi n.\frac{k}{N}}$</p><br/>
