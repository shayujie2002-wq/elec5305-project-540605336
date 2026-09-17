# ECG Signal Denoising Using Digital Filtering and Wavelet-Based Methods

**Student:** YUJIE SHA  
**SID:** 540605336  

Repository: https://github.com/shayujie2002-wq/elec5305-project-540605336  
GitHub Pages: https://shayujie2002-wq.github.io/elec5305-project-540605336/

## Feedback Two Progress

This implementation follows the Feedback 1 proposal. The project is implemented in Python and uses ECG recordings from the MIT-BIH Arrhythmia Database via PhysioNet/WFDB.

Current progress:
- Load MIT-BIH ECG data
- Add controlled baseline wander, 50 Hz power-line interference, and random noise
- Apply a 50 Hz notch filter
- Apply a 0.5-40 Hz Butterworth band-pass filter
- Apply preliminary DWT wavelet denoising
- Compare clean, noisy, filtered, and wavelet-denoised ECG signals
- Evaluate SNR, SNR improvement, MSE, and RMSE
- Save time-domain, frequency-domain, and morphology plots

## Run

```bash
pip install numpy scipy matplotlib pandas PyWavelets wfdb
python feedback2_ecg_preliminary.py
```

Internet access is required the first time MIT-BIH record 100 is downloaded.

## Next Steps

The next stage will test additional MIT-BIH recordings and compare different filter orders, cutoff frequencies, wavelet families, decomposition levels, and threshold values. ECG morphology preservation will also be evaluated more systematically.
