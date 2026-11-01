import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import wfdb

np.random.seed(540605336)
os.makedirs("results", exist_ok=True)

# 1. Load MIT-BIH ECG (record 100) from PhysioNet
signals, fields = wfdb.rdsamp("100", pn_dir="mitdb")
fs = fields["fs"]
ecg = signals[:, 0].astype(float)

start_sec = 5
duration_sec = 10
start = int(start_sec * fs)
stop = start + int(duration_sec * fs)

clean_ecg = ecg[start:stop]
clean_ecg = clean_ecg - np.mean(clean_ecg)
clean_ecg = clean_ecg / (np.max(np.abs(clean_ecg)) + 1e-12)
t = np.arange(len(clean_ecg)) / fs

# 2. Add controlled noise
baseline = 0.20 * np.sin(2*np.pi*0.33*t)
powerline = 0.12 * np.sin(2*np.pi*50*t)
random_noise = 0.04 * np.random.randn(len(t))
noisy_ecg = clean_ecg + baseline + powerline + random_noise

# 3. Conventional filtering
b_notch, a_notch = signal.iirnotch(50, 30, fs=fs)
after_notch = signal.filtfilt(b_notch, a_notch, noisy_ecg)

sos = signal.butter(4, [0.5, 40], btype="bandpass", fs=fs, output="sos")
filtered_ecg = signal.sosfiltfilt(sos, after_notch)

# 4. Wavelet denoising
wavelet = "db6"
level = 5
coeffs = pywt.wavedec(noisy_ecg, wavelet, level=level)
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
threshold = sigma * np.sqrt(2*np.log(len(noisy_ecg)))
coeffs_d = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
wavelet_ecg = pywt.waverec(coeffs_d, wavelet)[:len(clean_ecg)]

# 5. Metrics
def snr_db(ref, est):
    return 10*np.log10(np.sum(ref**2)/(np.sum((est-ref)**2)+1e-12))

def mse(ref, est):
    return np.mean((ref-est)**2)

def rmse(ref, est):
    return np.sqrt(mse(ref, est))

rows = []
for name, x in [("Noisy ECG", noisy_ecg),
                ("Digital filtering", filtered_ecg),
                ("Wavelet denoising", wavelet_ecg)]:
    rows.append([name, snr_db(clean_ecg, x), mse(clean_ecg, x), rmse(clean_ecg, x)])

metrics = pd.DataFrame(rows, columns=["Method","SNR_dB","MSE","RMSE"])
metrics["SNR_Improvement_dB"] = metrics["SNR_dB"] - metrics.loc[0, "SNR_dB"]
print(metrics.to_string(index=False))
metrics.to_csv("results/preliminary_metrics.csv", index=False)

# 6. Time-domain plots
fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for a, x, title in zip(
    ax,
    [clean_ecg, noisy_ecg, filtered_ecg, wavelet_ecg],
    ["MIT-BIH Clean ECG Reference",
     "Noisy ECG",
     "Conventional Digital Filtering",
     "Wavelet Denoising"]):
    a.plot(t, x)
    a.set_ylabel("Amplitude")
    a.set_title(title)
    a.grid(True)
ax[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("results/time_domain_comparison.png", dpi=200)

# 7. Frequency-domain plots
def welch(x):
    return signal.welch(x, fs=fs, nperseg=1024)

f, pc = welch(clean_ecg)
_, pn = welch(noisy_ecg)
_, pf = welch(filtered_ecg)
_, pw = welch(wavelet_ecg)

plt.figure(figsize=(10,6))
plt.plot(f, 10*np.log10(pc+1e-12), label="Clean")
plt.plot(f, 10*np.log10(pn+1e-12), label="Noisy")
plt.plot(f, 10*np.log10(pf+1e-12), label="Digital filtering")
plt.plot(f, 10*np.log10(pw+1e-12), label="Wavelet")
plt.axvline(50, linestyle="--", label="50 Hz")
plt.xlim(0, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (dB/Hz)")
plt.title("ECG Frequency-Domain Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/frequency_domain_comparison.png", dpi=200)

# 8. Morphology comparison
seg = (t >= 2) & (t <= 5)
plt.figure(figsize=(11,5))
plt.plot(t[seg], clean_ecg[seg], label="Clean")
plt.plot(t[seg], noisy_ecg[seg], label="Noisy", alpha=0.7)
plt.plot(t[seg], filtered_ecg[seg], label="Digital filtering")
plt.plot(t[seg], wavelet_ecg[seg], label="Wavelet")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("ECG Morphology Comparison (2-5 s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/morphology_comparison.png", dpi=200)
plt.show()

