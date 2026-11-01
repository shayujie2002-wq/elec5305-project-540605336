"""Deterministic ECG denoising study. Run: python experiment.py [--offline]."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as md
import itertools
import json
import platform
import time
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import wfdb

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results'
SEED = 540605336
DEV = ['100', '101']
TEST = ['103', '105', '109', '111']
STARTS = [5, 25, 45]
NOISES = ['baseline', 'powerline', 'white', 'mixed']
SCALES = [0.5, 1.0, 2.0]
METHODS = ['Noisy', 'Digital baseline', 'DWT baseline', 'Tuned DWT',
           'HP notch only', 'HP notch DWT', 'Tuned digital']

def download(offline=False):
    DATA.mkdir(exist_ok=True)
    for record in DEV + TEST:
        for ext in ['hea', 'dat', 'atr']:
            p = DATA / f'{record}.{ext}'
            if p.exists():
                continue
            if offline:
                raise FileNotFoundError(f'Missing cached data: {p}')
            url = f'https://physionet.org/files/mitdb/1.0.0/{record}.{ext}'
            print('Downloading', url, flush=True)
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            p.write_bytes(content)

def load_segment(record, start):
    x, meta = wfdb.rdsamp(str(DATA / record), sampfrom=int(start*360),
                         sampto=int((start+10)*360), channels=[0])
    fs = meta['fs']
    if fs != 360 or not np.isfinite(x).all():
        raise ValueError('Unexpected input data')
    x = x[:, 0] - np.mean(x[:, 0])
    x = x / (np.max(np.abs(x)) + 1e-12)
    return x, fs

def corrupt(x, fs, noise, scale, seed):
    t = np.arange(len(x))/fs
    components = {'baseline': 0.20*np.sin(2*np.pi*0.33*t),
                  'powerline': 0.12*np.sin(2*np.pi*50*t),
                  'white': 0.04*np.random.RandomState(seed).randn(len(x))}
    z = sum(components.values()) if noise == 'mixed' else components[noise]
    return x + scale*z

def dwt(y, wavelet='db6', level=5, alpha=1.0):
    coeffs = pywt.wavedec(y, wavelet, level=level, mode='symmetric')
    sigma = np.median(np.abs(coeffs[-1]))/0.6745
    threshold = alpha*sigma*np.sqrt(2*np.log(len(y)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode='symmetric')[:len(y)]

def notch(y, fs):
    b, a = signal.iirnotch(50, 30, fs=fs)
    return signal.filtfilt(b, a, y)

def digital(y, fs, order=4, low=0.5, high=40):
    sos = signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, notch(y, fs))

def hp_notch(y, fs, low=0.5):
    sos = signal.butter(4, low, btype='highpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, notch(y, fs))

def score(x, y):
    err = np.mean((x-y)**2)
    return {'SNR_dB': 10*np.log10(np.sum(x*x)/(np.sum((x-y)**2)+1e-12)),
            'MSE': err, 'RMSE': np.sqrt(err),
            'Correlation': np.corrcoef(x, y)[0, 1]}

def choose_parameters():
    # Development records only, mixed noise, all scales; two fixed seeds per segment.
    cases = []
    for r, start, scale, rep in itertools.product(DEV, STARTS, SCALES, range(2)):
        x, fs = load_segment(r, start)
        y = corrupt(x, fs, 'mixed', scale, SEED+int(r)*10000+start*10+rep)
        cases.append((x, y, fs))
    searches = {
        'dwt': [dict(wavelet=w, level=l, alpha=a) for w, l, a in
                itertools.product(['db4', 'db6', 'sym8'], [4, 5, 6], [0.25, 0.5, 1.0])],
        'digital': [dict(order=o, low=lo, high=hi) for o, lo, hi in
                    itertools.product([2, 4], [0.3, 0.5, 0.7], [35, 40, 45])],
        'hybrid': [dict(low=lo, wavelet=w, level=l, alpha=a) for lo, w, l, a in
                   itertools.product([0.3, 0.5, 0.7], ['db4', 'db6', 'sym8'], [4, 5, 6], [0.25, 0.5, 1.0])]}
    chosen = {}
    for family, configs in searches.items():
        rows = []
        prepared = {lo: [hp_notch(y, fs, lo) for x, y, fs in cases]
                    for lo in [0.3, 0.5, 0.7]} if family == 'hybrid' else {}
        for cfg in configs:
            scores = []
            for idx, (x, y, fs) in enumerate(cases):
                if family == 'digital': z = digital(y, fs, **cfg)
                elif family == 'dwt': z = dwt(y, **cfg)
                else: z = dwt(prepared[cfg['low']][idx], **{k:v for k,v in cfg.items() if k!='low'})
                scores.append(score(x, z)['SNR_dB'])
            rows.append(dict(**cfg, Mean_dev_SNR_dB=float(np.mean(scores))))
        table = pd.DataFrame(rows).sort_values('Mean_dev_SNR_dB', ascending=False, kind='stable')
        table.to_csv(OUT / f'tuning_{family}.csv', index=False)
        # Recover original Python parameter types rather than pandas row coercions.
        best_idx = int(np.argmax([r['Mean_dev_SNR_dB'] for r in rows]))
        chosen[family] = configs[best_idx]
        print('Selected', family, chosen[family], rows[best_idx]['Mean_dev_SNR_dB'], flush=True)
    (OUT/'selected_parameters.json').write_text(json.dumps(chosen, indent=2))
    return chosen

def methods(y, fs, cfg):
    h = cfg['hybrid']
    return {'Noisy': y,
            'Digital baseline': digital(y, fs),
            'DWT baseline': dwt(y),
            'Tuned DWT': dwt(y, **cfg['dwt']),
            'HP notch only': hp_notch(y, fs, h['low']),
            'HP notch DWT': dwt(hp_notch(y, fs, h['low']), **{k:v for k,v in h.items() if k!='low'}),
            'Tuned digital': digital(y, fs, **cfg['digital'])}

def peak_metrics(x, y, fs):
    # Reference-guided local peak comparison, not a QRS detection accuracy metric.
    p, _ = signal.find_peaks(x, distance=int(.25*fs), prominence=.25*np.ptp(x))
    p = p[(p>fs) & (p<len(x)-fs)]
    if len(p) == 0: return {'Peak_MAE': np.nan, 'Peak_shift_ms': np.nan, 'N_peaks': 0}
    offsets, amps = [], []
    radius = int(.05*fs)
    for idx in p:
        loc = idx-radius+np.argmax(y[idx-radius:idx+radius+1])
        offsets.append(abs(loc-idx)*1000/fs)
        amps.append(abs(y[loc]-x[idx]))
    return {'Peak_MAE': np.mean(amps), 'Peak_shift_ms': np.mean(offsets), 'N_peaks': len(p)}

def reproduce():
    x, fs = load_segment('100', 5)
    y = corrupt(x, fs, 'mixed', 1, SEED)
    rows = []
    for name, z in [('Noisy ECG', y), ('Digital filtering', digital(y, fs)), ('Wavelet denoising', dwt(y))]:
        rows.append(dict(Method=name, **score(x, z)))
    frame = pd.DataFrame(rows)
    frame['SNR_Improvement_dB'] = frame.SNR_dB-frame.SNR_dB.iloc[0]
    expected = np.array([-2.1307458092428186, 11.1753638009054, -0.91282050732706])
    frame['SNR_difference_from_repository'] = frame.SNR_dB-expected
    frame.to_csv(OUT/'baseline_reproduction.csv', index=False)
    if not np.allclose(frame.SNR_dB, expected, atol=1e-6, rtol=0):
        raise AssertionError('Feedback 2 reproduction mismatch')
    return frame

def evaluate(cfg):
    rows = []
    # 4 held-out records x 3 segments x 4 conditions x 3 scales x 3 seeds = 432 cases.
    for r, start, noise, scale, rep in itertools.product(TEST, STARTS, NOISES, SCALES, range(3)):
        x, fs = load_segment(r, start)
        seed = SEED+int(r)*10000+start*10+rep
        y = corrupt(x, fs, noise, scale, seed)
        in_snr = score(x, y)['SNR_dB']
        for name, z in methods(y, fs, cfg).items():
            row = dict(Record=r, Start_s=start, Noise=noise, Scale=scale, Repeat=rep,
                       Method=name, **score(x,z), **peak_metrics(x,z,fs))
            row['SNR_improvement_dB'] = row['SNR_dB']-in_snr
            row['Interior_RMSE'] = np.sqrt(np.mean((x[int(fs):-int(fs)]-z[int(fs):-int(fs)])**2))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'test_metrics.csv', index=False)
    # Average repetitions and segments within record before uncertainty summaries.
    metrics = ['SNR_dB','SNR_improvement_dB','RMSE','Interior_RMSE','Correlation','Peak_MAE','Peak_shift_ms']
    rec = df.groupby(['Record','Noise','Scale','Method'])[metrics].mean().reset_index()
    rec.to_csv(OUT/'test_record_means.csv', index=False)
    rec.groupby(['Noise','Scale','Method'])[metrics].agg(['mean','std']).to_csv(OUT/'test_summary.csv')
    return df

def plots(df, cfg):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    x, fs = load_segment('103', 5)
    y = corrupt(x, fs, 'mixed', 1, SEED+103*10000+5*10)
    z = methods(y, fs, cfg)
    t = np.arange(len(x))/fs
    fig, axes = plt.subplots(4,1,figsize=(9,7),sharex=True,layout='constrained')
    for ax, name in zip(axes,['Noisy','Digital baseline','DWT baseline','HP notch DWT']):
        ax.plot(t,x,color='0.65',lw=.9,label='Reference');ax.plot(t,z[name],lw=.8,label=name)
        ax.legend(loc='upper right',ncols=2,fontsize=8);ax.set_ylabel('Amplitude')
    axes[-1].set_xlabel('Time in segment (s)');fig.savefig(OUT/'waveforms.png',dpi=180);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for name in ['Digital baseline','DWT baseline','HP notch DWT']:
        sel=(t>=2)&(t<=4);axes[0].plot(t[sel],z[name][sel],label=name,lw=1)
        f,p=signal.welch(z[name],fs=fs,nperseg=1024);axes[1].plot(f,10*np.log10(p+1e-12),label=name)
    axes[0].plot(t[sel],x[sel],'k--',label='Reference',lw=.9)
    axes[0].set(xlabel='Time in segment (s)',ylabel='Normalized amplitude')
    axes[1].set(xlim=(0,100),xlabel='Frequency (Hz)',ylabel='PSD (dB/Hz)')
    for ax in axes: ax.legend(fontsize=7)
    fig.savefig(OUT/'morphology_psd.png',dpi=180);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    show=['Digital baseline','DWT baseline','Tuned DWT','HP notch only','HP notch DWT','Tuned digital']
    for name in show:
        sub=df[(df.Method==name)&(df.Noise=='mixed')]
        rec=sub.groupby(['Scale','Record']).SNR_improvement_dB.mean().unstack()
        axes[0].errorbar(rec.index,rec.mean(axis=1),yerr=rec.std(axis=1),marker='o',capsize=3,label=name)
        bynoise=df[(df.Method==name)&(df.Scale==1)].groupby(['Noise','Record']).SNR_improvement_dB.mean().unstack().reindex(NOISES)
        axes[1].plot(NOISES,bynoise.mean(axis=1),marker='o',label=name)
    axes[0].set(xlabel='Noise amplitude multiplier',ylabel='SNR improvement (dB)',xticks=SCALES)
    axes[1].set(xlabel='Noise condition',ylabel='SNR improvement (dB)')
    axes[0].legend(fontsize=7);axes[1].axhline(0,color='0.6',lw=.8)
    fig.savefig(OUT/'performance.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
    f=np.linspace(0,100,4000);b,a=signal.iirnotch(50,30,fs=360)
    _,hn=signal.freqz(b,a,worN=f,fs=360)
    for name,pars in [('Digital baseline',dict(order=4,low=.5,high=40)),('Tuned digital',cfg['digital'])]:
        sos=signal.butter(pars['order'],[pars['low'],pars['high']],btype='bandpass',fs=360,output='sos')
        _,h=signal.sosfreqz(sos,worN=f,fs=360)
        ax.plot(f,20*np.log10(np.maximum(np.abs(hn*h)**2,1e-8)),label=name)
    ax.set(xlabel='Frequency (Hz)',ylabel='Effective gain (dB)',ylim=(-100,3));ax.legend()
    fig.savefig(OUT/'filter_response.png',dpi=180);plt.close(fig)

def benchmark(cfg):
    x,fs=load_segment('103',5);y=corrupt(x,fs,'mixed',1,SEED)
    # End-to-end wrappers include filter design. No data access, metrics or plots.
    functions={'Digital baseline':lambda:digital(y,fs),'DWT baseline':lambda:dwt(y),
               'Tuned DWT':lambda:dwt(y,**cfg['dwt']),
               'HP notch only':lambda:hp_notch(y,fs,cfg['hybrid']['low']),
               'HP notch DWT':lambda:dwt(hp_notch(y,fs,cfg['hybrid']['low']),**{k:v for k,v in cfg['hybrid'].items() if k!='low'}),
               'Tuned digital':lambda:digital(y,fs,**cfg['digital'])}
    rows=[]
    for name,fn in functions.items():
        for _ in range(5): fn()
        samples=[]
        for _ in range(100):
            t=time.perf_counter_ns();fn();samples.append((time.perf_counter_ns()-t)/1e6)
        rows.append(dict(Method=name,Median_ms=np.median(samples),P95_ms=np.percentile(samples,95)))
    pd.DataFrame(rows).to_csv(OUT/'runtime.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True);download(args.offline)
    print(reproduce().to_string(index=False),flush=True)
    cfg=choose_parameters();df=evaluate(cfg);plots(df,cfg);benchmark(cfg)
    versions={p:md.version(p) for p in ['numpy','scipy','pandas','matplotlib','PyWavelets','wfdb']}
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir())}
    (OUT/'run_manifest.json').write_text(json.dumps(dict(seed=SEED,development_records=DEV,test_records=TEST,
        starts_s=STARTS,noise_conditions=NOISES,scales=SCALES,test_repeats=3,
        python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
        versions=versions,data_sha256=manifest),indent=2))
    print('Completed',len(df),'method rows',flush=True)

if __name__=='__main__': main()
