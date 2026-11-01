"""Check experiment integrity and numeric consistency without rerunning tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import experiment as ex

def main():
    root=Path(__file__).resolve().parent
    df=pd.read_csv(root/'results/test_metrics.csv')
    assert len(df)==3024
    keys=['Record','Start_s','Noise','Scale','Repeat','Method']
    assert not df.duplicated(keys).any()
    assert set(df.Record.astype(str))==set(ex.TEST)
    assert set(ex.DEV).isdisjoint(ex.TEST)
    assert np.isfinite(df.select_dtypes('number').to_numpy()).all()
    assert np.allclose(df.RMSE**2,df.MSE,atol=1e-14)
    pivot=df.pivot(index=keys[:-1],columns='Method',values='SNR_dB')
    increments=df.pivot(index=keys[:-1],columns='Method',values='SNR_improvement_dB')
    for method in ex.METHODS:
        assert np.allclose(pivot[method]-pivot['Noisy'],increments[method])
    cfg=json.loads((root/'results/selected_parameters.json').read_text())
    x,fs=ex.load_segment('103',5)
    y=ex.corrupt(x,fs,'mixed',1,123)
    assert np.array_equal(y,ex.corrupt(x,fs,'mixed',1,123))
    assert not np.array_equal(y,ex.corrupt(x,fs,'mixed',1,124))
    for name,z in ex.methods(y,fs,cfg).items():
        assert z.shape==x.shape and np.isfinite(z).all(),name
    assert np.allclose(ex.dwt(x,alpha=0),x,atol=1e-10)
    baseline=ex.reproduce()
    assert baseline.SNR_difference_from_repository.abs().max()<1e-6
    message='PASS: baseline reproduction, 432 paired cases, 3024 unique rows, metric identities, disjoint records, deterministic noise, reconstruction and output integrity.'
    (root/'results/verification.txt').write_text(message+'\n')
    print(message)

if __name__=='__main__': main()
