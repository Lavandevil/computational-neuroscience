"""Inspect EEG project files without loading huge arrays."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import Any
import numpy as np

try:
    from scipy.io import loadmat, whosmat
except ImportError as exc:
    raise SystemExit('Install dependencies: python -m pip install scipy numpy pandas h5py') from exc

try:
    import h5py
except ImportError:
    h5py = None
try:
    import pandas as pd
except ImportError:
    pd = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / 'reports'
MAX_FULL_LOAD_MB = 80
SUPPORTED = {'.mat', '.set', '.fdt', '.csv', '.txt', '.erpm'}

def human_size(n: int) -> str:
    x=float(n)
    for u in ('B','KB','MB','GB','TB'):
        if x<1024 or u=='TB': return f'{x:.2f} {u}'
        x/=1024

def is_hdf5(path: Path) -> bool:
    return bool(h5py and h5py.is_hdf5(path))

def unwrap(v: Any) -> Any:
    for _ in range(8):
        if isinstance(v,np.ndarray) and v.size==1: v=v.reshape(-1)[0]
        else: break
    return v

def field(obj: Any, name: str) -> Any:
    if hasattr(obj,name): return getattr(obj,name)
    if isinstance(obj,np.ndarray) and obj.dtype.names and name in obj.dtype.names: return obj[name]
    return None

def text(v: Any) -> str:
    v=unwrap(v)
    if v is None: return 'not found'
    if isinstance(v,bytes): return v.decode(errors='replace')
    if isinstance(v,str): return v
    a=np.asarray(v)
    if a.size==1: return str(a.reshape(-1)[0])
    return str(a.squeeze().tolist())

def inspect_hdf5(path: Path):
    out=[]
    if not h5py: return 'MATLAB v7.3 / HDF5', ['h5py not installed']
    with h5py.File(path,'r') as f:
        for k in list(f.keys())[:30]:
            o=f[k]
            if isinstance(o,h5py.Dataset):
                out.append(f'- {k}: shape={o.shape}, dtype={o.dtype}')
                try:
                    sl=tuple(slice(0,min(2,d)) for d in o.shape)
                    out.append(f'  sample={np.asarray(o[sl]).reshape(-1)[:8]}')
                except Exception as e: out.append(f'  sample unavailable: {e}')
            else: out.append(f'- {k}: group')
    return 'MATLAB v7.3 / HDF5', out

def inspect_set(path: Path):
    if is_hdf5(path):
        kind,details=inspect_hdf5(path); return 'EEGLAB SET ('+kind+')',details,{}
    details=[]; summary={}
    for name,shape,cls in whosmat(path): details.append(f'- {name}: shape={shape}, class={cls}')
    loaded=loadmat(path,squeeze_me=True,struct_as_record=False,chars_as_strings=True)
    eeg=loaded.get('EEG')
    if eeg is None:
        vals=[v for k,v in loaded.items() if not k.startswith('__')]
        if len(vals)==1: eeg=vals[0]
    if eeg is None: return 'EEGLAB SET',details+['EEG structure not found'],summary
    for key in ('setname','filename','nbchan','srate','pnts','trials','xmin','xmax','data'):
        val=field(eeg,key); s=text(val)
        if key=='data':
            u=unwrap(val)
            if not isinstance(u,str): s=f'embedded numeric data, shape={np.shape(val)}'
            else: s=f'external data reference: {u}'
        details.append(f'- {key}: {s}'); summary[key]=s
    fdt=path.with_suffix('.fdt')
    summary['paired_fdt']=fdt.name if fdt.exists() else 'not found'
    details.append(f'- paired FDT: {summary["paired_fdt"]}')
    return 'EEGLAB dataset header',details,summary

def inspect_mat(path: Path):
    if is_hdf5(path): return inspect_hdf5(path)
    details=[]; vars_=whosmat(path)
    for n,s,c in vars_: details.append(f'- {n}: shape={s}, class={c}')
    if path.stat().st_size/1024**2 <= MAX_FULL_LOAD_MB:
        try:
            loaded=loadmat(path,variable_names=[v[0] for v in vars_],squeeze_me=False,struct_as_record=False)
            for n,_,_ in vars_[:10]:
                a=np.asarray(loaded[n]); details.append(f'  sample {n}: shape={a.shape}, dtype={a.dtype}, values={a.reshape(-1)[:8]}')
        except Exception as e: details.append(f'sample loading failed: {e}')
    else: details.append('large file: metadata only; full array not loaded into RAM')
    return 'MATLAB MAT-file',details

def inspect_csv(path: Path):
    if pd is None:
        return 'CSV text table',path.read_text(encoding='utf-8-sig',errors='replace').splitlines()[:5]
    df=pd.read_csv(path)
    return 'CSV table',[f'shape={df.shape}',f'columns={list(df.columns)}']+df.head(3).to_string(index=False).splitlines()

def inspect_text(path: Path):
    return 'Text/configuration file',path.read_text(encoding='utf-8-sig',errors='replace').splitlines()[:8]

def inspect(path: Path):
    try:
        if path.suffix.lower()=='.set': return inspect_set(path)
        if path.suffix.lower()=='.mat':
            k,d=inspect_mat(path); return k,d,{}
        if path.suffix.lower()=='.fdt':
            pair=path.with_suffix('.set')
            return 'EEGLAB FDT binary signal data',[f'paired SET: {pair.name if pair.exists() else "not found"}','shape must be read from the SET header'],{}
        if path.suffix.lower()=='.csv':
            k,d=inspect_csv(path); return k,d,{}
        k,d=inspect_text(path); return k,d,{}
    except Exception as e:
        return 'Inspection error',[f'{type(e).__name__}: {e}'],{}

def main():
    REPORT_DIR.mkdir(exist_ok=True)
    files=sorted(p for p in PROJECT_ROOT.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED and 'reports' not in p.parts)
    lines=['EEG PROJECT FILE INVENTORY',f'Project root: {PROJECT_ROOT}','='*90]
    rows=[]
    for p in files:
        kind,details,extra=inspect(p)
        rel=p.relative_to(PROJECT_ROOT)
        lines += ['',f'FILE: {rel}',f'SIZE: {human_size(p.stat().st_size)}',f'TYPE: {kind}',*details,'-'*90]
        rows.append({'relative_path':str(rel),'extension':p.suffix.lower(),'size_human':human_size(p.stat().st_size),'detected_type':kind,'nbchan':extra.get('nbchan',''),'srate':extra.get('srate',''),'pnts':extra.get('pnts',''),'trials':extra.get('trials',''),'xmin':extra.get('xmin',''),'xmax':extra.get('xmax',''),'data_storage':extra.get('data',''),'paired_fdt':extra.get('paired_fdt','')})
    txt=REPORT_DIR/'eeg_file_inventory.txt'; csvp=REPORT_DIR/'eeg_file_inventory.csv'
    txt.write_text('\n'.join(lines),encoding='utf-8')
    with csvp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('Done'); print(txt); print(csvp)

if __name__=='__main__':
    sys.exit(main())
