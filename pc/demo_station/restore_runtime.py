"""Restore the verified G1 runtime exclusively inside demo_station/data."""
from pathlib import Path
import shutil
import zipfile
from .core import HOME,ARCHIVE,digest,read_json,write_json

def restore():
    package=HOME/'data/downloads/G1_Windows_AMD_CPU_12FPS_20261018_9b54f3393e8f.zip'
    expected=package.with_suffix('.zip.sha256').read_text().split()[0]
    if digest(package)!=expected:raise ValueError('G1 archive SHA256 mismatch')
    dest=HOME/'data/g1';dest.mkdir(parents=True,exist_ok=True)
    base=dest/'G1_Windows_AMD_CPU';changes=ARCHIVE/'g1_changes'
    if not (base/'manifest.json').exists():
        with zipfile.ZipFile(package) as archive:
            for name in archive.namelist():
                if not (dest/name).resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe archive path')
            archive.extractall(dest)
        for name,sha in read_json(changes/'manifests/original_package.json').items():
            if digest(base/name)!=sha:raise ValueError('G1 base mismatch: '+name)
    for src in (changes/'delta').rglob('*'):
        if src.is_file():
            target=base/src.relative_to(changes/'delta');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    current=read_json(changes/'manifests/current_managed_files.json')
    for name,sha in current.items():
        if digest(base/name)!=sha:raise ValueError('G1 restored mismatch: '+name)
    write_json(base/'manifest.json',current)
    write_json(HOME/'data/g1/restoration.json',dict(files_verified=len(current),archive_sha256=expected,path=str(base)))
    print(f'Verified {len(current)} files: {base}')
    return base

if __name__=='__main__':restore()
