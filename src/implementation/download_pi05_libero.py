"""Download public official checkpoint locally, verify GCS MD5, resume via Range."""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import google_crc32c


PREFIX='checkpoints/pi05_libero/'


def checksum(path,kind):
    h=hashlib.md5() if kind=='md5Hash' else google_crc32c.Checksum()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    url='https://storage.googleapis.com/storage/v1/b/openpi-assets/o?'+urllib.parse.urlencode({'prefix':PREFIX,'maxResults':1000})
    with urllib.request.urlopen(url,timeout=30) as r: listing=json.load(r)
    if listing.get('nextPageToken'): raise RuntimeError('Unexpected pagination')
    (a.out/'download_manifest.json').write_text(json.dumps(listing,indent=2))
    def fetch(item):
        rel=Path(item['name'][len(PREFIX):])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError(rel)
        dest=a.out/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        size=int(item['size']); kind='md5Hash' if 'md5Hash' in item else 'crc32c'; digest=item[kind]
        if dest.exists():
            if dest.stat().st_size==size and checksum(dest,kind)==digest:
                print('VERIFIED_EXISTING',rel,flush=True); return
            raise RuntimeError(f'Existing mismatching file: {dest}')
        partial=dest.with_name(dest.name+'.partial'); offset=partial.stat().st_size if partial.exists() else 0
        if offset>size: raise RuntimeError(f'Oversize partial {partial}')
        if offset<size:
            url='https://storage.googleapis.com/openpi-assets/'+urllib.parse.quote(item['name'],safe='/')
            request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=120) as r:
                if offset and r.status!=206: raise RuntimeError('Server ignored resume range')
                with partial.open('ab' if offset else 'wb') as f:
                    last=offset
                    while chunk:=r.read(8*1024*1024):
                        f.write(chunk); offset+=len(chunk)
                        if offset-last>=256*1024*1024:
                            print('PROGRESS',str(rel),offset,size,flush=True); last=offset
        if partial.stat().st_size!=size or checksum(partial,kind)!=digest:
            raise RuntimeError(f'Checksum mismatch: {partial}')
        os.replace(partial,dest); print('VERIFIED',str(rel),size,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,listing['items']))
    hashes=[]
    for item in listing['items']:
        rel=item['name'][len(PREFIX):]; h=hashlib.sha256()
        with (a.out/rel).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        hashes.append({'path':rel,'bytes':int(item['size']),'sha256':h.hexdigest()})
    (a.out/'transfer_sha256.json').write_text(json.dumps(hashes,indent=2))
    (a.out/'DOWNLOAD_VERIFIED.json').write_text(json.dumps({'status':'VERIFIED','objects':len(listing['items']),
        'bytes':sum(int(i['size']) for i in listing['items']),'source':'gs://openpi-assets/'+PREFIX},indent=2))
    print('DOWNLOAD_VERIFIED',flush=True)


if __name__=='__main__': main()
