"""Audit checkpoint key coverage, then test VAE reconstruction on an actual frame."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from safetensors import safe_open
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
from fastwam.models.wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter, wan_video_dit_from_diffusers, wan_video_dit_state_dict_converter


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    root=Path('__WAM2REPAIR_ROOT__')
    cfg=OmegaConf.load(root/'workspace/FastWAM/configs/model/fastwam_p027c_libero_ac7.yaml')
    with torch.device('meta'):
        model=WanVideoDiT(**OmegaConf.to_container(cfg.video_dit_config))
    expected={k:tuple(v.shape) for k,v in model.state_dict().items()}
    keys={}
    for path in sorted((root/'models/fastwam/Wan-AI/Wan2.2-TI2V-5B').glob('*.safetensors')):
        with safe_open(str(path),framework='pt',device='cpu') as f:
            keys.update({k:tuple(f.get_slice(k).get_shape()) for k in f.keys()})
    variants={'raw':keys,'native_converter':wan_video_dit_state_dict_converter(keys),'diffusers_converter':wan_video_dit_from_diffusers(keys)}
    report={'torch':torch.__version__,'coverage':{}}
    for name, candidate in variants.items():
        missing=sorted(set(expected)-set(candidate)); unexpected=sorted(set(candidate)-set(expected))
        mismatch=[k for k in expected.keys() & candidate.keys() if expected[k]!=candidate[k]]
        matched=[k for k in expected.keys() & candidate.keys() if expected[k]==candidate[k]]
        report['coverage'][name]={'matched_keys':len(matched),'expected_keys':len(expected),
            'matched_numel_fraction':sum(int(np.prod(expected[k])) for k in matched)/sum(int(np.prod(s)) for s in expected.values()),
            'missing':missing,'unexpected':unexpected,'shape_mismatch':mismatch}
    (out/'weight_coverage.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{x:y for x,y in v.items() if x not in ['missing','unexpected']} for k,v in report['coverage'].items()}),flush=True)
    torch.manual_seed(7)
    witness=torch.randn(8,8,device='cuda'); print('WITNESS_CUDA',float((witness@witness).sum()),flush=True)
    vae=WanVideoVAE38()
    path=root/'models/fastwam/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors'
    with safe_open(str(path),framework='pt',device='cpu') as f:
        weights={k:f.get_tensor(k) for k in f.keys()}
    converted=wan_video_vae_state_dict_converter(weights)
    incompat=vae.load_state_dict(converted,strict=False)
    report['vae_missing']=list(incompat.missing_keys); report['vae_unexpected']=list(incompat.unexpected_keys)
    if incompat.missing_keys or incompat.unexpected_keys:
        (out/'vae_report.json').write_text(json.dumps(report,indent=2))
        raise RuntimeError('VAE weight coverage failed')
    vae=vae.to(device='cuda',dtype=torch.bfloat16).eval()
    data=np.load(root/'results/policy-relevant-imagined-state-repair/e1/prepared_goal03.npz')
    rgb=np.concatenate([np.asarray(Image.fromarray(data['task0_'+cam]).resize((224,224))) for cam in ['primary','wrist']],axis=1)
    x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None,:,None].to('cuda',torch.bfloat16)/127.5-1
    with torch.inference_mode():
        z=vae.model.encode(x,vae.scale)
        y=vae.decode(z,device='cuda',tiled=False)
    restored=((y[0,:,0].float().clamp(-1,1)+1)*127.5).permute(1,2,0).cpu().numpy().astype(np.uint8)
    Image.fromarray(np.concatenate([rgb,restored],axis=0)).save(out/'vae_reconstruction.png')
    mse=float(np.mean((rgb.astype(float)-restored.astype(float))**2))
    report.update(vae_mse=mse,vae_psnr=float(10*np.log10(255**2/max(mse,1e-9))),latent_shape=list(z.shape),status='MEASURED')
    (out/'vae_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='coverage'}),flush=True)


if __name__=='__main__': main()
